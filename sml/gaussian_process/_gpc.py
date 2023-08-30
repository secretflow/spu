import copy
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit
from jax.lax.linalg import cholesky
from jax.scipy.linalg import cho_solve, solve
from jax.scipy.special import erf, expit

sys.path.append(os.path.join(os.path.dirname(__file__), "./"))
from kernels import RBF
from ovo_ovr import OneVsOneClassifier, OneVsRestClassifier

LAMBDAS = jnp.array([0.41, 0.4, 0.37, 0.44, 0.39])[:, jnp.newaxis]
COEFS = jnp.array(
    [-1854.8214151, 3516.89893646, 221.29346712, 128.12323805, -2010.49422654]
)[:, jnp.newaxis]


class _BinaryGaussianProcessClassifierLaplace:
    def __init__(
        self,
        kernel=None,
        *,
        poss="sigmoid",
        optimizer="lbfgs",
        n_restarts_optimizer=0,
        max_iter_predict=100,
        warm_start=False,
        random_state=None,
        copy_X_train=True,
    ):
        self.kernel = kernel
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.max_iter_predict = max_iter_predict
        self.warm_start = warm_start
        self.random_state = random_state
        self.poss = poss
        self.copy_X_train = (copy_X_train,)

    def fit(self, X, y):
        self._check_kernal()

        self.X_train_ = jnp.asarray(X)

        if self.poss == "sigmoid":
            self.approx_func = expit
        else:
            raise ValueError(
                f"Unsupported prior-likelihood function {self.poss}. Please try the default option which is sigmoid"
            )

        # Encode class labels and check that it is a binary classification (self.classes_, self.y_train_ is a np.ndarrays for now, np.unique ias not permitted in spu)

        self.y_train = y

        # self.y_train = 2 * (self.y_train - 0.5)  # turn y_train into a -1 and 1 array

        K = self.kernel_(self.X_train_)
        _, self.f_ = self._posterior_mode(K, return_temporaries=True)
        return self.f_

    # def log_and_grad(self, f, y_train):
    #     _tmp = lambda f, y_train: jnp.sum(self.approx_func(y_train*f))
    #     return grad(_tmp)(f, y_train)/self.approx_func(y_train*f)

    # def log_and_2grads_and_negtive(self, f, y_train):
    #     _tmp = lambda f, y: jnp.sum(self.log_and_grad(f, y))
    #     return -grad(_tmp)(f, y_train)

    # def log_and_3grads(self, f, y_train):
    #     _tmp = lambda f, y_train: jnp.sum(-self.log_and_2grads_and_negtive(f, y_train))
    #     return grad(_tmp)(f, y_train)

    def predict(self, Xll):
        X = jnp.asarray(Xll)
        K_star = self.kernel_(self.X_train_, X)
        # f_star = K_star.T.dot(self.log_and_grad(self.f_, self.y_train))
        f_star = K_star.T.dot(self.y_train - self.approx_func(self.f_))

        return jnp.where(f_star > 0, 1, 0)

    def predict_proba(self, Xll):
        X = jnp.asarray(Xll)
        K = self.kernel_(self.X_train_)
        # Based on Algorithm 3.2 of GPML

        # W = self.log_and_2grads_and_negtive(self.f_, self.y_train)
        pi = self.approx_func(self.f_)
        W = pi * (1 - pi)

        W_sqr = jnp.sqrt(W)
        W_sqr_K = W_sqr[:, jnp.newaxis] * K
        B = jnp.eye(W.shape[0]) + W_sqr_K * W_sqr
        L = cholesky(B)

        K_star = self.kernel_(self.X_train_, X)
        # f_star = K_star.T.dot(self.log_and_grad(self.f_, self.y_train))
        f_star = K_star.T.dot(self.y_train - pi)
        v = solve(L, W_sqr[:, jnp.newaxis] * K_star)
        var_f_star = jnp.diag(self.kernel_(X)) - jnp.einsum("ij,ij->j", v, v)

        alpha = 1 / (2 * var_f_star)
        gamma = LAMBDAS * f_star
        integrals = (
            jnp.sqrt(jnp.pi / alpha)
            * erf(gamma * jnp.sqrt(alpha / (alpha + LAMBDAS**2)))
            / (2 * jnp.sqrt(var_f_star * 2 * jnp.pi))
        )
        pi_star = (COEFS * integrals).sum(axis=0) + 0.5 * COEFS.sum()

        return jnp.vstack((1 - pi_star, pi_star)).T

    def _posterior_mode(self, K, return_temporaries=False):
        # Based on Algorithm 3.1 of GPML
        if (
            self.warm_start
            and hasattr(self, "f_cached")
            and self.f_cached.shape == self.y_train_.shape
        ):
            f = self.f_cashed
        else:
            f = jnp.zeros_like(
                self.y_train, dtype=jnp.float32
            )  # a warning is triggered if float64 is used

        log_marginal_likelihood = -jnp.inf

        for _ in range(self.max_iter_predict):
            # W = self.log_and_2grads_and_negtive(f, self.y_train)
            pi = self.approx_func(f)
            W = pi * (1 - pi)
            W_sqr = jnp.sqrt(W)
            W_sqr_K = W_sqr[:, jnp.newaxis] * K

            B = jnp.eye(W.shape[0]) + W_sqr_K * W_sqr
            L = cholesky(B)
            # b = W * f + self.log_and_grad(f, self.y_train)
            b = W * f + (self.y_train - pi)
            a = b - jnp.dot(
                W_sqr[:, jnp.newaxis] * cho_solve((L, True), jnp.eye(W.shape[0])),
                W_sqr_K.dot(b),
            )
            f = K.dot(a)

            lml = (
                -0.5 * a.T.dot(f)
                + jnp.log(self.approx_func(self.y_train * f)).sum()
                - jnp.log(jnp.diag(L)).sum()
            )

            # if (lml - log_marginal_likelihood) < 1e-10:
            #     log_marginal_likelihood = lml
            #     break
            log_marginal_likelihood = lml

        self.f_cached = f  # for warm-start

        if return_temporaries:
            return log_marginal_likelihood, f
        else:
            return log_marginal_likelihood

    def _check_optimizer(self):
        if self.optimizer != "lbfgs":
            raise ValueError(
                f"Unsupported optimizer{self.optimizer}. Please try the default option which is lbfgs"
            )

    def _check_kernal(self):
        if self.kernel is None:  # Use an RBF kernel as default
            self.kernel_ = RBF()
        else:
            self.kernel_ = copy.deepcopy(self.kernel)

class GaussianProcessClassifier:
    def __init__(
        self,
        kernel=None,
        *,
        poss="sigmoid",
        optimizer="lbfgs",
        n_restarts_optimizer=5,
        max_iter_predict=100,
        warm_start=False,
        copy_X_train=True,
        random_state=None,
        multi_class="one_vs_rest",
        n_jobs=None,
    ):
        self.kernel = kernel
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.max_iter_predict = max_iter_predict
        self.warm_start = warm_start
        self.copy_X_train = copy_X_train
        self.random_state = random_state
        self.multi_class = multi_class
        self.n_jobs = n_jobs
        self.poss = poss

    def fit(self, X, y, n_classes_):
        self.n_classes_ = n_classes_
        # Encode class labels and check that it is a binary classification (self.classes_, self.y_train_ is a np.ndarrays for now, jnp.unique doesn't allow string transformation)
        self.y = y
        # y0 = jnp.array(y)
        # unique_labels = set(y)
        # self.n_classes_ = []
        # self.y_train = jnp.zeros(y0.shape, dtype=int)
        # for index, label in enumerate(unique_labels):
        #     self.n_classes_.append(label)
        #     self.y_train = jnp.where(y0 == label, index, self.y_train)

        self.y_train = jnp.array(y)
        # self.n_classes_ = jnp.max(self.y_train) + 1

        if self.n_classes_ == 1:
            raise ValueError(
                "GaussianProcessClassifier requires 2 or more "
                "distinct classes; got 1 class (only class %s "
                "is present)" % self.n_classes_[0]
            )

        self.base_estimator_ = _BinaryGaussianProcessClassifierLaplace(
            kernel=self.kernel,
            optimizer=self.optimizer,
            n_restarts_optimizer=self.n_restarts_optimizer,
            max_iter_predict=self.max_iter_predict,
            warm_start=self.warm_start,
            copy_X_train=self.copy_X_train,
            random_state=self.random_state,
            poss=self.poss,
        )

        if self.n_classes_ > 2:
            if self.multi_class == "one_vs_rest":
                self.base_estimator_ = OneVsRestClassifier(
                    self.base_estimator_, self.n_classes_, n_jobs=self.n_jobs
                )
            elif self.multi_class == "one_vs_one":
                raise ValueError("one_vs_one classifier is not supported yet")
                self.base_estimator_ = OneVsOneClassifier(
                    self.base_estimator_, self.n_classes_, n_jobs=self.n_jobs
                )
            else:
                raise ValueError("Unknown multi-class mode %s" % self.multi_class)

        self.base_estimator_.fit(X, self.y_train)

    def predict(self, X):
        a = self.base_estimator_.predict(X)
        # result = jnp.array(a)
        # for index, label in enumerate(self.n_classes_):
        #     result = jnp.where(index == a, label, result)
        return a
