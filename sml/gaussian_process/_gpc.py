# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax.numpy as jnp
from jax import grad
from jax.lax.linalg import cholesky
from jax.scipy.linalg import cho_solve, solve
from jax.scipy.special import erf, expit

from .kernels import RBF
from .ovo_ovr import OneVsRestClassifier

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
        max_iter_predict=100,
    ):
        self.kernel = kernel
        self.max_iter_predict = max_iter_predict
        self.poss = poss

    def fit(self, X, y):
        self._check_kernel()

        self.X_train_ = jnp.asarray(X)

        if self.poss == "sigmoid":
            self.approx_func = expit
        else:
            raise ValueError(
                f"Unsupported prior-likelihood function {self.poss}."
                "Please try the default dunction sigmoid."
            )

        self.y_train = y

        K = self.kernel_(self.X_train_)

        (
            self.pi_,
            self.W_sr_,
            self.L_,
        ) = self._posterior_mode(K)
        return self

    def predict(self, Xll):
        X = jnp.asarray(Xll)
        K_star = self.kernel_(self.X_train_, X)
        # f_star = K_star.T.dot(self.log_and_grad(self.f_, self.y_train))
        f_star = K_star.T.dot(self.y_train - self.pi_)

        return jnp.where(f_star > 0, 1, 0)

    def predict_proba(self, Xll):
        X = jnp.asarray(Xll)

        K_star = self.kernel_(self.X_train_, X)
        # f_star = K_star.T.dot(self.log_and_grad(self.f_, self.y_train))
        f_star = K_star.T.dot(self.y_train - self.pi_)
        v = solve(self.L_, self.W_sr_[:, jnp.newaxis] * K_star)
        # var_f_star = jnp.diag(self.kernel_(X)) - jnp.einsum("ij,ij->j", v, v)
        var_f_star = self.kernel_.diag(X) - jnp.einsum("ij,ij->j", v, v)

        alpha = 1 / (2 * var_f_star)
        gamma = LAMBDAS * f_star

        integrals = (
            jnp.sqrt(jnp.pi / alpha)
            * erf(gamma * jnp.sqrt(alpha / (alpha + LAMBDAS**2)))
            / (2 * jnp.sqrt(var_f_star * 2 * jnp.pi))
        )
        pi_star = (COEFS * integrals).sum(axis=0) + 0.5 * COEFS.sum()

        return jnp.vstack((1 - pi_star, pi_star)).T

    def _posterior_mode(self, K):
        # Based on Algorithm 3.1 of GPML
        f = jnp.zeros_like(
            self.y_train, dtype=jnp.float32
        )  # a warning is triggered if float64 is used

        for _ in range(self.max_iter_predict):
            # W = self.log_and_2grads_and_negtive(f, self.y_train)
            pi = self.approx_func(f)
            W = pi * (1 - pi)
            W_sqr = jnp.sqrt(W)
            W_sqr_K = W_sqr[:, jnp.newaxis] * K

            B = jnp.eye(W.shape[0]) + W_sqr_K * W_sqr
            L = cholesky(B, symmetrize_input=False)
            # b = W * f + self.log_and_grad(f, self.y_train)
            b = W * f + (self.y_train - pi)
            a = b - jnp.dot(
                W_sqr[:, jnp.newaxis] * cho_solve((L, True), jnp.eye(W.shape[0])),
                W_sqr_K.dot(b),
            )
            f = K.dot(a)

            # no early stop here...

        # for warm-start
        # self.f_cached = f
        return pi, W_sqr, L

    def _check_kernel(self):
        if self.kernel is None:  # Use an RBF kernel as default
            self.kernel_ = RBF()
        else:
            raise NotImplemented("Only RBF kernel is supported now.")


class GaussianProcessClassifier:
    """Gaussian process classification (GPC) based on Laplace approximation.

    The implementation is based on Algorithm 3.1, 3.2 from [RW2006]_.

    Internally, the Laplace approximation is used for approximating the
    non-Gaussian posterior by a Gaussian.

    Currently, the implementation is restricted to using the logistic link
    function. For multi-class classification, several binary one-versus rest
    classifiers are fitted. Note that this class thus does not implement
    a true multi-class Laplace approximation.

    IMPORTANT NOTES:
      1. Current implementations will not optimize the parameters of kernel during training.
      2. ONLY RBF kernel is supported now.
      3. ONLY one_vs_rest mode is supported for multi-class tasks. (You should pre-process your label to 0,1,2... like)
      4. You MUST specify n_classes_ explicitly, because we can not do data inspections under MPC.

    Parameters
    ----------
    kernel : kernel instance, default=None
        The kernel specifying the covariance function of the GP. If None is
        passed, the kernel "1.0 * RBF(1.0)" is used as default.
        Only RBF kernel is supported now.

    max_iter_predict : int, default=20
        The maximum number of iterations in Newton's method for approximating
        the posterior during predict. Smaller values will reduce computation
        time at the cost of worse results.

    multi_class : 'one_vs_rest', default='one_vs_rest'
        Specifies how multi-class classification problems are handled.
        One binary Gaussian process classifier is fitted for each class, which
        is trained to separate this class from the rest.

    poss : "sigmoid", allable or None, default="sigmoid", the predefined
        likelihood function which computes the possibility of the predict output
        w.r.t. f value.

    n_classes_ : int
        The number of classes in the training data.

    References
    ----------
    .. [RW2006] `Carl E. Rasmussen and Christopher K.I. Williams,
       "Gaussian Processes for Machine Learning",
       MIT Press 2006 <https://www.gaussianprocess.org/gpml/chapters/RW.pdf>`_
    """

    def __init__(
        self,
        kernel=None,
        *,
        poss="sigmoid",
        max_iter_predict=20,
        multi_class="one_vs_rest",
        n_classes_=2,
    ):
        self.kernel = kernel
        self.max_iter_predict = max_iter_predict
        self.multi_class = multi_class
        self.poss = poss

        self.n_classes_ = n_classes_
        self.base_estimator_ = None

        assert (
            n_classes_ > 1
        ), "GaussianProcessClassifier requires 2 or more distinct classes"

    def fit(self, X, y):
        """Fit Gaussian process classification model.

        Parameters
        ----------
        X : jax numpy array (n_samples, n_features) of object
            Feature vectors of training data.

        y : jax numpy array (n_samples,) Target values,
        must be preprocessed to 0, 1, 2, ...

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        self.y_train = jnp.array(y)

        self.base_estimator_ = _BinaryGaussianProcessClassifierLaplace(
            kernel=self.kernel,
            max_iter_predict=self.max_iter_predict,
            poss=self.poss,
        )

        if self.n_classes_ > 2:
            if self.multi_class == "one_vs_rest":
                self.base_estimator_ = OneVsRestClassifier(
                    _BinaryGaussianProcessClassifierLaplace,
                    self.n_classes_,
                    kernel=self.kernel,
                    max_iter_predict=self.max_iter_predict,
                    poss=self.poss,
                )
            elif self.multi_class == "one_vs_one":
                raise ValueError("one_vs_one classifier is not supported")
            else:
                raise ValueError("Unknown multi-class mode %s" % self.multi_class)

        self.X = jnp.array(X)
        self.base_estimator_.fit(self.X, self.y_train)

        return self

    def predict(self, X):
        """Perform classification on an array of test vectors X.

        Parameters
        ----------
        X : jax numpy array (n_samples, n_features) of object
            Query points where the GP is evaluated for classification.

        Returns
        -------
        C : jax numpy array (n_samples,)
            Predicted target values for X.
        """
        self.check_is_fitted()
        return self.base_estimator_.predict(X)

    def predict_proba(self, X):
        """Return probability estimates for the test vector X.

        Parameters
        ----------
        X : jax numpy array (n_samples, n_features) of object
            Query points where the GP is evaluated for classification.

        Returns
        -------
        C : jax numpy array (n_samples, n_classes)
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order.
        """
        self.check_is_fitted()
        return self.base_estimator_.predict_proba(X)

    def check_is_fitted(self):
        """Perform is_fitted validation for estimator."""
        assert self.base_estimator_ is not None, "Model should be fitted first."
