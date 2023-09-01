import jax.numpy as jnp
from jax import vmap
from jax.scipy.special import erf, expit
from kernels import RBF
from jax.lax.linalg import cholesky
from jax.scipy.linalg import cho_solve, solve

LAMBDAS = jnp.array([0.41, 0.4, 0.37, 0.44, 0.39])[:, jnp.newaxis]
COEFS = jnp.array(
    [-1854.8214151, 3516.89893646, 221.29346712, 128.12323805, -2010.49422654]
)[:, jnp.newaxis]

class OneVsRestClassifier:
    def __init__(self, estimator, n_classes):
        self.estimator = estimator
        self.classes_ = n_classes

    def fit(self, X, y):

        self.estimator.approx_func = expit
        self.estimator.X_train_ = jnp.array(X)

        if self.estimator.kernel is None:  # Use an RBF kernel as default
            self.estimator.kernel_ = RBF()
        else:
            self.estimator.kernel_ = self.estimator.kernel

        self.K = self.estimator.kernel_(X)

        self.y_binary = jnp.array(
            [jnp.where(y == i, 0, 1) for i in range(self.classes_)]
        )

        self.fs_ = vmap(self._posterior_mode, in_axes=(None, 0, None))(X, self.y_binary, self.K)
        self.f_cached = self.fs_

    def predict(self, X_test):
        X = jnp.array(X_test)
        K_star = self.estimator.kernel_(self.estimator.X_train_, X)
        diag_K_Xtest = jnp.diag(self.estimator.kernel_(X))
        maxima = vmap(self.predict_proba_oneclass, in_axes=(0, None, 0, None, None, None))(
            self.y_binary, X_test, self.fs_, self.K, K_star, diag_K_Xtest
        )
        return maxima.argmax(axis=0)

    def predict_proba(self, X_test):
        X = jnp.array(X_test)
        K_star = self.estimator.kernel_(self.estimator.X_train_, X)
        diag_K_Xtest = jnp.diag(self.estimator.kernel_(X))
        maxima = vmap(self.predict_proba_oneclass, in_axes=(0, None, 0, None, None, None))(
            self.y_binary, X_test, self.fs_, self.K, K_star, diag_K_Xtest
        )
        maxima = maxima / jnp.sum(maxima, axis=0)
        return maxima.T

    def predict_proba_oneclass(self, y_binary, X_test, f_, K, K_star, diag_K_Xtest):
        X = jnp.asarray(X_test)
        # K = self.estimator.kernel_(self.estimator.X_train_)

        pi = self.estimator.approx_func(f_)
        W = pi * (1 - pi)

        W_sqr = jnp.sqrt(W)
        W_sqr_K = W_sqr[:, jnp.newaxis] * K
        B = jnp.eye(W.shape[0]) + W_sqr_K * W_sqr
        L = cholesky(B)

        # K_star = self.estimator.kernel_(self.estimator.X_train_, X)
        # f_star = K_star.T.dot(self.log_and_grad(self.f_, self.y_train))
        f_star = K_star.T.dot(y_binary - pi)
        v = solve(L, W_sqr[:, jnp.newaxis] * K_star)
        var_f_star = diag_K_Xtest - jnp.einsum("ij,ij->j", v, v)

        alpha = 1 / (2 * var_f_star)
        gamma = LAMBDAS * f_star
        integrals = (
            jnp.sqrt(jnp.pi / alpha)
            * erf(gamma * jnp.sqrt(alpha / (alpha + LAMBDAS**2)))
            / (2 * jnp.sqrt(var_f_star * 2 * jnp.pi))
        )
        pi_star = (COEFS * integrals).sum(axis=0) + 0.5 * COEFS.sum()

        return 1 - pi_star

    def _posterior_mode(self, X, y_binary, K):
        # K = self.estimator.kernel_(X)
        # Based on Algorithm 3.1 of GPML
        f = jnp.zeros_like(
            y_binary, dtype=jnp.float32
        )  # a warning is triggered if float64 is used

        for _ in range(self.estimator.max_iter_predict):
            # W = self.log_and_2grads_and_negtive(f, self.y_train)
            pi =  self.estimator.approx_func(f)
            W = pi * (1 - pi)
            W_sqr = jnp.sqrt(W)
            W_sqr_K = W_sqr[:, jnp.newaxis] * K

            B = jnp.eye(W.shape[0]) + W_sqr_K * W_sqr
            L = cholesky(B)
            # b = W * f + self.log_and_grad(f, self.y_train)
            b = W * f + (y_binary - pi)
            a = b - jnp.dot(
                W_sqr[:, jnp.newaxis] * cho_solve((L, True), jnp.eye(W.shape[0])),
                W_sqr_K.dot(b),
            )
            f = K.dot(a)

        return f
