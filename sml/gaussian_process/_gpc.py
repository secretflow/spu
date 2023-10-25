import os
import sys

import jax
import jax.numpy as jnp
from jax import grad
from jax.lax.linalg import cholesky
from jax.scipy.linalg import cho_solve, solve
from jax.scipy.special import erf, expit

sys.path.append(os.path.join(os.path.dirname(__file__), "./"))
from kernels import RBF
from ovo_ovr import OneVsRestClassifier

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
        self._check_kernal()

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
        self.f_ = self._posterior_mode(K)
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
            L = cholesky(B)
            # b = W * f + self.log_and_grad(f, self.y_train)
            b = W * f + (self.y_train - pi)
            a = b - jnp.dot(
                W_sqr[:, jnp.newaxis] * cho_solve((L, True), jnp.eye(W.shape[0])),
                W_sqr_K.dot(b),
            )
            f = K.dot(a)

        self.f_cached = f  # for warm-start
        return f

    def _check_kernal(self):
        if self.kernel is None:  # Use an RBF kernel as default
            self.kernel_ = RBF()
        else:
            self.kernel_ = self.kernel


class GaussianProcessClassifier:
    """Gaussian process classification (GPC) based on Laplace approximation.

    The implementation is based on Algorithm 3.1, 3.2, and 5.1 from [RW2006]_.

    Internally, the Laplace approximation is used for approximating the
    non-Gaussian posterior by a Gaussian.

    Currently, the implementation is restricted to using the logistic link
    function. For multi-class classification, several binary one-versus rest
    classifiers are fitted. Note that this class thus does not implement
    a true multi-class Laplace approximation.

    Read more in the :ref:`User Guide <gaussian_process>`.

    Parameters
    ----------
    kernel : kernel instance, default=None
        The kernel specifying the covariance function of the GP. If None is
        passed, the kernel "1.0 * RBF(1.0)" is used as default. Note that
        the kernel's hyperparameters are optimized during fitting. Also kernel
        cannot be a `CompoundKernel`.

    max_iter_predict : int, default=100
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

    Attributes
    ----------
    base_estimator_ : ``Estimator`` instance
        The estimator instance that defines the likelihood function
        using the observed data.

    kernel_ : kernel instance
        The kernel used for prediction. In case of binary classification,
        the structure of the kernel is the same as the one passed as parameter
        but with optimized hyperparameters. In case of multi-class
        classification, a CompoundKernel is returned which consists of the
        different kernels used in the one-versus-rest classifiers.

    n_classes_ : int
        The number of classes in the training data

    References
    ----------
    .. [RW2006] `Carl E. Rasmussen and Christopher K.I. Williams,
       "Gaussian Processes for Machine Learning",
       MIT Press 2006 <https://www.gaussianprocess.org/gpml/chapters/RW.pdf>`_

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.gaussian_process import GaussianProcessClassifier
    >>> from sklearn.gaussian_process.kernels import RBF
    >>> X, y = load_iris(return_X_y=True)
    >>> kernel = 1.0 * RBF(1.0)
    >>> gpc = GaussianProcessClassifier(kernel=kernel,
    ...         random_state=0).fit(X, y, 3)
    >>> gpc.predict_proba(X[:2,:])
    array([[0.83548752, 0.03228706, 0.13222543],
           [0.79064206, 0.06525643, 0.14410151]])
    """

    def __init__(
        self,
        kernel=None,
        *,
        poss="sigmoid",
        max_iter_predict=100,
        multi_class="one_vs_rest",
    ):
        self.kernel = kernel
        self.max_iter_predict = max_iter_predict
        self.multi_class = multi_class
        self.poss = poss

    def fit(self, X, y, n_classes_):
        """Fit Gaussian process classification model.

        Parameters
        ----------
        X : jax numpy array (n_samples, n_features) of object
            Feature vectors of training data.

        y : jax numpy array (n_samples,) Target values, 
        must be preprocessed to 0, 1, 2, ...

        n_classes_ : The number of classes in the training data

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        self.n_classes_ = n_classes_
        self.y_train = jnp.array(y)

        if self.n_classes_ == 1:
            raise ValueError(
                "GaussianProcessClassifier requires 2 or more "
                "distinct classes; got 1 class (only class %s "
                "is present)" % self.n_classes_[0]
            )

        self.base_estimator_ = _BinaryGaussianProcessClassifierLaplace(
            kernel=self.kernel,
            max_iter_predict=self.max_iter_predict,
            poss=self.poss,
        )

        if self.n_classes_ > 2:
            if self.multi_class == "one_vs_rest":
                self.base_estimator_ = OneVsRestClassifier(
                    self.base_estimator_, self.n_classes_
                )
            elif self.multi_class == "one_vs_one":
                raise ValueError("one_vs_one classifier is not supported")
            else:
                raise ValueError("Unknown multi-class mode %s" % self.multi_class)

        self.X = jnp.array(X)
        self.base_estimator_.fit(self.X, self.y_train)

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
        """Perform is_fitted validation for estimator.

        Checks if the estimator is fitted by verifying the presence of
        fitted attribute self.n_classes_ and otherwise
        raises a NotFittedError with the given message.

        Raises
        ------
        Exception
            If the attribute is not found.
        """
        try: 
            self.n_classes_
        except: 
            raise Exception('Model is not fitted yet')
