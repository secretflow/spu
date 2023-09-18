import jax
from jax import random
import jax.numpy as jnp
from utils.solver import *
from utils.loss import *
from utils.link import *
import warnings
import os

DEBUG = 0


# Define the _GeneralizedLinearRegressor class using JAX
class _GeneralizedLinearRegressor:
    def __init__(
        self,
        fit_intercept=True,  # Whether to fit the intercept term, default is True
        alpha=0,  # L2 regularization strength, default is 0 (no regularization)
        solver="newton-cholesky",  # Optimization algorithm, default is Newton-Cholesky
        max_iter=20,  # Maximum number of iterations, default is 20
        warm_start=False,  # Whether to use warm start, default is False
        n_threads=None,  # Deprecated parameter (no longer used)
        tol=None,  # Deprecated parameter (no longer used)
        verbose=0,  # Level of verbosity, default is 0 (no output)
    ):
        """
        Initialize the generalized linear regression model.

        Parameters:
        ----------
        fit_intercept : bool, optional
            Whether to fit the intercept term, default is True.
        alpha : float, optional
            L2 regularization strength, default is 0 (no regularization).
        solver : str, optional
            Optimization algorithm, default is Newton-Cholesky. Supported values are "lbfgs" or "newton-cholesky".
        max_iter : int, optional
            Maximum number of iterations, default is 20.
        warm_start : bool, optional
            Whether to use warm start, default is False.
        n_threads : int, optional
            Number of threads for parallel computation, default is 1.
            If set to 0, it will detect all available CPUs for parallel computation.
        tol : deprecated
            This parameter is deprecated and no longer used. It was used to set an early stop threshold.
        verbose : int, optional
            Level of verbosity, default is 0 (no output).

        """
        self.l2_reg_strength = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.max_iter = max_iter
        self.warm_start = warm_start
        self.verbose = verbose
        if n_threads:
            warnings.warn(
                "SPU does not need n_threads.",
                category=DeprecationWarning,
                stacklevel=2,
            )
        if warm_start:
            warnings.warn("Using minibatch in the second optimizer may cause problems.")
        if tol:
            warnings.warn(
                "SPU does not support early stop.",
                category=DeprecationWarning,
                stacklevel=2,
            )

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = jnp.ones(y.shape[0])
        assert sample_weight.shape == y.shape

        self._check_solver_support()
        self.loss_model = self._get_loss()
        self.link_model = self._get_link()
        self.loss_model.get_sampleweight(sample_weight)
        # y=self.link_model.inverse(y)
        if not self.warm_start or not hasattr(self, "coef_"):
            self.coef_ = None
        if self.solver == "lbfgs":
            warnings.warn(
                "LBFGS algorithm cannot be accurately implemented on SPU platform, only approximate implementation is available.",
                UserWarning,
            )
            self._fit_lbfgs(X, y)
        elif self.solver == "newton-cholesky":
            self._fit_newton_cholesky(X, y)
        else:
            raise ValueError(f"Invalid solver={self.solver}.")

    def _get_loss(self):
        return HalfSquaredLoss()  # Choose the loss function as needed

    def _get_link(self):
        return IdentityLink()

    def _fit_newton_cholesky(self, X, y):
        # Use the NewtonCholeskySolver class to implement the Newton-Cholesky optimization algorithm
        solver = NewtonCholeskySolver(
            loss_model=self.loss_model,
            l2_reg_strength=self.l2_reg_strength,
            max_iter=self.max_iter,
            verbose=self.verbose,
            link=self.link_model,
            coef=self.coef_,
        )
        self.coef_ = solver.solve(X, y)

    def _fit_lbfgs(self, X, y):
        # Use the LBFGSSolver class to implement the Newton-Cholesky optimization algorithm
        solver = LBFGSSolver(
            loss_model=self.loss_model,
            max_iter=self.max_iter,
            l2_reg_strength=self.l2_reg_strength,
            verbose=self.verbose,
            link=self.link_model,
            coef=self.coef_,
        )
        self.coef_ = solver.solve(X, y)

    def predict(self, X):
        # Calculate the predictions
        if self.fit_intercept:
            X = jnp.hstack([jnp.ones((X.shape[0], 1)), X])  # Add the intercept term
        y_pred = self.link_model.inverse(X @ self.coef_)
        return y_pred

    def score(self, X, y, sample_weight=None):
        """
        D^2 is the evaluation metric for the generalized linear regression model.
        """

        # Calculate the model's predictions
        prediction = self.predict(X)
        squared_error = lambda y_true, prediction: jnp.mean((y_true - prediction) ** 2)
        # Calculate the model's deviance
        deviance = squared_error(y_true=y, prediction=prediction)
        # Calculate the null deviance
        deviance_null = squared_error(
            y_true=y, prediction=jnp.tile(jnp.average(y), y.shape[0])
        )
        # Calculate D^2
        d2 = 1 - (deviance) / (deviance_null)
        return d2

    def _check_solver_support(self):
        supported_solvers = [
            "lbfgs",
            "newton-cholesky",
        ]  # List of supported optimization algorithms
        if self.solver not in supported_solvers:
            raise ValueError(
                f"Invalid solver={self.solver}. Supported solvers are {supported_solvers}."
            )


# The PoissonRegressor class represents a generalized linear model with Poisson distribution using JAX.
class PoissonRegressor(_GeneralizedLinearRegressor):
    """Generalized linear model with Poisson distribution, implemented using JAX.

    This regressor uses the 'log' link function.
    """

    def _get_loss(self):
        return HalfPoissonLoss()

    def _get_link(self):
        return LogLink()


# The GammaRegressor class represents a generalized linear model with Gamma distribution using JAX.
class GammaRegressor(_GeneralizedLinearRegressor):
    def _get_loss(self):
        return HalfGammaLoss()

    def _get_link(self):
        return LogLink()


# The TweedieRegressor class represents a generalized linear model with Tweedie distribution using JAX.
class TweedieRegressor(_GeneralizedLinearRegressor):
    def __init__(
        self,
        power=0.5,
    ):
        super().__init__()
        # Ensure that the power is within the valid range for the Tweedie distribution
        assert power >= 0 and power <= 3
        self.power = power

    def _get_loss(self):
        return HalfTweedieLoss(
            self.power,
        )

    def _get_link(self):
        if self.power > 0:
            return LogLink()
        else:
            return IdentityLink()
