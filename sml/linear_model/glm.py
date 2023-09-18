# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings

import jax.numpy as jnp
from sml.linear_model.utils.link import *
from sml.linear_model.utils.loss import *
from sml.linear_model.utils.solver import *


class _GeneralizedLinearRegressor:
    def __init__(
        self,
        fit_intercept=True,
        alpha=0,
        solver="newton-cholesky",
        max_iter=20,
        warm_start=False,
        tol=None,
    ):
        """
        GLMs based on a reproductive Exponential Dispersion Model (EDM) aim at fitting and
        predicting the mean of the target y as y_pred=h(X*w) with coefficients w.

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
        tol : float, optional,
            Stopping criterion. For the lbfgs solver, the iteration will stop when ``max{|g_j|, j = 1, ..., d} <= tol``
            where ``g_j`` is the j-th component of the gradient (derivative) of the objective function.
        """
        self.l2_reg_strength = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.max_iter = max_iter
        self.warm_start = warm_start
        self.tol = tol

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = jnp.ones(y.shape[0])
        assert sample_weight.shape == y.shape

        self._check_solver_support()
        self.loss_model = self._get_loss()
        self.link_model = self._get_link()
        self.loss_model.set_sample_weight(sample_weight)

        if not self.warm_start or not hasattr(self, "coef_"):
            self.coef_ = None
        if self.solver == "lbfgs":
            warnings.warn(
                "LBFGS algorithm will be very costly, because of the dummy early stop schema",
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
        # todo: current only implement D2 score for square loss and normal dist.
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
        power=1.5,
        fit_intercept=True,
        alpha=0,
        solver="newton-cholesky",
        max_iter=20,
        warm_start=False,
        tol=None,
    ):
        super().__init__(
            fit_intercept=fit_intercept,
            alpha=alpha,
            solver=solver,
            max_iter=max_iter,
            warm_start=warm_start,
            tol=tol,
        )
        # Ensure that the power is within the valid range for the Tweedie distribution
        assert power > 0 and power <= 3
        self.power = power

    def _get_loss(self):
        return HalfTweedieLoss(
            self.power,
        )

    def _get_link(self):
        return LogLink()
