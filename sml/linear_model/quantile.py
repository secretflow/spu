# Copyright 2024 Ant Group Co., Ltd.
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

import jax
import jax.numpy as jnp
import pandas as pd
from jax import grad

from sml.linear_model.utils._linprog_simplex import _linprog_simplex


class QuantileRegressor:
    """
    Initialize the quantile regression model.
    Parameters
    ----------
    quantile : float, default=0.5
        The quantile to be predicted. Must be between 0 and 1.
        A quantile of 0.5 corresponds to the median (50th percentile).
    alpha : float, default=1.0
        Regularization strength; must be a positive float.
        Larger values specify stronger regularization, reducing model complexity.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for the model.
        If False, no intercept will be used in calculations, meaning the model will
        assume that the data is already centered.
    lr : float, default=0.01
        Learning rate for the optimization process. This controls the size of
        the steps taken in each iteration towards minimizing the objective function.
    max_iter : int, default=1000
        The maximum number of iterations for the optimization algorithm.
        This controls how long the model will continue to update the weights
        before stopping.
    max_val : float, default=1e10
        The maximum value allowed for the model parameters.
    Attributes
    ----------
    coef_ : array-like of shape (n_features,)
        The coefficients (weights) assigned to the input features. These will be
        learned during model fitting.
    intercept_ : float
        The intercept (bias) term. If `fit_intercept=True`, this will be
        learned during model fitting.
    """

    def __init__(
        self,
        quantile=0.5,
        alpha=1.0,
        fit_intercept=True,
        lr=0.01,
        max_iter=1000,
        max_val=1e10,
    ):
        self.quantile = quantile
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.lr = lr
        self.max_iter = max_iter
        self.max_val = max_val

        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y, sample_weight=None):
        """
        Fit the quantile regression model using linear programming.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), optional
            Individual weights for each sample. If not provided, all samples
            are assumed to have equal weight.
        Returns
        -------
        self : object
            Returns an instance of self.
        Steps:
        1. Determine the number of parameters (`n_params`), accounting for the intercept if needed.
        2. Define the objective function `c`, incorporating both the L1 regularization and the pinball loss.
        3. Set up the equality constraint matrix `A_eq` and vector `b_eq` based on the input data `X` and `y`.
        4. Solve the linear programming problem using `_linprog_simplex`.
        5. Extract the model parameters (intercept and coefficients) from the solution.
        """
        n_samples, n_features = X.shape
        n_params = n_features

        if sample_weight is None:
            sample_weight = jnp.ones((n_samples,))

        if self.fit_intercept:
            n_params += 1

        alpha = jnp.sum(sample_weight) * self.alpha

        # After rescaling alpha, the minimization problem is
        #     min sum(pinball loss) + alpha * L1
        # Use linear programming formulation of quantile regression
        #     min_x c x
        #           A_eq x = b_eq
        #                0 <= x
        # x = (s0, s, t0, t, u, v) = slack variables >= 0
        # intercept = s0 - t0
        # coef = s - t
        # c = (0, alpha * 1_p, 0, alpha * 1_p, quantile * 1_n, (1-quantile) * 1_n)
        # residual = y - X@coef - intercept = u - v
        # A_eq = (1_n, X, -1_n, -X, diag(1_n), -diag(1_n))
        # b_eq = y
        # p = n_features
        # n = n_samples
        # 1_n = vector of length n with entries equal one
        # see https://stats.stackexchange.com/questions/384909/
        c = jnp.concatenate(
            [
                jnp.full(2 * n_params, fill_value=alpha),
                sample_weight * self.quantile,
                sample_weight * (1 - self.quantile),
            ]
        )

        if self.fit_intercept:
            c = c.at[0].set(0)
            c = c.at[n_params].set(0)

        eye = jnp.eye(n_samples)
        if self.fit_intercept:
            ones = jnp.ones((n_samples, 1))
            A = jnp.concatenate([ones, X, -ones, -X, eye, -eye], axis=1)
        else:
            A = jnp.concatenate([X, -X, eye, -eye], axis=1)

        b = y

        result = _linprog_simplex(
            c, A, b, maxiter=self.max_iter, tol=1e-3, max_val=self.max_val
        )

        solution = result

        params = solution[:n_params] - solution[n_params : 2 * n_params]

        if self.fit_intercept:
            self.coef_ = params[1:]
            self.intercept_ = params[0]
        else:
            self.coef_ = params
            self.intercept_ = 0.0
        return self

    def predict(self, X):
        """
        Predict target values using the fitted quantile regression model.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data for which predictions are to be made.
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted target values.
        Notes
        -----
        The predict method computes the predicted target values using the model's
        learned coefficients and intercept (if fit_intercept=True).
        - If the model includes an intercept, a column of ones is added to the input data `X` to account
        for the intercept in the linear combination.
        - The method then computes the dot product between the modified `X` and the stacked vector of
        intercept and coefficients.
        - If there is no intercept, the method simply computes the dot product between `X` and the coefficients.
        """

        assert (
            self.coef_ is not None and self.intercept_ is not None
        ), "Model has not been fitted yet. Please fit the model before predicting."

        n_features = len(self.coef_)
        assert X.shape[1] == n_features, (
            f"Input X must have {n_features} features, "
            f"but got {X.shape[1]} features instead."
        )

        return jnp.dot(X, self.coef_) + self.intercept_
