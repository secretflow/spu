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

import numbers
import warnings
from warnings import warn

import jax
import jax.numpy as jnp
import pandas as pd
from jax import grad

from sml.linear_model.utils.linprog import _linprog_simplex


class QuantileRegressor:

    def __init__(
        self, quantile=0.5, alpha=1.0, fit_intercept=True, lr=0.01, max_iter=1000
    ):
        self.quantile = quantile
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.lr = lr
        self.max_iter = max_iter

        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y, sample_weight=None):
        n_samples, n_features = X.shape
        n_params = n_features

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

        n, m = A.shape
        av = jnp.arange(n) + m

        result = _linprog_simplex(c, A, b, maxiter=self.max_iter, tol=1e-3)

        solution = result[0]

        params = solution[:n_params] - solution[n_params : 2 * n_params]
        self.n_iter_ = result[2]

        if self.fit_intercept:
            self.coef_ = params[1:]
            self.intercept_ = params[0]
        else:
            self.coef_ = params
            self.intercept_ = 0.0
        return self

    def predict(self, X):
        if self.fit_intercept:
            X = jnp.column_stack((jnp.ones(X.shape[0]), X))

            return jnp.dot(X, jnp.hstack([self.intercept_, self.coef_]))
        else:
            return jnp.dot(X, self.coef_)
