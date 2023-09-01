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
from enum import Enum

import jax.numpy as jnp
import jax.scipy as jsci

import sml.utils.extmath as extmath


class Solver(Enum):
    SVD = 'svd'
    CHOLESKY = 'cholesky'


class Ridge:
    """Linear least squares with l2 regularization.

    Minimizes the objective function::

    ||y - Xw||^2_2 + alpha * ||w||^2_2

    This model solves a regression model where the loss function is
    the linear least squares function and regularization is given by
    the l2-norm. Also known as Ridge Regression or Tikhonov regularization.

    Parameters
    ----------
    alpha : {float}, default=1.0
        Constant that multiplies the L2 term, controlling regularization
        strength. `alpha` must be a non-negative float i.e. in `[0, inf)`.

    fit_bias : bool, default=True
        Whether to fit the bias for this model. If set
        to false, no bias will be used in calculations

    solver : {'svd', 'cholesky'}, default='cholesky'
        Solver to use in the computational routines:

        - 'svd' uses a Singular Value Decomposition of X to compute the Ridge
          coefficients.

        - 'cholesky' uses the standard jax.scipy.linalg.solve function to
          obtain a closed-form solution via a Cholesky decomposition of
          dot(X.T, X)

    max_iter : int, default=100
        Maximum number of iterations for svd solver.
        For 'svd' solvers, the default value is 100.
    """

    def __init__(
        self, alpha=1.0, fit_bias=True, solver="cholesky", max_iter=100
    ) -> None:
        self.alpha = alpha
        self.solver = solver
        self.fit_bias = fit_bias
        self.max_iter = max_iter

    def fit(self, x, y):
        """Fit Ridge regression model.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Training data.

        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        alpha = float(self.alpha)

        x, y, x_offset, y_offset = self.preprocess_data(x, y)

        if self.solver == Solver.SVD.value:
            self.coef = _solve_svd(x, y, alpha, self.max_iter)
        if self.solver == Solver.CHOLESKY.value:
            self.coef = _solve_cholesky(x, y, alpha)
        self.coef = self.coef.ravel()

        self.set_bias(x_offset, y_offset)

        return self

    def predict(self, x):
        """
        Predict using the linear model.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Input data for prediction.

        Returns
        -------
        C : array, shape (n_samples,)
            Returns predicted values.
        """
        a = x
        b = self.coef.T
        ret = jnp.dot(a, b) + self.bias
        return ret

    def preprocess_data(self, x, y):
        # Center and scale data.
        if self.fit_bias:
            x_offset = jnp.average(x, axis=0)
            x -= x_offset
            y_offset = jnp.average(y, axis=0)
            y -= y_offset
        else:
            x_offset = None
            y_offset = None
        return x, y, x_offset, y_offset

    def set_bias(self, x_offset, y_offset):
        if self.fit_bias:
            self.bias = y_offset - jnp.dot(x_offset, self.coef.T)
        else:
            self.bias = 0.0


def _solve_cholesky(x, y, alpha):
    # w = inv(X^t X + alpha*Id) * X.T y
    n_features = x.shape[1]

    A = jnp.dot(x.T, x)
    Xy = jnp.dot(x.T, y)

    A += jnp.diag(jnp.ones(n_features) * alpha)

    coefs = jsci.linalg.solve(A, Xy, assume_a="pos", overwrite_a=True).T
    return coefs


def _solve_svd(x, y, alpha, max_iter):
    U, s, V = extmath.svd(x, max_iter)
    s_nnz = s[:, jnp.newaxis]
    UTy = jnp.dot(U.T, y)
    d = s_nnz / (s_nnz**2 + alpha)
    d_UT_y = d * UTy
    return jnp.dot(V.T, d_UT_y).T
