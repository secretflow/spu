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

from scipy import linalg
from enum import Enum
from scipy import linalg, sparse
import jax.numpy as jnp
import jax.scipy as jsci
import numpy as np


class Solver(Enum):
    SVD = 'svd'  # not supported
    CHOLESKY = 'cholesky'


class Ridge:
    """Linear least squares with l2 regularization.

    Minimizes the objective function::

    ||y - Xw||^2_2 + alpha * ||w||^2_2

    This model solves a regression model where the loss function is
    the linear least squares function and regularization is given by
    the l2-norm. Also known as Ridge Regression or Tikhonov regularization.
    This estimator has built-in support for multi-variate regression
    (i.e., when y is a 2d-array of shape (n_samples, n_targets)).

    Parameters
    ----------
    alpha : {float}, default=1.0
        Constant that multiplies the L2 term, controlling regularization
        strength. `alpha` must be a non-negative float i.e. in `[0, inf)`.

    solver : {'svd', 'cholesky'}, default='cholesky'
        Solver to use in the computational routines:

        - 'svd' uses a Singular Value Decomposition of X to compute the Ridge
          coefficients.

        - 'cholesky' uses the standard jax.scipy.linalg.solve function to
          obtain a closed-form solution via a Cholesky decomposition of
          dot(X.T, X)
    """

    def __init__(self, alpha=1.0, solver="lsqr") -> None:
        self.alpha = alpha
        self.solver = solver

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
        alpha = jnp.asarray(self.alpha, dtype=x.dtype).ravel()
        print(f"<<<solver: {self.solver}")
        if self.solver == Solver.CHOLESKY.value:
            self.coef = _solve_cholesky(x, y, alpha)
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
        ret = jnp.dot(a, b)
        return ret

def _solve_cholesky(x, y, alpha):
    # w = inv(X^t X + alpha*Id) * X.T y
    n_features = x.shape[1]

    A = jnp.dot(x.T, x)
    Xy = jnp.dot(x.T, y)

    for i in range(n_features):
        A = A.at[i, i].set(A[i][i] + alpha[0])

    coefs = jsci.linalg.solve(A, Xy, assume_a="pos", overwrite_a=True).T
    return coefs
