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

import jax
import jax.numpy as jnp
from enum import Enum

class Method(Enum):
    PCA = 'full'


class SimplePCA:
    def __init__(
        self,
        method: str,
        n_components: int,
    ):
        # parameter check.
        assert n_components > 0, f"n_components should >0"
        assert method in [
            e.value for e in Method
        ], f"method should in {[e.value for e in Method]}, but got {method}"

        self._n_components = n_components
        self._mean = None
        self._components = None
        self._variances = None
        self._method = Method(method)

    def fit(self, X):
        """Fit the estimator to the data.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        assert len(X.shape) == 2, f"Expected X to be 2 dimensional array, got {X.shape}"

        self._mean = jnp.mean(X, axis=0)
        X = X - self._mean

        # The covariance matrix
        cov_matrix = jnp.cov(X, rowvar=False)

        # Cholesky decomposition
        L = jnp.linalg.cholesky(cov_matrix)

        # QR decomposition on L
        q, r = jnp.linalg.qr(L)

        # We get eigenvalues from r
        eigvals = jnp.square(jnp.diag(r))   # Take square of diagonal elements

        # Get indices of the largest eigenvalues
        idx = jnp.argsort(eigvals)[::-1][:self._n_components]

        # Get the sorted eigenvectors using indices
        self._components = q[:, idx]

        # Save the variances of the principal components
        self._variances = eigvals[idx]
        return self

    def transform(self, X):
        """Transform the data to the first `n_components` principal components.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Data to be transformed.

        Returns
        -------
        X_transformed : array, shape (n_samples, n_components)
            Transformed data.
        """
        assert len(X.shape) == 2, f"Expected X to be 2 dimensional array, got {X.shape}"

        X = X - self._mean
        return jnp.dot(X, self._components)
