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
        """Fit the estimator to the data."""
        assert len(X.shape) == 2, f"Expected X to be 2 dimensional array, got {X.shape}"

        self._mean = jnp.mean(X, axis=0)
        X_centered = X - self._mean

        # The covariance matrix
        cov_matrix = jnp.cov(X_centered, rowvar=False)

        # Initialization
        components = []
        variances = []

        for _ in range(self._n_components):
            # Initialize a random vector
            vec = jnp.ones((X_centered.shape[1],))

            for _ in range(100):  # Max iterations
                # Power iteration
                vec = jnp.dot(cov_matrix, vec)
                vec /= jnp.linalg.norm(vec)

            # Compute the corresponding eigenvalue
            eigval = jnp.dot(vec.T, jnp.dot(cov_matrix, vec))

            components.append(vec)
            variances.append(eigval)

            # Remove the component from the covariance matrix
            cov_matrix -= eigval * jnp.outer(vec, vec)

        self._components = jnp.column_stack(components)
        self._variances = jnp.array(variances)

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

    def inverse_transform(self, X_transformed):
        """Transform the data back to the original space.

        Parameters
        ----------
        X_transformed : {array-like}, shape (n_samples, n_components)
            Data in the transformed space.

        Returns
        -------
        X_original : array, shape (n_samples, n_features)
            Data in the original space.
        """
        assert len(X_transformed.shape) == 2, f"Expected X_transformed to be 2 dimensional array, got {X_transformed.shape}"

        X_original = jnp.dot(X_transformed, self._components.T) + self._mean

        return X_original
