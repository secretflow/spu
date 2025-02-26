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

import os
import sys
from enum import Enum

import jax.numpy as jnp

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from sml.fyy_pca.jacobi_evd import serial_jacobi_evd


class Method(Enum):
    PCA_power = 'power_iteration'
    PCA_jacobi = 'serial_jacobi_iteration'


class PCA:
    def __init__(
        self,
        method: str,
        n_components: int,
        max_power_iter: int = 300,
        max_jacobi_iter: int = 10,
        projection_iter: int = 4,
        rotate_matrix=None,
    ):
        """A PCA estimator implemented with Jacobi Method.
        Parameters
        ----------
        method : str
            The method to compute the principal components.
            'power_iteration' uses Power Iteration to compute the eigenvalues and eigenvectors.
            'serial_jacobi_iteration' uses Jacobi Method to compute the eigenvalues and eigenvectors.
        n_components : int
            Number of components to keep.
        max_power_iter : int, default=300
            Maximum number of iterations for Power Iteration, larger numbers mean higher accuracy and more time consuming.
        max_jacobi_iter : int, default=5/10
            Maximum number of iterations for Jacobi Method, larger numbers mean higher accuracy and more time consuming.

        References
        ----------
        .. Jacobi Method: https://en.wikipedia.org/wiki/Jacobi_eigenvalue_algorithm
        """
        # parameter check.
        assert n_components > 0, f"n_components should >0"
        assert method in [
            e.value for e in Method
        ], f"method should in {[e.value for e in Method]}, but got {method}"

        self._n_components = n_components
        self._max_power_iter = max_power_iter
        self._max_jacobi_iter = max_jacobi_iter
        self._mean = None
        self._components = None
        self._variances = None
        self._rotate_matrix = rotate_matrix  # used in serial_jacobi
        self._method = Method(method)

    def fit(self, X):

        self._mean = jnp.mean(X, axis=0)
        assert len(X.shape) == 2, f"Expected X to be 2 dimensional array, got {X.shape}"
        X_centered = X - self._mean

        if self._method == Method.PCA_power:
            # The covariance matrix
            cov_matrix = jnp.cov(X_centered, rowvar=False)

            # Initialization
            components = []
            variances = []

            for _ in range(self._n_components):
                # Initialize a random vector
                vec = jnp.ones((X_centered.shape[1],))

                for _ in range(self._max_power_iter):  # Max iterations
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

        elif self._method == Method.PCA_jacobi:

            cov_matrix = jnp.cov(X_centered, rowvar=False)

            result = serial_jacobi_evd(
                cov_matrix, J=self._rotate_matrix, max_jacobi_iter=self._max_jacobi_iter
            )

            sorted_indices = jnp.argsort(result[0])[::-1]
            top_k_indices = sorted_indices[: self._n_components]
            components = result[1].T[top_k_indices]

            self._components = components.T
            self._variances = result[0][top_k_indices]

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
        assert (
            len(X.shape) == 2
        ), f"Expected X_transformed to be 2 dimensional array, got {X.shape}"

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
        assert (
            len(X_transformed.shape) == 2
        ), f"Expected X_transformed to be 2 dimensional array, got {X_transformed.shape}"

        X_original = jnp.dot(X_transformed, self._components.T) + self._mean

        return X_original
