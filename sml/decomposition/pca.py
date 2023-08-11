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

import jax.numpy as jnp
from enum import Enum
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from sml.utils import extmath 


class Method(Enum):
    PCA_power = 'power_iteration'
    PCA_rsvd = 'rsvd'


class PCA:
    def __init__(
        self,
        method: str,
        n_components: int,
        max_iter: int = 100,
        projection_iter: int = 4,
        random_matrix=None,
        scale=None,
    ):
        """A PCA estimator implemented with Power Iteration/Randomized SVD.
        Parameters
        ----------
        method : str
            The method to compute the principal components.
            'power_iteration' uses Power Iteration to compute the eigenvalues and eigenvectors.
            'rsvd' uses Randomized SVD to compute the eigenvalues and eigenvectors.
        n_components : int
            Number of components to keep.
        max_iter : int, default=100
            Maximum number of iterations for Power Iteration, larger numbers mean higher accuracy and more time consuming.
        projection_iter : int, default=4
            Used when the 'rsvd' method is used. Number of projection iterations.
            It is set to 4, unless `n_components` is small (< .1 * min(X.shape)) in which case `n_iter` is set to 7.
        random_matrix : Array, default=None
            Used when the 'rsvd' method is used. The seed of the pseudo random number generator to use when
            shuffling the data, i.e. getting the random vectors to initialize.
        scale : list, default=None
            Used when the 'rsvd' method is used. Prevents overflow errors when caculating np.dot(matrix_A, matrix_B) in spu.

        References
        ----------
        .. Power Iteration: https://en.wikipedia.org/wiki/Power_iteration

        .. Randomized SVD: arxiv:`"Finding structure with randomness: Stochastic algorithms
        for constructing approximate matrix decompositions" <0909.4061>` Halko, et al. (2009)
        """
        # parameter check.
        assert n_components > 0, f"n_components should >0"
        assert method in [
            e.value for e in Method
        ], f"method should in {[e.value for e in Method]}, but got {method}"

        self._n_components = n_components
        self._max_iter = max_iter
        self._mean = None
        self._components = None
        self._variances = None
        self._projection_iter = projection_iter  # used in rsvd
        self._random_matrix = random_matrix  # used in rsvd
        self._scale = scale  # used in rsvd
        self._method = Method(method)

    def fit(self, X):
        """Fit the estimator to the data.
        In the 'power_iteration' method, we use the Power Iteration algorithm to compute the eigenvalues and eigenvectors.
        The Power Iteration algorithm works by repeatedly multiplying a vector by the matrix to inflate the largest eigenvalue,
        and then normalizing to keep numerical stability.
        After finding the largest eigenvalue and eigenvector, we deflate the matrix by subtracting the outer product of the 
        eigenvector and itself, scaled by the eigenvalue. This leaves a matrix with the same eigenvectors, but the largest 
        eigenvalue is replaced by zero.

        In the 'rsvd' method, we use the Randomized SVD to compute the eigenvalues and eigenvectors.
        Step 0: For matrix A of size n_sample * n_feature, identify a target rank(n_component).
        Step 1: Using random projections Omega of size n_sample * n_component to sample the column space,
                find a matrix Q whose columns approximate the column space of X,
                so that X ≈ Q.T @ Q @ X, we can use power_iteration and QR decomposition to find matrix Q.
        Step 2: Project X onto the Q subspace, B = Q.T @ X, and then compute the matrix decomposition on B(n_component * n_feature),
                we can implement singular value decomposition using power_iteration.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Training data.
        Returns
        -------
        self : object
            Returns an instance of self.
        Notes
        -----
        When use rsvd, there are a large number of continuous matrix multiplies inside, which will make the value expand rapidly and overflow,
        we can solve it in the following ways.
        Step 0: Modify the definition of simulator as follows:
        config = spu_pb2.RuntimeConfig(
            protocol=spu_pb2.ProtocolKind.ABY3,
            field=spu_pb2.FieldType.FM128,
            fxp_fraction_bits=30,
            )
        sim_aby = spsim.Simulator(3, config)
        Step 1: Select the appropriate parameter of scale, overflow errors often occur in this function: extmath.rsvd_iteration.
        """

        self._mean = jnp.mean(X, axis=0)
        assert (
            len(X.shape) == 2
        ), f"Expected X to be 2 dimensional array, got {X.shape}"
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

                for _ in range(self._max_iter):  # Max iterations
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

        elif self._method == Method.PCA_rsvd:
            result = extmath.randomized_svd(
                X_centered,
                n_components=self._n_components,
                random_matrix=self._random_matrix,
                n_iter=self._projection_iter,
                scale=self._scale,
                eigh_iter=self._max_iter,
            )
            self._components = (result[2]).T
            self._variances = (result[1] ** 2) / (X.shape[0] - 1)

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
        assert len(X_transformed.shape) == 2, f"Expected X_transformed to be 2 dimensional array, got {X_transformed.shape}"

        X_original = jnp.dot(X_transformed, self._components.T) + self._mean

        return X_original
