# Copyright 2025 Ant Group Co., Ltd.
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

from sml.manifold.jacobi import Jacobi
from sml.manifold.kneighbors import mpc_kneighbors_graph


class SE:
    """Project the sample on the first eigenvectors of the graph Laplacian.

    The adjacency matrix is used to compute a normalized graph Laplacian
    whose spectrum (especially the eigenvectors associated to the
    smallest eigenvalues) has an interpretation in terms of minimal
    number of cuts necessary to split the graph into comparably sized
    components.

    This embedding can also 'work' even if the ``adjacency`` variable is
    not strictly the adjacency matrix of a graph but more generally
    an affinity or similarity matrix between samples (for instance a k-NN matrix).

    However care must taken to always make the affinity matrix symmetric
    so that the eigenvector decomposition works as expected.

    Parameters
    ----------
    n_components : int, default=2
        The dimension of the projected subspace.

    affinity : str, default="nearest_neighbors"
        The type of affinity matrix. Currently, only "nearest_neighbors" is supported.

    n_neighbors : int
        The number of neighbors to use for constructing the k-NN graph.

    n_samples : int
            The number of samples.

    n_features : int
        The number of features.

    max_iterations : int
        Maximum number of iterations of jacobi algorithm.

    Attributes
    ----------
    affinity_matrix_ : ndarray
        The computed affinity matrix.

    D_ : ndarray
        Degree matrix.

    L_ : ndarray
        Laplacian matrix.

    embedding_ : ndarray
        Reduced-dimensional sample embedding.

    Methods
    ----------
    _get_affinity_matrix(X):
        Computes the affinity matrix and makes it symmetric.

    _normalization_affinity_matrix(norm_laplacian=True):
        Normalizes the affinity matrix and computes the Laplacian matrix.

    fit(X):
        Fits the model, computes the eigenvectors of the Laplacian matrix, and generates the embedding.

    fit_transform(X):
        Fits the model and returns the reduced-dimensional embedding.
    """

    def __init__(
        self,
        n_neighbors,
        n_samples,
        n_features,
        *,
        n_components=2,
        max_iterations=5,
        affinity="nearest_neighbors",
    ):
        assert affinity == "nearest_neighbors", "affinity must be 'nearest_neighbors'"
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.n_samples = n_samples
        self.n_features = n_features
        self.max_iterations = max_iterations

    def _get_affinity_matrix(self, X):
        """Computes the affinity matrix and makes it symmetric.

        Parameters
        ----------
        X : ndarray
            The input data matrix.

        Returns
        -------
        affinity_matrix_ : ndarray
            The symmetric affinity matrix.
        """
        self.affinity_matrix_ = mpc_kneighbors_graph(X, self.n_neighbors)
        # currently only symmetric affinity_matrix supported
        self.affinity_matrix_ = 0.5 * (self.affinity_matrix_ + self.affinity_matrix_.T)
        return self.affinity_matrix_

    def _normalization_affinity_matrix(
        self,
        norm_laplacian=True,
    ):
        """Normalizes the affinity matrix and computes the Laplacian matrix.

        Parameters
        ----------
        norm_laplacian : bool, default=True
            Whether to use the symmetric normalized Laplacian matrix.

        Returns
        -------
        D_ : ndarray
            The degree matrix.

        L_ : ndarray
            The Laplacian matrix.
        """
        self.D_ = jnp.sum(self.affinity_matrix_, axis=1)
        self.D_ = jnp.diag(self.D_)

        self.L_ = self.D_ - self.affinity_matrix_
        self.D2_ = jnp.diag(jnp.reciprocal(jnp.sqrt(jnp.diag(self.D_))))
        if norm_laplacian == True:
            # normalization
            self.L_ = jnp.dot(self.D2_, self.L_)
            self.L_ = jnp.dot(self.L_, self.D2_)
        return self.D_, self.L_

    def fit(self, X):
        """Fits the model, computes the eigenvectors of the Laplacian matrix, and generates the embedding.

        Parameters
        ----------
        X : ndarray
            The input data matrix.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._get_affinity_matrix(X)
        D, L = self._normalization_affinity_matrix()

        L, Q = Jacobi(L, self.max_iterations)
        L = jnp.diag(L)
        L = jnp.array(L)

        Q = Q.T
        L = jnp.tile(L, (self.n_samples, 1))
        self.embedding_ = jax.lax.sort(operand=(L, Q), dimension=-1, num_keys=1)[1]

        self.embedding_ = self.embedding_[:, 1 : self.n_components + 1]

        D = jnp.diag(D)
        self.embedding_ = self.embedding_.T * jnp.reciprocal(jnp.sqrt(D))
        self.embedding_ = self.embedding_.T
        return self

    def fit_transform(self, X):
        """Fits the model and returns the reduced-dimensional embedding.

        Parameters
        ----------
        X : ndarray
            The input data matrix.

        Returns
        -------
        embedding_ : ndarray
            The reduced-dimensional sample embedding.
        """
        self.fit(X)
        return self.embedding_
