# Copyright 2024 Ant Group Co., Ltd.
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

from sml.manifold.floyd import floyd_opt
from sml.manifold.jacobi import Jacobi
from sml.manifold.kneighbors import mpc_kneighbors_graph

class ISOMAP:
    """Isomap Embedding.

    Non-linear dimensionality reduction through Isometric Mapping

    The core idea of ISOMAP is to compute the distances between samples using a k-nearest neighbors graph (k-NN graph), and then project the data into a low-dimensional space using Multidimensional Scaling (MDS).

    Parameters
    ----------
    n_neighbors : int
        The number of neighbors to use when constructing the k-nearest neighbors graph.

    n_samples : int
        The number of samples.

    n_features : int
        The number of features.

    n_components : int, default=2
        The number of dimensions for the reduced representation.

    metric : str, default="minkowski"
        The type of distance metric to use. Currently, only "minkowski" is supported.

    p : int, default=2
        The power parameter for the Minkowski distance. p=2 indicates Euclidean distance.

    max_iterations : int
        Maximum number of iterations of jacobi algorithm.
    
    Attributes
    ----------
    nbrs_ : ndarray
        The k-nearest neighbors graph.

    dist_matrix_ : ndarray
        The distance matrix.

    embedding_ : ndarray
        The reduced-dimensional sample embedding.

    Methods
    ----------
    _get_knn_matrix(X):
        Computes the k-nearest neighbors graph.

    _get_shortest_paths():
        Computes the shortest paths between samples.

    MDS():
        Computes the low-dimensional embedding using Multidimensional Scaling (MDS).

    fit(X):
        Fits the ISOMAP model.

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
        metric="minkowski",
        p=2,
        max_iterations=5,
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.n_samples = n_samples
        self.n_features = n_features
        self.max_iterations = max_iterations

    def _get_knn_matrix(self, X):
        """Computes the k-nearest neighbors graph.

        Parameters
        ----------
        X : ndarray
            The input data matrix.

        Returns
        -------
        nbrs_ : ndarray
            The k-nearest neighbors graph.
        """
        self.nbrs_ = mpc_kneighbors_graph(
            X, self.n_samples, self.n_features, self.n_neighbors
        )
        return self.nbrs_

    def _get_shortest_paths(self):
        """Compute the shortest paths between all pairs of samples.

        Returns
        -------
        dist_matrix_ : ndarray
            The distance matrix containing shortest paths.
        """

        # Replace zero distances with infinity
        self.nbrs_ = (self.nbrs_ == 0) * jnp.inf + self.nbrs_
        # Set diagonal to zero
        self.nbrs_ = jnp.where(jnp.eye(self.nbrs_.shape[0]), 0, self.nbrs_)
        # Ensure symmetry of the distance matrix
        flag = self.nbrs_ <= self.nbrs_.T
        self.nbrs_ = flag * self.nbrs_ + (1 - flag) * self.nbrs_.T

        # Compute the shortest path distances using the Floyd-Warshall algorithm
        self.dist_matrix_ = floyd_opt(self.nbrs_)

        return self.dist_matrix_

    def mds(self):
        """Computes the low-dimensional embedding using Multidimensional Scaling (MDS).

        Returns
        -------
        embedding_ : ndarray
            The reduced-dimensional embedding of the samples.
        """

        D_2 = jnp.square(self.dist_matrix_)
        B = jnp.zeros((self.n_samples, self.n_samples))
        B = -0.5 * D_2
        # Sum by row
        dist_2_i = jnp.sum(B, axis=1)
        dist_2_i = dist_2_i / self.n_samples
        # Sum by column
        dist_2_j = dist_2_i.T
        # sum all
        dist_2 = jnp.sum(dist_2_i)
        dist_2 = dist_2 / (self.n_samples)
        B = B - dist_2_i[:, None] - dist_2_j[None, :] + dist_2
        
        # Compute eigenvalues and eigenvectors
        values, vectors = Jacobi(B, self.n_samples, self.max_iterations)

        values = jnp.diag(values)
        values = jnp.array(values)

        # Retrieve the largest n_components values and their corresponding vectors.
        # Sort each column of vectors according to values.
        vectors = vectors.T
        values_rows = jnp.tile(values, (self.n_samples, 1))
        values = jax.lax.sort(values)
        vectors = jax.lax.sort(
            operand=(values_rows, vectors), dimension=-1, num_keys=1
        )[1]

        vectors = vectors[:, self.n_samples - self.n_components : self.n_samples]
        values = values[self.n_samples - self.n_components : self.n_samples]
        values = jnp.sqrt(values)
        values = jnp.diag(values)
        self.embedding_ = jnp.dot(vectors, values)
        self.embedding_ = self.embedding_[:, ::-1]

        return self.embedding_

    def fit(self, X):
        """Fit the ISOMAP model to the data.

        Parameters
        ----------
        X : ndarray
            The input data matrix.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._get_knn_matrix(X)
        self._get_shortest_paths()
        self.mds()
        return self

    def fit_transform(self, X):
        """Fit the model and return the reduced-dimensional embedding.

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
