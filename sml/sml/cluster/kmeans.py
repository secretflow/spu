# Copyright 2023 Ant Group Co., Ltd.
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

import math

import jax
import jax.numpy as jnp


def _kmeans_plusplus_single(X, n_clusters, init_center_id, init_params):
    """Computational component for initialization of n_clusters by k-means++.

    Parameters
    ----------
    x : {array-like}, shape (n_samples, n_features)
        Input data.

    n_clusters : int
        The number of clusters to form as well as the number of
        centroids to generate.

    init_center_id : {array-like}, shape (1,)
        Randome variable generated before SPU running time.

    init_params : {array-like}, shape (self.n_clusters-1, 2 + int(math.log(n_clusters)))
        Randome variables generated before SPU running time.

    Returns
    -------
    centers : {array-like}, shape (n_clusters, n_features)
        Centers calculated for initial centers.
    """
    x_squared_norms = jnp.einsum("ij,ij->i", X, X)
    centers = X[init_center_id][jnp.newaxis, :]
    closest_dist_sq = _euclidean_distances(centers, X, Y_norm_squared=x_squared_norms)
    current_pot = jnp.sum(closest_dist_sq)

    for c in range(0, n_clusters - 1):
        rand_vals = init_params[c, :] * current_pot
        bin = jnp.cumsum(closest_dist_sq)

        def searchsorted_element(x):
            encoding = jnp.where(x >= bin[0:-1], 1, 0)
            return jnp.sum(encoding)

        candidate_ids = jax.vmap(searchsorted_element)(rand_vals)
        distance_to_candidates = _euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms
        )
        distance_to_candidates = jnp.minimum(closest_dist_sq, distance_to_candidates)
        candidates_pot = jnp.sum(distance_to_candidates, axis=1).reshape(-1, 1)
        best_candidate = jnp.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate, :]
        best_candidate = candidate_ids[best_candidate]
        centers = jnp.concatenate([centers, X[best_candidate][jnp.newaxis, :]], axis=0)
    return centers


def _euclidean_distances(X, Y, Y_norm_squared):
    """Computational part of euclidean_distances of X and Y"""
    XX = jnp.einsum("ij,ij->i", X, X)[:, jnp.newaxis]
    YY = Y_norm_squared.reshape(1, -1)
    distances = -2 * jnp.dot(X, Y.T) + XX + YY
    return jnp.maximum(distances, 0)


def _kmeans_single(
    x,
    centers,
    n_clusters,
    max_iter=300,
):
    """A single run of k-means.

    Parameters
    ----------

    x : {array-like}, shape (n_samples, n_features)
        Input data.

    centers_init : {array-like}, shape (n_clusters, n_features)
        The initial centers.

    n_clusters : int
        The number of clusters to form as well as the number of
        centroids to generate.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    Returns
    -------
    centers : {array-like}, shape (n_clusters, n_features)
        Centers found at the last iteration of k-means.

    """
    for _ in range(max_iter):
        C = x.reshape((1, x.shape[0], x.shape[1])) - centers.reshape(
            (centers.shape[0], 1, centers.shape[1])
        )
        C = jnp.argmin(jnp.sum(jnp.square(C), axis=2), axis=0)
        S = jnp.tile(C, (n_clusters, 1))
        ks = jnp.arange(n_clusters)
        aligned_array_raw = (S.T - ks).T
        aligned_array = jnp.equal(aligned_array_raw, 0)

        centers_raw = x.reshape((1, x.shape[0], x.shape[1])) * aligned_array.reshape(
            (aligned_array.shape[0], aligned_array.shape[1], 1)
        )
        equals_sum = jnp.sum(aligned_array, axis=1)
        centers_sum = jnp.sum(centers_raw, axis=1)
        centers = jnp.divide(centers_sum.T, equals_sum).T
    return centers


def _inertia(x, centers):
    C = x.reshape((1, x.shape[0], x.shape[1])) - centers.reshape(
        (centers.shape[0], 1, centers.shape[1])
    )
    distance = jnp.sum(jnp.square(C), axis=2)
    return jnp.sum(jnp.min(distance, axis=0))


class KMEANS:
    """
    Parameters
    ----------
    n_clusters : int
        The number of clusters to form as well as the number of
        centroids to generate.

    n_samples : int
        The number of samples.

    init : {'k-means++', 'random'}, callable or array-like of shape
            (n_clusters, n_features), default='k-means++'
        When 'k-means++' and 'random' are passed, since the random variable
        generation in SPU runtime is not well supported, please initialize the
        KMEANS in outer scope.

    n_init : int, default=1
        Number of times the k-means algorithm is run with different centroid
        seeds.
        When an array to init, n_init will be set to 1.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    """

    def __init__(
        self,
        n_clusters,
        n_samples,
        init="k-means++",
        n_init=1,
        max_iter=300,
        init_center_id=None,
        init_params=None,
        centers=None,
    ):
        ### if init is array like, then n_init will be set to 1. init is array with shape (n_clusters, n_features)
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.init = init
        self.n_init = n_init

        self.init_center_id = None
        self.init_params = None
        self.n_samples = n_samples
        self._centers = None

        if centers is not None:
            # model has been fitted
            self._centers = centers
            return

        if init_params is not None:
            self.init_params = init_params
            self.init_center_id = init_center_id
        else:
            self.__generate_random_state()

    def tree_flatten(self):
        static_data = [
            self.n_clusters,
            self.max_iter,
            self.init,
            self.n_init,
            self.n_samples,
        ]
        dynamic_data = [self._centers, self.init_center_id, self.init_params]
        return dynamic_data, static_data

    @classmethod
    def tree_unflatten(cls, static_data, dynamic_data):
        n_clusters, max_iter, init, n_init, n_samples = static_data
        _centers, init_center_id, init_params = dynamic_data
        return cls(
            n_clusters,
            n_samples,
            init,
            n_init,
            max_iter,
            init_center_id,
            init_params,
            _centers,
        )

    def __generate_random_state(self):
        if self.init == "k-means++":
            ### if the n_init is 1, reduce the dimension of init_params to eliminate the reshape operations
            if self.n_init == 1:
                self.init_center_id = jax.random.choice(
                    jax.random.PRNGKey(1), self.n_samples
                )
                self.init_params = jax.random.uniform(
                    jax.random.PRNGKey(1),
                    shape=(self.n_clusters - 1, 2 + int(math.log(self.n_clusters))),
                )
            else:
                self.init_center_id = jax.random.choice(
                    jax.random.PRNGKey(1), self.n_samples, shape=[self.n_init]
                )
                self.init_params = jax.random.uniform(
                    jax.random.PRNGKey(1),
                    shape=(
                        self.n_init,
                        self.n_clusters - 1,
                        2 + int(math.log(self.n_clusters)),
                    ),
                )
        elif self.init == "random":
            ### if the n_init is 1, reduce the dimension of init_params to eliminate the reshape operations
            if self.n_init == 1:
                self.init_params = jax.random.randint(
                    jax.random.PRNGKey(1),
                    shape=[self.n_clusters],
                    minval=0,
                    maxval=self.n_samples,
                )
            else:
                self.init_params = jax.random.randint(
                    jax.random.PRNGKey(1),
                    shape=[self.n_init, self.n_clusters],
                    minval=0,
                    maxval=self.n_samples,
                )
        else:
            ### If init is array like, then n_init will be set to 1.
            self.n_init = 1

    def fit(self, x):
        """Fit KMEANS.

        Firstly, select the initial centers according to init method from self.init. Then calculate the distance
        between each sample and each center, and assign each sample to the nearest center. Use an `aligned_array`
        to indicate the samples in a cluster, where unrelated samples will be set to 0. Once all samples are
        assigned, the center of each cluster will be updated to the average. The average could be got by `sum(data
        * aligned_array) / sum(aligned_array)`. Different clusters could use broadcast for better performance.
        If n_init>= 1, then n_init groups of initial centers will be used. The final results is the best output of
        `n_init` consecutive runs in terms of inertia.

        Parameters
        ----------
        x : {array-like}, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        self._centers = jnp.zeros(())

        init = self.init
        n_init = self.n_init
        if init == "k-means++":
            if n_init == 1:
                centers = _kmeans_plusplus_single(
                    x, self.n_clusters, self.init_center_id, self.init_params
                )
            else:
                centers = jax.vmap(_kmeans_plusplus_single, in_axes=(None, None, 0, 0))(
                    x, self.n_clusters, self.init_center_id, self.init_params
                )
        elif init == "random":
            centers = jnp.array([x[i] for i in self.init_params])
        else:
            centers = init

        if n_init == 1:
            centers_best = _kmeans_single(x, centers, self.n_clusters, self.max_iter)
        else:
            centers = jax.vmap(_kmeans_single, in_axes=(None, 0, None, None))(
                x, centers, self.n_clusters, self.max_iter
            )
            inertia = jax.vmap(_inertia, in_axes=(None, 0))(x, centers)
            centers_best = centers[jnp.argmin(inertia)]

        self._centers = centers_best
        return self

    def predict(self, x):
        """Result estimates.

        Calculate the distance between each sample and each center,
        and assign each sample to the nearest center.

        Parameters
        ----------
        x : {array-like}, shape (n_samples, n_features)
            Input data for prediction.

        Returns
        -------
        ndarray of shape (n_samples)
            Returns the result of the sample for each class in the model.
        """
        centers = self._centers
        y = x.reshape((1, x.shape[0], x.shape[1])) - centers.reshape(
            (centers.shape[0], 1, centers.shape[1])
        )
        y = jnp.argmin(jnp.sum(jnp.square(y), axis=2), axis=0)
        return y


jax.tree_util.register_pytree_node_class(KMEANS)
