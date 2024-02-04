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

import jax
import jax.numpy as jnp

def _kmeans_single(
    x,
    centers_init,
    n_clusters,
    max_iter=300,
):
    centers = jnp.array([x[i] for i in centers_init])
    for _ in range(max_iter):
        C = x.reshape((1, x.shape[0], x.shape[1])) - centers.reshape(
            (centers.shape[0], 1, centers.shape[1])
        )
        C = jnp.argmin(jnp.sum(jnp.square(C), axis=2), axis=0)
        S = jnp.tile(C, (n_clusters, 1))
        ks = jnp.arange(n_clusters)
        aligned_array_raw = (S.T - ks).T
        aligned_array = jnp.equal(aligned_array_raw, 0)

        centers_raw = x.reshape(
            (1, x.shape[0], x.shape[1])
        ) * aligned_array.reshape(
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
    distance =  jnp.sum(jnp.square(C), axis=2)
    index = jnp.argmin(distance, axis=0)
    return jnp.sum(jnp.take(distance, index))
    # return jnp.argmin(jnp.sum(jnp.take(distance, index)))


class KMEANS:
    """
    Parameters
    ----------
    n_clusters : int
        The number of clusters to form as well as the number of
        centroids to generate.

    n_samples : int
        The number of samples.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    """

    def __init__(self, n_clusters, n_samples, init="random", n_init=1, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.init = init
        self.n_init = n_init
        ### if the n_init is 1, reduce the dimension of init_params to eliminate the reshape operations
        if n_init == 1:
            if init == "random":
                self.init_params = jax.random.randint(
                    jax.random.PRNGKey(1), shape=[self.n_clusters], minval=0, maxval=n_samples
                )
        else:
            if init == "random":
                self.init_params = jax.random.randint(
                    jax.random.PRNGKey(1), shape=[n_init, self.n_clusters], minval=0, maxval=n_samples
                )
        self._centers = jnp.zeros(())

    def fit(self, x):
        """Fit KMEANS.

        Firstly, randomly select the initial centers. Then calculate the distance between each sample and each center,
        and assign each sample to the nearest center. Use an `aligned_array` to indicate the samples in a cluster,
        where unrelated samples will be set to 0. Once all samples are assigned, the center of each cluster will
        be updated to the average. The average could be got by `sum(data * aligned_array) / sum(aligned_array)`.
        Different clusters could use broadcast for better performance.

        Parameters
        ----------
        x : {array-like}, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        self : object
            Returns an instance of self.
        """

        # centers = jnp.array([x[i] for i in self.init_params])
        # centers = _kmeans_single(x, centers, self.n_clusters, self.max_iter)
        if self.n_init == 1:
            centers_best = _kmeans_single(x, self.init_params, self.n_clusters, self.max_iter)
        else:
            centers = jax.vmap(_kmeans_single, in_axes=(None, 0, None, None))(x, self.init_params, self.n_clusters, self.max_iter)
            C = x.reshape((1, 1, x.shape[0], x.shape[1])) - centers.reshape(
                (centers.shape[0], centers.shape[1], 1, centers.shape[2])
            )
            distance =  jnp.sum(jnp.square(C), axis=3)
            index = jnp.argmin(distance, axis=1)
            # return jnp.sum(jnp.take(distance, index))
            inertia = jnp.sum(jnp.take(distance, index), axis=1)
            centers_best = centers[jnp.argmin(inertia)]
            # inertia = jax.vmap(_inertia, in_axes=(None, 0))(x, centers)
            # centers_best = centers[jnp.argmin(inertia)]
            # centers_best = centers[inertia]
            # centers_best = jnp.take(centers, jnp.argmin(inertia)[:, jnp.newaxis, jnp.newaxis])
            # centers_best = centers[jnp.argmin(inertia)]
            # centers_best = centers[jnp.argmin(inertia)]
            # centers_best = jnp.take(centers, jnp.argmin(inertia)[:, jnp.newaxis])
        # for _ in range(self.max_iter):
        #     C = x.reshape((1, x.shape[0], x.shape[1])) - centers.reshape(
        #         (centers.shape[0], 1, centers.shape[1])
        #     )
        #     C = jnp.argmin(jnp.sum(jnp.square(C), axis=2), axis=0)
        #     S = jnp.tile(C, (self.n_clusters, 1))
        #     ks = jnp.arange(self.n_clusters)
        #     aligned_array_raw = (S.T - ks).T
        #     aligned_array = jnp.equal(aligned_array_raw, 0)

        #     centers_raw = x.reshape(
        #         (1, x.shape[0], x.shape[1])
        #     ) * aligned_array.reshape(
        #         (aligned_array.shape[0], aligned_array.shape[1], 1)
        #     )
        #     equals_sum = jnp.sum(aligned_array, axis=1)
        #     centers_sum = jnp.sum(centers_raw, axis=1)
        #     centers = jnp.divide(centers_sum.T, equals_sum).T

        self._centers = centers_best
        # self._inertia = inertia
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
