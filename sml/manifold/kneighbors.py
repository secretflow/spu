# Copyright 2024 Ant Group Co., Ltd.
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

import spu.intrinsic as si


def test_mpc_kneighbors_graph(
    X,  # the input samples to calculate the nearest neighbors
    num_samples,
    num_features,
    n_neighbors,  # Define the number of nearest neighbors, excluding the sample itself
    *,
    mode="distance",
    metric="minkowski",  # Distance is defined as the Euclidean distance between samples
    p=2,
):

    # Calculate the square of the Euclidean distance between every two samples
    X_expanded = jnp.expand_dims(X, axis=1) - jnp.expand_dims(X, axis=0)
    X_expanded = jnp.square(X_expanded)
    Dis = jnp.sum(X_expanded, axis=-1)

    # Sort each row of Dis, first calculate the permutation, and then apply the permutation to Dis
    # Index_Dis = jnp.argsort(Dis, axis=1)

    # Knn = jnp.zeros((num_samples, num_samples))
    # for i in range(num_samples):
    #     temp_pi = jnp.arange(num_samples)
    #     per_dis = si.permute(Dis[i], si.permute(temp_pi, Index_Dis[i]))
    #     for j in range(num_samples):
    #         Knn = Knn.at[i, j].set(per_dis[j])
    
    # top_k行向量
    Knn = jnp.zeros((num_samples, num_samples))
    MIndex_Dis = jnp.zeros((num_samples, num_samples))
    Index_Dis = jnp.zeros(num_samples)
    temp_pi = jnp.arange(num_samples)
    for i in range(num_samples):
        _, Index_Dis = jax.lax.top_k(-Dis[i], n_neighbors)
        for j in range(num_samples):
            MIndex_Dis=MIndex_Dis.at[i,j].set(Index_Dis[j])
        Knn = Knn.at[i].set(si.permute(Dis[i], si.permute(temp_pi, Index_Dis)))
    return MIndex_Dis,Knn


def mpc_kneighbors_graph(
    X,  # the input samples to calculate the nearest neighbors
    num_samples,
    num_features,
    n_neighbors,  # Define the number of nearest neighbors, excluding the sample itself
    *,
    mode="distance",
    metric="minkowski",  # Distance is defined as the Euclidean distance between samples
    p=2,
):

    # Calculate the square of the Euclidean distance between every two samples
    X_expanded = jnp.expand_dims(X, axis=1) - jnp.expand_dims(X, axis=0)
    X_expanded = jnp.square(X_expanded)
    Dis = jnp.sum(X_expanded, axis=-1)

    # Sort each row of Dis, first calculate the permutation, and then apply the permutation to Dis
    Index_Dis = jnp.argsort(Dis, axis=1)

    Knn = jnp.zeros((num_samples, num_samples))
    for i in range(num_samples):
        temp_pi = jnp.arange(num_samples)
        per_dis = si.permute(Dis[i], si.permute(temp_pi, Index_Dis[i]))
        for j in range(num_samples):
            Knn = Knn.at[i, j].set(per_dis[j])
    
    
    # Find the square root of the Euclidean distance of the nearest neighbor previously calculated, and set the distance of non nearest neighbors to 0
    Knn2 = jnp.zeros((num_samples, num_samples))

    def update_knn_row(i, Knn_row, n_neighbors):
        def update_element(j, Knn_value):
            return jnp.where(j <= n_neighbors, jnp.sqrt(Knn_value), 0)

        # Vectorize the inner loop over `j`
        Knn_row_updated = jax.vmap(update_element, in_axes=(0, 0))(
            jnp.arange(Knn_row.shape[0]), Knn_row
        )
        return Knn_row_updated

    # Vectorize the outer loop over `i`
    Knn2 = jax.vmap(lambda i, Knn_row: update_knn_row(i, Knn_row, n_neighbors))(
        jnp.arange(num_samples), Knn
    )

    # Reverse permutation of Dis to restore the previous order
    Knn3 = jnp.zeros((num_samples, num_samples))
    for i in range(num_samples):
        per_dis = si.permute(Knn2[i], Index_Dis[i])
        for j in range(num_samples):
            Knn3 = Knn3.at[i, j].set(per_dis[j])

    return Knn3