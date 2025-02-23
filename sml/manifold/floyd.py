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
import jax.numpy as jnp
import numpy


# Unoptimized Floyd algorithm
def floyd(dist):
    n = len(dist)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist = dist.at[i, j].set(
                    jnp.minimum(dist[i, j], dist[i, k] + dist[k, j])
                )

    return dist


# Optimized Floyd algorithm, but due to the. at.set problem, the run time is relatively long
def floyd_opt_1(dist):
    # 1. Require that dist is a symmetric matrix.
    # 2. Require that the diagonal elements of dist are 0.
    # 3. Require that in the off-diagonal elements of dist, if the distance between samples i and j is infinite, it should be represented as an infinite or a very large element.
    n = len(dist)

    for k in range(n):
        # Package calculation batch_2,batch_3
        batch_2 = dist
        batch_2 = jnp.delete(batch_2, k, axis=0)
        col_k_without_dkk = batch_2[:, k]
        batch_2 = jnp.delete(batch_2, k, axis=1)
        dist_ik = jnp.zeros_like(batch_2)
        dist_kj = jnp.zeros_like(batch_2)

        for i in range(n - 1):
            if i < k:
                dist_ik = dist_ik.at[i].set(jnp.full(n - 1, dist[i][k]))
            else:
                dist_ik = dist_ik.at[i].set(jnp.full(n - 1, dist[i + 1][k]))

        dist_kj = dist_ik.T

        # Take out the upper triangle and calculate
        indices = numpy.triu_indices(batch_2.shape[0], k=1)
        batch_2_upper_triangle = batch_2[indices]
        dist_ik_upper_triangle = dist_ik[indices]
        dist_kj_upper_triangle = dist_kj[indices]

        batch_2_upper_triangle = jnp.minimum(
            batch_2_upper_triangle, dist_ik_upper_triangle + dist_kj_upper_triangle
        )

        # Put the upper triangle back
        batch_2 = jnp.zeros_like(batch_2)
        batch_2 = batch_2.at[indices].set(batch_2_upper_triangle)
        batch_2 += batch_2.T

        batch_2 = jnp.insert(
            batch_2, k, col_k_without_dkk, axis=1
        )  # Put the updated values back to their original positions
        batch_2 = jnp.insert(batch_2, k, dist[k], axis=0)
        dist = batch_2

    return dist


# Floyd optimized without using. at.set
def floyd_opt(D):
    # 1. Require that dist is a symmetric matrix.
    # 2. Require that the diagonal elements of dist are 0.
    # 3. Require that in the off-diagonal elements of dist, if the distance between samples i and j is infinite, it should be represented as an infinite or a very large element.
    n = D.shape[0]
    for k in range(n):
        # Update distance through intermediate node k
        D_tmp = D[:, k][:, None] + D[k, :][None, :]
        D = jnp.minimum(D, D_tmp)
    return D
