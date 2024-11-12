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

def floyd(
    dist
):
    dist=(dist==0)*jnp.inf+dist
    dist=jnp.where(jnp.eye(dist.shape[0]),0,dist)
    n = len(dist)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist = dist.at[i, j].set(jnp.minimum(dist[i, j], dist[i, k] + dist[k, j]))

    return dist

def floyd_opt(
    dist
):
    dist=(dist==0)*jnp.inf+dist
    dist=jnp.where(jnp.eye(dist.shape[0]),0,dist)
    n = len(dist)

    for k in range(n):
        # # 打包计算batch_2。
        # dist_kk = jnp.full(n-1, dist[k][k])

        # batch_1 = jnp.delete(dist[k], k)
        # batch_1 = jnp.minimum(batch_1, batch_1 + dist_kk)

        # dist = dist.at[k].set(jnp.insert(batch_1, k, dist[k][k]))   # 把更新的值放回原位置
        # dist = dist.at[:, k].set(dist[k])

        # 打包计算batch_3
        batch_2 = dist
        batch_2 = jnp.delete(batch_2, k, axis=0)
        col_k_without_dkk = batch_2[:, k]
        batch_2 = jnp.delete(batch_2, k, axis=1)
        dist_ik = jnp.zeros_like(batch_2)  
        dist_kj = jnp.zeros_like(batch_2)

        for i in range(n-1):
            if(i < k):
                dist_ik = dist_ik.at[i].set(jnp.full(n-1, dist[i][k]))
            else:
                dist_ik = dist_ik.at[i].set(jnp.full(n-1, dist[i+1][k]))

        for j in range(n-1):
            if(j < k):
                dist_kj = dist_kj.at[:, j].set(jnp.full(n-1, dist[k][j]))
            else:
                dist_kj = dist_kj.at[:, j].set(jnp.full(n-1, dist[k][j+1]))
        # 能替换成这个吗？哪个效率更高？
        # dist_kj = dist_ik.

        # 把上三角拿出来算
        indices = numpy.triu_indices(batch_2.shape[0], k=1)
        batch_2_upper_triangle = batch_2[indices]
        dist_ik_upper_triangle = dist_ik[indices]
        dist_kj_upper_triangle = dist_kj[indices]

        batch_2_upper_triangle = jnp.minimum(batch_2_upper_triangle, dist_ik_upper_triangle + dist_kj_upper_triangle)

        # 把上三角放回去
        batch_2 = jnp.zeros_like(batch_2)
        batch_2 = batch_2.at[indices].set(batch_2_upper_triangle)
        batch_2 += batch_2.T

        batch_2 = jnp.insert(batch_2, k, col_k_without_dkk, axis=1)      # 把更新的值放回原位置
        batch_2 = jnp.insert(batch_2, k, dist[k], axis=0)
        dist = batch_2   
            
    return dist
