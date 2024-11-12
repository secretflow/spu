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

from sml.manifold.jacobi import Jacobi

def mds(D, num_samples, n_components):
    D_2 = jnp.square(D)
    B = jnp.zeros((num_samples, num_samples))
    B = -0.5 * D_2
    # 按行求和(使用英语的注释)
    dist_2_i = jnp.sum(B, axis=1)
    dist_2_i = dist_2_i / num_samples
    # 按列求和
    dist_2_j = dist_2_i.T
    # 全部求和
    dist_2 = jnp.sum(dist_2_i)
    dist_2 = dist_2 / (num_samples)
    for i in range(num_samples):
        for j in range(num_samples):
            B = B.at[i, j].set(B[i][j] - dist_2_i[i] - dist_2_j[j] + dist_2)

    values, vectors = Jacobi(B, num_samples)
    values = jnp.diag(values)
    values = jnp.array(values)
    values = jnp.expand_dims(values, axis=1).repeat(vectors.shape[1], axis=1)
    values,vectors=jax.lax.sort_key_val(values.T,vectors.T)
    vectors=vectors[:,num_samples - n_components:num_samples]
    values=values[0,num_samples - n_components:num_samples]
    values = jnp.sqrt(jnp.diag(values))
    
    ans = jnp.dot(vectors, values)

    return B, ans, values, vectors
