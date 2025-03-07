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
import jax.numpy as jnp

import spu.intrinsic as si
from sml.manifold.jacobi import Jacobi


def mds(D, num_samples, n_components):
    D_2 = jnp.square(D)
    B = jnp.zeros((num_samples, num_samples))
    B = -0.5 * D_2
    # Sum by row
    dist_2_i = jnp.sum(B, axis=1)
    dist_2_i = dist_2_i / num_samples
    # Sum by column
    dist_2_j = dist_2_i.T
    # sum all
    dist_2 = jnp.sum(dist_2_i)
    dist_2 = dist_2 / (num_samples)
    for i in range(num_samples):
        for j in range(num_samples):
            B = B.at[i, j].set(B[i][j] - dist_2_i[i] - dist_2_j[j] + dist_2)

    values, vectors = Jacobi(B, num_samples)

    values = jnp.diag(values)
    values = jnp.array(values)

    # Retrieve the largest n_components values and their corresponding vectors.
    # Sort each column of vectors according to values.
    vectors = vectors.T
    Index_value = jnp.argsort(values)
    values = si.perm(values, Index_value)
    for i in range(num_samples):
        per_vectors = si.perm(vectors[i], Index_value)
        for j in range(num_samples):
            vectors = vectors.at[i, j].set(per_vectors[j])

    vectors = vectors[:, num_samples - n_components : num_samples]
    values = values[num_samples - n_components : num_samples]
    values = jnp.sqrt(values)
    values = jnp.diag(values)
    ans = jnp.dot(vectors, values)
    ans = ans[:, ::-1]

    return B, ans, values, vectors
