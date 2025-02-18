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
import numpy as np

import spu.intrinsic as si
from sml.manifold.jacobi import Jacobi


def se(X, num_samples, D, n_components):
    X, Q = Jacobi(X, num_samples)
    X = jnp.diag(X)
    X = jnp.array(X)

    # X2 = jnp.expand_dims(X, axis=1).repeat(Q.shape[1], axis=1)
    # X3, ans = jax.lax.sort_key_val(X2.T, Q.T)

    perm = jnp.argsort(X)
    ans=jnp.zeros((num_samples, num_samples))
    Q=Q.T
    for i in range(num_samples):
        temp_pi = jnp.arange(num_samples)
        per_Q = si.permute(Q[i], si.permute(temp_pi, perm))
        for j in range(num_samples):
            ans = ans.at[i, j].set(per_Q[j])
    
    ans = ans[:, 1 : n_components + 1]
    
    
    D = jnp.diag(D)
    ans = ans.T * jnp.reciprocal(jnp.sqrt(D))
    return ans.T

def normalization(
    adjacency,
    norm_laplacian=True,  # If True, use symmetric normalized Laplacian matrix; If False, use a non normalized Laplacian matrix.
):
    D = jnp.sum(adjacency, axis=1)
    D = jnp.diag(D)

    L = D - adjacency
    D2 = jnp.diag(jnp.reciprocal(jnp.sqrt(jnp.diag(D))))
    if norm_laplacian == True:
        # normalization
        L = jnp.dot(D2, L)
        L = jnp.dot(L, D2)
    return D, L
