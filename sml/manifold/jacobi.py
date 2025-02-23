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


def compute_elements(X, k, l, num_samples):
    # Calculate the rotation angle for the element X[k][l].
    tar_elements = X[k][l]
    tar_diff = X[k][k] - X[l][l]

    cos_2theta = jnp.reciprocal(
        jnp.sqrt(1 + 4 * jnp.square(tar_elements * jnp.reciprocal(tar_diff)))
    )
    cos2 = 0.5 + 0.5 * cos_2theta
    sin2 = 0.5 - 0.5 * cos_2theta
    flag_zero = jnp.equal(tar_elements, 0)
    cos = jnp.sqrt(cos2) * (1 - flag_zero) + flag_zero
    sin = (
        (jnp.where(jnp.logical_and(tar_elements == 0, tar_diff == 0), 0, 1))
        * jnp.sqrt(sin2)
        * ((jnp.greater(tar_elements * tar_diff, 0)) * 2 - 1)
    )

    return cos, sin


def Rotation_Matrix(X, k, l, num_samples):
    # Calculate the rotation matrix J based on the selected X [k] [l]
    J = jnp.eye(num_samples)
    k_values = jnp.array(k)
    l_values = jnp.array(l)

    # Parallelize using vmap
    cos_values, sin_values = jax.vmap(compute_elements, in_axes=(None, 0, 0, None))(
        X, k_values, l_values, num_samples
    )

    J = J.at[k, k].set(cos_values)
    J = J.at[l, l].set(cos_values)
    J = J.at[k, l].set(-sin_values)
    J = J.at[l, k].set(sin_values)

    return J


# Ref:
# https://arxiv.org/abs/2105.07612
def Jacobi(X, num_samples):
    Q = jnp.eye(num_samples)
    k = 0
    # Generally, convergence is achieved within five iterations.
    while k < 5:
        # For each iteration, it is necessary to rotate all elements in the lower triangular part.
        # To reduce the number of rounds, elements with non-repeating indices should be rotated in parallel as much as possible.
        for i in range(1, num_samples + (num_samples - 1) // 2):
            if i < num_samples:
                l_0 = i
                r_0 = 0
            else:
                l_0 = num_samples - 1
                r_0 = i - l_0

            n = (l_0 - r_0 - 1) // 2 + 1

            j_indices = jnp.arange(n)
            l = l_0 - j_indices
            r = r_0 + j_indices

            if i < num_samples // 2:
                l_1 = num_samples - 1 - r_0
                r_1 = num_samples - 1 - l_0
                n = (l_1 - r_1 - 1) // 2 + 1
                j_indices = jnp.arange(n)
                l_1 = l_1 - j_indices
                r_1 = r_1 + j_indices
                l = jnp.concatenate([l, l_1])
                r = jnp.concatenate([r, r_1])

            # Calculate rotation matrix
            J = Rotation_Matrix(X, l, r, num_samples)
            # Update X and Q with rotation matrix
            X = jnp.dot(J.T, jnp.dot(X, J))
            Q = jnp.dot(J.T, Q)
        k = k + 1

    return X, Q
