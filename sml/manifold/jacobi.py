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

MAX_ITERATIONS = 5  # Generally, convergence is achieved within five iterations.


def compute_elements(X, k, l, num_samples):
    """Compute the rotation angle (cosine and sine) for the element X[k][l] in a matrix.

    This function calculates the cosine and sine values required for a Jacobi rotation to eliminate the off-diagonal element X[k][l].

    Args:
        X: The input matrix for which the rotation angle is computed.
        k: The row index of the target element.
        l: The column index of the target element.
        num_samples: The size of the matrix (number of rows/columns).

    Returns:
        cos: The cosine of the rotation angle.
        sin: The sine of the rotation angle.
    """
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


def rotation_matrix(X, k, l, num_samples):
    """
    Compute the Jacobi rotation matrix for eliminating the off-diagonal element X[k][l].

    This function constructs a Jacobi rotation matrix `J`, which is used in the Jacobi eigenvalue algorithm
    to zero out the off-diagonal element at position (k, l). The rotation is determined by computing the
    cosine and sine values using the `compute_elements` function.

    Args:
        X: The input matrix for which the rotation matrix is computed.
        k: The row index of the target off-diagonal element.
        l: The column index of the target off-diagonal element.
        num_samples: The size of the matrix (number of rows/columns).

    Returns:
        J: The Jacobi rotation matrix of size (num_samples, num_samples).
    """
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
    """
    Perform Jacobi eigenvalue decomposition on matrix X.

    This function applies the Jacobi method to iteratively diagonalize the input matrix X.
    The method rotates elements in the lower triangular part to eliminate off-diagonal elements.
    It uses parallelized rotations to improve efficiency.

    Args:
        X: The input symmetric matrix (num_samples x num_samples) to be diagonalized.
        num_samples: The size of the matrix (number of rows/columns).

    Returns:
        X: The diagonalized matrix (eigenvalues on the diagonal).
        Q: The accumulated orthogonal transformation matrix (eigenvectors).
    """
    Q = jnp.eye(num_samples)
    k = 0
    while k < MAX_ITERATIONS:
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
            J = rotation_matrix(X, l, r, num_samples)
            # Update X and Q with rotation matrix
            X = jnp.dot(J.T, jnp.dot(X, J))
            Q = jnp.dot(J.T, Q)
        k = k + 1

    return X, Q
