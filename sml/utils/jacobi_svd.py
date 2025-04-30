# Copyright 2025 Ant Group Co., Ltd.
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
from jax import lax


def _jacobi_rotation(A, p, q):
    """
    Compute the Jacobi rotation parameters (c, s) to zero out the off-diagonal element A[p, q].

    Arguments:
        A -- input matrix.
        p, q -- indices of the pivot element.

    Returns:
        c, s -- cosine and sine of the rotation angle.
    """
    # Compute tau using the difference of diagonal elements and the off-diagonal element.
    tau = (A[q, q] - A[p, p]) / (2 * A[p, q] + 5e-6)
    # Calculate the tangent (t) of the rotation angle.
    t = jnp.sign(tau) / (jnp.abs(tau) + jnp.sqrt(1 + tau**2))
    # Compute cosine and sine values.
    c = 1 / jnp.sqrt(1 + t**2)
    s = t * c
    return c, s


def _apply_jacobi_rotation(A, V, c, s, p, q):
    """
    Apply the Jacobi rotation to update matrix A and accumulate the singular vectors in V.

    Arguments:
        A -- input matrix.
        V -- matrix accumulating the singular vectors.
        c, s -- cosine and sine of the rotation angle.
        p, q -- indices for the rotation.

    Returns:
        Updated matrices A and V.
    """
    A_new = A.copy()
    # Update rows p and q of A using the rotation parameters.
    A = A.at[p, :].set(c * A_new[p, :] - s * A_new[q, :])
    A = A.at[q, :].set(s * A_new[p, :] + c * A_new[q, :])

    A_new = A.copy()
    # Update columns p and q of A.
    A = A.at[:, p].set(c * A_new[:, p] - s * A_new[:, q])
    A = A.at[:, q].set(s * A_new[:, p] + c * A_new[:, q])

    V_new = V.copy()
    # Update the singular vector matrix V in the same way.
    V = V.at[:, p].set(c * V_new[:, p] - s * V_new[:, q])
    V = V.at[:, q].set(s * V_new[:, p] + c * V_new[:, q])

    return A, V


def jacobi_svd(A, max_iter=100, compute_uv=True):
    """
    Perform the Jacobi algorithm to compute the Singular Value Decomposition (SVD) of a symmetric matrix A.
    The algorithm iterates by selecting the largest off-diagonal element and applying a Jacobi rotation to zero it out.

    Arguments:
        A -- input symmetric matrix (n x n)
        max_iter -- maximum number of iterations (default 100)
        compute_uv -- whether to compute singular vectors U and V (default is True)

    Note:
        For matrices of different sizes, the value of max_iter should be adjusted according to the matrix dimension.
        Generally, as the matrix size n increases, max_iter should also be increased to ensure the algorithm has sufficient iterations to converge.
        Based on empirical testing, we recommend the following (n, max_iter) pairs:
          (n = 10, max_iter = 100)
          (n = 20, max_iter = 400)
          (n = 30, max_iter = 1200)
          (n = 40, max_iter = 2000)
          (n = 50, max_iter = 3000)
          (n = 100, max_iter = 12000)
        There is no need to worry that increasing max_iter will significantly impact the runtime.
        Our tests indicate that the algorithm converges rapidly, and even with a higher max_iter, the overall execution time does not increase noticeably.

    Returns:
        Singular values of matrix A in descending order.
        If compute_uv=True, returns (U, S, V);
        If compute_uv=False, returns only S.
    """
    # Determine matrix dimension and ensure input is a JAX array.
    n = A.shape[0]
    A = jnp.array(A)

    # Initialize V:
    # If computing singular vectors, start with the identity matrix.
    # Otherwise, initialize V as a zero matrix as a placeholder.
    if compute_uv:
        V = jnp.eye(n)  # Right singular vector matrix
    else:
        V = jnp.zeros((n, n))

    # Initialize state variables:
    # i: iteration counter, A: current matrix, V: singular vector matrix,
    init_state = (0, A, V)

    def cond_fun(state):
        """
        Continue iterating while:
          - The number of iterations is less than max_iter
        """
        i, A, V = state
        return i < max_iter

    def body_fun(state):
        """
        At each iteration:
          - Identify the largest off-diagonal element.
          - Compute the Jacobi rotation to zero it.
          - Update the matrix A and the singular vector matrix V.
        """
        i, A, V = state
        # Zero out the diagonal elements to focus on off-diagonal values.
        A_no_diag = A - jnp.diag(jnp.diag(A))
        # Find the indices (p, q) of the largest off-diagonal element.
        p, q = jnp.unravel_index(jnp.argmax(jnp.abs(A_no_diag)), A.shape)
        # Compute the rotation parameters.
        c, s = _jacobi_rotation(A, p, q)
        # Apply the rotation to update A and V.
        A_new, V_new = _apply_jacobi_rotation(A, V, c, s, p, q)
        # Return the updated state.
        return (i + 1, A_new, V_new)

    # Run the iterative process using JAX's while_loop.
    final_state = lax.while_loop(cond_fun, body_fun, init_state)
    _, A_final, V_final = final_state

    # Compute the singular values from the diagonal of A_final.
    S = jnp.abs(jnp.diag(A_final))
    U_final = V_final  # For symmetric matrices, U equals V.

    # Return the results based on the compute_uv flag.
    if compute_uv:
        return U_final, S, V_final.T
    else:
        return S
