import jax.numpy as jnp
from jax import jit, lax


def jacobi_rotation(A, p, q):
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


def apply_jacobi_rotation(A, V, c, s, p, q):
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


def jacobi_svd(A, tol=5e-6, tol_diag=5e-6, max_iter=100, compute_uv=True):
    """
    Perform the Jacobi algorithm to compute the Singular Value Decomposition (SVD) of a symmetric matrix A.

    The algorithm iterates by selecting the largest off-diagonal element and applying a Jacobi rotation to zero it out.
    The process continues until the largest off-diagonal element is smaller than the specified tolerance or the maximum number of iterations is reached.

    Arguments:
        A -- input symmetric matrix (n x n)
        tol -- off-diagonal convergence tolerance (default 5e-6)
        tol_diag -- diagonal change tolerance (default 5e-6)
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
    # max_off_diag: current largest off-diagonal value, prev_diag: previous diagonal values,
    # diff: change in diagonal values.
    prev_diag = jnp.diag(A)
    init_state = (0, A, V, jnp.inf, prev_diag, jnp.inf)

    def cond_fun(state):
        """
        Continue iterating while:
          - The number of iterations is less than max_iter, and
          - Either the maximum off-diagonal element is above tol or the change in diagonal is above tol_diag.
        """
        i, A, V, max_off_diag, prev_diag, diff = state
        return (i < max_iter) & ((max_off_diag >= tol) | (diff >= tol_diag))

    def body_fun(state):
        """
        At each iteration:
          - Identify the largest off-diagonal element.
          - Compute the Jacobi rotation to zero it.
          - Update the matrix A and the singular vector matrix V.
          - Compute the new diagonal and its difference from the previous iteration.
        """
        i, A, V, _, prev_diag, _ = state
        # Zero out the diagonal elements to focus on off-diagonal values.
        A_no_diag = A - jnp.diag(jnp.diag(A))
        # Find the indices (p, q) of the largest off-diagonal element.
        p, q = jnp.unravel_index(jnp.argmax(jnp.abs(A_no_diag)), A.shape)
        # Get the value of the largest off-diagonal element.
        current_off_diag = jnp.abs(A[p, q])
        # Compute the rotation parameters.
        c, s = jacobi_rotation(A, p, q)
        # Apply the rotation to update A and V.
        A_new, V_new = apply_jacobi_rotation(A, V, c, s, p, q)
        # Extract the new diagonal and calculate the difference from the previous diagonal.
        curr_diag = jnp.diag(A_new)
        diff_new = jnp.linalg.norm(curr_diag - prev_diag)
        # Return the updated state.
        return (i + 1, A_new, V_new, current_off_diag, curr_diag, diff_new)

    # Run the iterative process using JAX's while_loop.
    final_state = lax.while_loop(cond_fun, body_fun, init_state)
    _, A_final, V_final, _, _, _ = final_state

    # Compute the singular values from the diagonal of A_final.
    S = jnp.abs(jnp.diag(A_final))
    # Sort singular values in descending order.
    sorted_indices = jnp.argsort(S)[::-1]
    S = S[sorted_indices]

    # Rearrange the singular vectors based on the sorted indices.
    V_final = V_final[:, sorted_indices]
    U_final = V_final  # For symmetric matrices, U equals V.

    # Return the results based on the compute_uv flag.
    if compute_uv:
        return U_final, S, V_final.T
    else:
        return S


# The function is now JIT compiled with compute_uv as a static argument.
jacobi_svd = jit(jacobi_svd, static_argnames=("compute_uv",))
