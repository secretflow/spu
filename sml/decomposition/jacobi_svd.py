import jax.numpy as jnp
import jax.random as random
import jax.lax as lax
from jax import jit, vmap
import numpy as np
import time


@jit
def jacobi_rotation(A, p, q):
    """Compute the Jacobi rotation parameters (c, s) for the elements A[p, q]."""
    tau = (A[q, q] - A[p, p]) / (2 * A[p, q])
    t = jnp.sign(tau) / (jnp.abs(tau) + jnp.sqrt(1 + tau**2))
    c = 1 / jnp.sqrt(1 + t**2)
    s = t * c
    return c, s


@jit
def apply_jacobi_rotation_A(A, c, s, p, q):
    """Apply the Jacobi rotation to matrix A and update its values."""
    A_new = A.copy()
    # Update rows p and q
    A = A.at[p, :].set(c * A_new[p, :] - s * A_new[q, :])
    A = A.at[q, :].set(s * A_new[p, :] + c * A_new[q, :])
    A_new = A.copy()
    # Update columns p and q
    A = A.at[:, p].set(c * A_new[:, p] - s * A_new[:, q])
    A = A.at[:, q].set(s * A_new[:, p] + c * A_new[:, q])
    return A


@jit
def jacobi_svd(A, tol=1e-10, max_iter=100):
    """
    Perform the Jacobi algorithm to compute the Singular Value Decomposition (SVD) of a symmetric matrix A.

    The algorithm iterates by selecting the largest off-diagonal element and applying a Jacobi rotation
    to zero it out. The process continues until the largest off-diagonal element is smaller than the specified tolerance or the maximum number of iterations is reached.

    Arguments:
    A -- input symmetric matrix (n x n)
    tol -- tolerance for stopping criteria (default 1e-10)
    max_iter -- maximum number of iterations (default 100)

    Note:
    For matrices of different sizes, the value of `max_iter` should be adjusted according to the matrix dimension. 
    Typically, as the size of the matrix `n` increases, `max_iter` should also be increased to ensure the algorithm can converge within a larger computational effort. 

    Returns:
    Singular values of matrix A in descending order.
    """
    n = A.shape[0]
    A = jnp.array(A)

    def body_fun(i, val):
        """Body function for the iterative loop to apply Jacobi rotations."""
        A, max_off_diag = val

        # Get the off-diagonal part of A
        A_no_diag = A - jnp.diag(jnp.diagonal(A))
        p, q = jnp.unravel_index(jnp.argmax(jnp.abs(A_no_diag)), A.shape)

        max_off_diag = jnp.abs(A[p, q])

        def apply_rotation(A):
            """Apply the Jacobi rotation to matrix A."""
            c, s = jacobi_rotation(A, p, q)
            A = apply_jacobi_rotation_A(A, c, s, p, q)
            return A

        # Conditionally apply rotation if the maximum off-diagonal element is greater than tolerance
        A, max_off_diag = lax.cond(
            max_off_diag < tol,
            lambda val: val,
            lambda val: (apply_rotation(val[0]), val[1]),
            (A, max_off_diag),
        )

        return A, max_off_diag

    # Initialize max_off_diag as a very large number
    max_off_diag = jnp.inf
    A, _ = lax.fori_loop(0, max_iter, body_fun, (A, max_off_diag))

    # Extract the singular values from the diagonal of A
    singular_values = jnp.abs(jnp.diag(A))

    # Sort singular values in descending order
    singular_values = jnp.sort(singular_values)[::-1]

    return singular_values
