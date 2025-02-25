import jax.numpy as jnp
import numpy as np
from jax import jit


def generate_ring_sequence(n):
    """ An algorithm for generating a fixed cyclic sequence
    Parameters
    ----------
    n : int
        The range of subscript pairs (i,j), where 0 < i,j < n.

    Returns
    -------
    index_pairs : list of shape (len, n//2), where len = n if n is odd, n-1 if n is even
        Output a list of n//2 non-common rows and columns of non-diagonal element subscripts (i, j) in each group.

    Examples
    -------
    n = 6: [[(0, 1), (2, 3), (4, 5)], [(1, 5), (0, 2), (3, 4)], [(3, 5), (1, 2), (0, 4)], [(0, 5), (1, 3), (2, 4)], [(2, 5), (0, 3), (1, 4)]]
    n = 5: [[(0, 1), (2, 3)], [(0, 2), (3, 4)], [(1, 2), (0, 4)], [(1, 3), (2, 4)], [(0, 3), (1, 4)]]
    """
    # init
    upper_row = list(range(0, n, 2))
    lower_row = list(range(1, n + 1, 2))

    index_pairs = []

    loop = n - (n - 1) % 2

    for step in range(loop):
        # record index
        pairs = []
        for i in range((n + 1) // 2):
            a, b = upper_row[i], lower_row[i]
            if max(a, b) < n:
                pairs.append((min(a, b), max(a, b)))

        index_pairs.append(pairs)

        swap = step // 2
        upper_row[swap], lower_row[swap] = lower_row[swap], upper_row[swap]

        # right shift
        lower_row = [lower_row[-1]] + lower_row[:-1]

    return index_pairs


def serial_jacobi_evd(
    A, 
    J,                      # rotate matrix，init as jnp.eye(A.shape[0])
    max_jacobi_iter  
):
    """ A Eigendecomposition algorithm using the Jacobi Method.
    Parameters
    ----------
    A : {array-like}, shape (n, n)
        The input matrix used to perform EVD. 
        Note that A must be a symmetric matrix, because the employed two-sided Jacobi-EVD algorithm only supports symmetric matrices.
    J : {array-like}, shape (n, n)
        The required rotation matrix, the initial input is the unit matrix jnp.eye(), which needs to be kept secret during execution.
    max_jacobi_iter : int, default=5/10
        Maximum number of iterations for Jacobi Method, larger numbers mean higher accuracy and more time consuming;
        for most cases, iter = 5 can reach convergence.

    Returns
    -------
    eigenvalues : ndarray of shape (n )
        Returns a list of all eigenvalues of A.
    eigenvectors: ndarray of shape (n, n)
        Returns all eigenvectors corresponding to the eigenvalues of A.

    References
    ----------
    .. Jacobi Method: https://en.wikipedia.org/wiki/Jacobi_eigenvalue_algorithm
    """

    n = A.shape[0]
    eigenvectors = jnp.eye(n)

    for _ in range(max_jacobi_iter):
        # Select n/2 pairs of non-diagonal elements that don't share rows/columns
        selected_pairs = generate_ring_sequence(n)

        for pair in selected_pairs:

            # Combine rotation matrices for selected pairs
            ks, ls = zip(*pair)
            k_list = jnp.array(ks)
            l_list = jnp.array(ls)
            mask = jnp.not_equal(A[k_list, l_list], 0)
            diff = A[l_list, l_list] - A[k_list, k_list]

            '''
            # fitting the rotate matrix elements by sqrt & rsqrt (derived from the trigonometric functions)
            # But the fitting accuracy is lower than the trigonometric method.
            
            # cos2theta = jnp.where(mask, diff / jnp.sqrt((4 * A[k_list, l_list]**2 + diff **2) ), 0)
            # cos_squrare = 0.5 * (1 + cos2theta)
            # sin_squrare = 0.5 * (1 - cos2theta)
            # combined_squares = jnp.stack([cos_squrare, sin_squrare], axis=0)
            # sqrt_combined = jnp.sqrt(combined_squares)
            # c = sqrt_combined[0]
            # s = sqrt_combined[1] * jnp.sign(A[k_list, l_list])
            '''

            # trigonometric method
            theta = jnp.where(mask, 0.5 * jnp.arctan2(2 * A[k_list, l_list], diff), 0)
            theta_cosine = 0.5 * jnp.pi - theta
            combined_theta = jnp.stack([theta_cosine, theta], axis=0)
            sin_combined = jnp.sin(combined_theta)
            c = sin_combined[0]
            s = sin_combined[1]

            J_combined = J.copy()
            # for i in range(len(pair)):
            #     if mask.at[i]:  # 此处mask为secret类型,使用 if else存在问题
            #         k, l = ks[i], ls[i]
            #         rows, cols, vals = (
            #             [k, l, k, l],
            #             [k, l, l, k],
            #             [c[i], c[i], s[i], -s[i]],
            #         )
            #         J_combined = J_combined.at[rows, cols].set(vals)
            rows, cols, vals = (
                [ks, ls, ks, ls],
                [ks, ls, ls, ks],
                [c, c, s, -s],
            )
            J_combined = J_combined.at[rows, cols].set(vals)

            # Update A and eigenvectors using the rotation matrix
            A = jnp.dot(J_combined.T, jnp.dot(A, J_combined))
            eigenvectors = jnp.dot(eigenvectors, J_combined)

            '''
            # Update the Matrix A with the mapping of the chosen rotation matrix and using the hardmard product instead of matrix multiplication
            
            # r = np.arange(n)
            # c_r = [1.0]*n
            # s_r = [0.0]*n

            # for i in range(len(pair)):
            #     r[ks[i]], r[ls[i]] = ls[i], ks[i]      
            #     if(mask.at[i]):
            #         k, l = ks[i], ls[i]
            #         c_r[k], c_r[l], s_r[l], s_r[k] = c[i], c[i], s[i], -s[i]
    
            # c_list = jnp.array(c_r)
            # s_list = jnp.array(s_r)

            # # Update matrix A and eigenvectors using hardmard product
            # A_row = (c_list * s_list[:,None]) * A[r,:]
            # A = (c_list * c_list[:,None]) * A + A_row + A_row.T + (s_list * s_list[:,None]) * A[r,:][:,r]
            # eigenvectors = c_list * eigenvectors + s_list * eigenvectors[:,r]
            '''

    eigenvalues = jnp.diag(A)
    return eigenvalues, eigenvectors



