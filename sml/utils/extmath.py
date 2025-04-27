# Copyright 2023 Ant Group Co., Ltd.
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
import numpy as np


def qr_Householder(A):
    return jnp.linalg.qr(A)


def qr_Gram_schmidt(A):
    Q = jnp.zeros_like(A)
    cnt = 0
    for a in A.T:
        u = jnp.copy(a)
        for i in range(0, cnt):
            u -= jnp.dot(jnp.dot(Q[:, i].T, a), Q[:, i])
        e = u / jnp.linalg.norm(u)
        Q = Q.at[:, cnt].set(e)
        cnt += 1
    return Q


def eigh_power(A, max_iter, rank=None):
    n = A.shape[0]
    if rank is None:
        rank = n
    eig_val = []
    eig_vec = []
    for _ in range(rank):
        vec = jnp.ones((n,))
        for _ in range(max_iter):
            vec = jnp.dot(A, vec)
            vec /= jnp.linalg.norm(vec)
        eigval = jnp.dot(vec.T, jnp.dot(A, vec))
        eig_vec.append(vec)
        eig_val.append(eigval)
        A -= eigval * jnp.outer(vec, vec)
    eig_vecs = jnp.column_stack(eig_vec)
    eig_vals = jnp.array(eig_val)
    return eig_vals, eig_vecs


def eigh_qr(A, max_iter):
    qr = []
    n = A.shape[0]
    Q = jnp.eye(n)
    for _ in range(max_iter):
        qr = jnp.linalg.qr(A)
        Q = jnp.dot(Q, qr[0])
        A = jnp.dot(qr[1], qr[0])
    AK = jnp.dot(qr[0], qr[1])
    eig_val = jnp.diag(AK)
    eig_vec = Q
    return eig_val, eig_vec


def rsvd_iteration(A, Omega, scale, power_iter):
    # iter = 7 if rank < 0.1 * min(A.shape) else 4
    Y = jnp.dot(A, Omega)

    for _ in range(power_iter):
        Y = jnp.dot(A, jnp.dot(A.T, Y))
    Q = qr_Gram_schmidt(Y / scale)
    return Q


def svd(A, eigh_iter):
    """
    Get the svd decomposition of matrix A using power iteration.

    Note:
        1. please make sure the matrix is either full row rank or full column rank.
        2. for A (m,n), k=min(m,n), get U (m,k), s (k,), Vt (k,n) like np.linalg.svd with ull_matrices=False
        3. this implementation is now not scalable for large matrix (accuracy and efficiency)

    Args:
        A (ndarray): the matrix to decomposition
        eigh_iter (int): iter nums for power iteration

    Returns:
        U, s, Vt
    """
    m, n = A.shape
    k = min(m, n)

    if m <= n:
        sigma, U = eigh_power(jnp.dot(A, A.T), eigh_iter, rank=k)
        sigma_sqrt = jnp.sqrt(sigma)
        sigma_rec_sqrt = 1 / sigma_sqrt
        V = jnp.dot(A.T, U) * sigma_rec_sqrt.reshape(1, k)
    else:
        sigma, V = eigh_power(jnp.dot(A.T, A), eigh_iter, rank=k)
        sigma_sqrt = jnp.sqrt(sigma)
        sigma_rec_sqrt = 1 / sigma_sqrt
        U = jnp.dot(A, V) * sigma_rec_sqrt.reshape((1, k))

    Vt = V.T

    return U, sigma_sqrt, Vt


def randomized_svd(
    A,
    n_components,
    n_oversamples,
    random_matrix,
    n_iter=4,
    scale=None,
    eigh_iter=300,
):
    if scale is None:
        scale = [10000000, 10000]
    assert random_matrix.shape == (
        A.shape[1],
        n_components + n_oversamples,
    ), f"Expected random_matrix to be ({A.shape[1]}, {n_components + n_oversamples}) array, got {random_matrix.shape}"
    Omega = random_matrix / scale[0]
    Q = rsvd_iteration(A, Omega, scale[1], n_iter)
    B = jnp.dot(Q.T, A)
    u_tilde, s, v = svd(B, eigh_iter)
    u = jnp.dot(Q, u_tilde)
    return u[:, :n_components], s[:n_components], v[:n_components, :]


def _generate_ring_sequence(n):
    """An algorithm for generating a fixed cyclic sequence
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


def serial_jacobi_evd(A, max_jacobi_iter=5):
    """An Eigen decomposition algorithm using the Jacobi Method.
    Parameters
    ----------
    A : {array-like}, shape (n, n)
        The input matrix used to perform EVD.
        Note that A must be a symmetric matrix, because the employed two-sided Jacobi-EVD algorithm only supports symmetric matrices.
    max_jacobi_iter : int, default=5/10
        Maximum number of iterations for Jacobi Method, larger numbers mean higher accuracy and more time consuming;
        for most cases, iter = 5 can reach convergence for 64*64 matrix.

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
        selected_pairs = _generate_ring_sequence(n)

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

            J_combined = jnp.eye(n)
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
