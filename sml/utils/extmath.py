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
    sigma, U = eigh_power(jnp.dot(A, A.T), eigh_iter)
    sigma_sqrt = jnp.sqrt(sigma)
    sigma_inv = jnp.diag((1 / sigma_sqrt))
    V = jnp.dot(jnp.dot(sigma_inv, U.T), A)
    return (U, sigma_sqrt, V)


def randomized_svd(
    A,
    n_components,
    random_matrix,
    n_iter=4,
    scale=None,
    eigh_iter=100,
):
    if scale is None:
        scale = [10000000, 10000]
    assert random_matrix.shape == (
        A.shape[1],
        n_components,
    ), f"Expected random_matrix to be ({A.shape[1]}, {n_components}) array, got {random_matrix.shape}"
    Omega = random_matrix / scale[0]
    Q = rsvd_iteration(A, Omega, scale[1], n_iter)
    B = jnp.dot(Q.T, A)
    u_tilde, s, v = svd(B, eigh_iter)
    u = jnp.dot(Q, u_tilde)
    return u, s, v
