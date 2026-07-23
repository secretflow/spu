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
from jax import jit, lax


def _pivot_col(T, tol=1e-5):
    mask = T[-1, :-1] >= -tol

    all_masked = jnp.all(mask)

    ma = jnp.where(mask, jnp.inf, T[-1, :-1])
    min_col = jnp.argmin(ma)

    valid = ~all_masked
    result = jnp.where(all_masked, 0, min_col)

    return valid, result


def _pivot_row(T, pivcol, phase, tol=1e-5, max_val=1e10):
    if phase == 1:
        k = 2
    else:
        k = 1

    mask = T[:-k, pivcol] <= tol
    ma = jnp.where(mask, jnp.inf, T[:-k, pivcol])
    mb = jnp.where(mask, jnp.inf, T[:-k, -1])

    q = jnp.where(ma >= max_val, jnp.inf, mb / ma)

    min_rows = jnp.nanargmin(q)
    all_masked = jnp.all(mask)

    row = min_rows
    row = jnp.where(all_masked, 0, row)

    return ~all_masked, row


def _apply_pivot(T, basis, pivrow, pivcol):
    pivrow = jnp.int32(pivrow)
    pivcol = jnp.int32(pivcol)

    basis = basis.at[pivrow].set(pivcol)

    pivrow_one_hot = jax.nn.one_hot(pivrow, T.shape[0])
    pivcol_one_hot = jax.nn.one_hot(pivcol, T.shape[1])

    pivval = jnp.dot(pivrow_one_hot, jnp.dot(T, pivcol_one_hot))

    updated_row = T[pivrow] / pivval
    T = pivrow_one_hot[:, None] * updated_row + T * (1 - pivrow_one_hot[:, None])

    scalar = jnp.dot(T, pivcol_one_hot).reshape(-1, 1)

    updated_T = T - scalar * T[pivrow]

    row_restore_matrix = pivrow_one_hot[:, None] * T[pivrow]
    updated_T = row_restore_matrix + updated_T * (1 - pivrow_one_hot[:, None])

    return updated_T, basis


def _solve_simplex(
    T,
    n,
    basis,
    maxiter=100,
    tol=1e-5,
    max_val=1e10,
    phase=2,
):
    complete = False

    num = 0
    pivcol = 0
    pivrow = 0
    while num < maxiter:
        pivcol_found, pivcol = _pivot_col(T, tol)

        def cal_pivcol_found_True(T, pivcol, phase, tol, complete):
            pivrow_found, pivrow = _pivot_row(T, pivcol, phase, tol, max_val)

            pivrow_isnot_found = pivrow_found == False
            complete = jnp.where(pivrow_isnot_found, True, complete)

            return pivrow, complete

        pivcol_is_found = pivcol_found == True
        pivrow_True, complete_True = cal_pivcol_found_True(
            T, pivcol, phase, tol, complete
        )

        pivrow = jnp.where(pivcol_is_found, pivrow_True, 0)

        complete = jnp.where(pivcol_is_found, complete_True, complete)

        complete_is_False = complete == False
        apply_T, apply_basis = _apply_pivot(T, basis, pivrow, pivcol)
        T = jnp.where(complete_is_False, apply_T, T)
        basis = jnp.where(complete_is_False, apply_basis, basis)
        num = num + 1

    return T, basis


def _linprog_simplex(c, A, b, c0=0, maxiter=300, tol=1e-5, max_val=1e10):
    n, m = A.shape

    # All constraints must have b >= 0.
    is_negative_constraint = jnp.less(b, 0)
    A = jnp.where(is_negative_constraint[:, None], A * -1, A)
    b = jnp.where(is_negative_constraint, b * -1, b)

    av = jnp.arange(n) + m
    basis = av.copy()

    row_constraints = jnp.hstack((A, jnp.eye(n), b[:, jnp.newaxis]))
    row_objective = jnp.hstack((c, jnp.zeros(n), c0))
    row_pseudo_objective = -row_constraints.sum(axis=0)
    row_pseudo_objective = row_pseudo_objective.at[av].set(0)
    T = jnp.vstack((row_constraints, row_objective, row_pseudo_objective))

    # phase 1
    T, basis = _solve_simplex(
        T, n, basis, maxiter=maxiter, tol=tol, max_val=max_val, phase=1
    )

    T_new = T[:-1, :]
    T = jnp.delete(T_new, av, 1, assume_unique_indices=True)

    # phase 2
    T, basis = _solve_simplex(
        T, n, basis, maxiter=maxiter, tol=tol, max_val=max_val, phase=2
    )

    solution = jnp.zeros(n + m)
    solution = solution.at[basis[:n]].set(T[:n, -1])
    x = solution[:m]

    return x
