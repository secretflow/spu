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

import warnings
from warnings import warn

import jax
import jax.numpy as jnp
from jax import jit, lax


def _pivot_col(T, tol=1e-5, bland=False):
    mask = T[-1, :-1] >= -tol

    all_masked = jnp.all(mask)

    bland_first_col = jnp.argmin(jnp.where(mask, jnp.inf, jnp.arange(T.shape[1] - 1)))
    # 定义根据最小值选择列的函数
    ma = jnp.where(mask, jnp.inf, T[-1, :-1])
    min_col = jnp.argmin(ma)

    result = jnp.where(bland, bland_first_col, min_col)

    valid = ~all_masked
    result = jnp.where(all_masked, 0, result)

    return valid, result


def _pivot_row(T, basis, pivcol, phase, tol=1e-5, bland=False):

    def true_mask_func(T, pivcol):
        mask = T[:-2, pivcol] <= tol
        ma = jnp.where(mask, jnp.inf, T[:-2, pivcol])
        mb = jnp.where(mask, jnp.inf, T[:-2, -1])

        q = jnp.where(ma == 1.75921860e13, jnp.inf, mb / ma)

        # 选择最小比值的行
        min_rows = jnp.nanargmin(q)
        all_masked = jnp.all(mask)
        return min_rows, all_masked

    def false_mask_func(T, pivcol):
        mask = T[:-1, pivcol] <= tol
        ma = jnp.where(mask, jnp.inf, T[:-1, pivcol])
        mb = jnp.where(mask, jnp.inf, T[:-1, -1])

        q = jnp.where(ma == 1.75921860e13, jnp.inf, mb / ma)

        # 选择最小比值的行
        min_rows = jnp.nanargmin(q)
        all_masked = jnp.all(mask)
        return min_rows, all_masked

    true_min_rows, true_all_masked = true_mask_func(T, pivcol)
    false_min_rows, false_all_masked = false_mask_func(T, pivcol)
    min_rows = jnp.where(phase == 1, true_min_rows, false_min_rows)
    all_masked = jnp.where(phase == 1, true_all_masked, false_all_masked)

    # 检查掩码数组是否全被掩盖
    has_valid_row = min_rows.size > 0
    row = min_rows

    # 处理全被掩盖的情况
    row = jnp.where(all_masked, 0, row)

    # 处理没有满足条件的行的情况
    row = jnp.where(has_valid_row, row, 0)

    return ~all_masked & has_valid_row, row


def _apply_pivot(T, basis, pivrow, pivcol, tol=1e-5):
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
    maxiter=300,
    tol=1e-5,
    phase=2,
    bland=False,
):
    status = 0
    complete = False

    num = 0
    pivcol = 0
    pivrow = 0
    while num < maxiter:
        pivcol_found, pivcol = _pivot_col(T, tol, bland)

        def cal_pivcol_found_True(
            T, basis, pivcol, phase, tol, bland, status, complete
        ):
            pivrow_found, pivrow = _pivot_row(T, basis, pivcol, phase, tol, bland)

            pivrow_isnot_found = pivrow_found == False
            status = jnp.where(pivrow_isnot_found, 1, status)
            complete = jnp.where(pivrow_isnot_found, True, complete)

            return pivrow, status, complete

        pivcol_isnot_found = pivcol_found == False
        pivcol = jnp.where(pivcol_isnot_found, 0, pivcol)
        pivrow = jnp.where(pivcol_isnot_found, 0, pivrow)
        status = jnp.where(pivcol_isnot_found, 0, status)
        complete = jnp.where(pivcol_isnot_found, True, complete)

        pivcol_is_found = pivcol_found == True
        pivrow_True, status_True, complete_True = cal_pivcol_found_True(
            T, basis, pivcol, phase, tol, bland, status, complete
        )

        pivrow = jnp.where(pivcol_is_found, pivrow_True, pivrow)
        status = jnp.where(pivcol_is_found, status_True, status)
        complete = jnp.where(pivcol_is_found, complete_True, complete)

        complete_is_False = complete == False
        apply_T, apply_basis = _apply_pivot(T, basis, pivrow, pivcol, tol)
        T = jnp.where(complete_is_False, apply_T, T)
        basis = jnp.where(complete_is_False, apply_basis, basis)
        num = num + 1

    return T, basis, status


def _linprog_simplex(c, A, b, c0=0, maxiter=300, tol=1e-5, bland=False):
    status = 0
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
    T, basis, status = _solve_simplex(
        T, n, basis, maxiter=maxiter, tol=tol, phase=1, bland=bland
    )

    status = jnp.where(jnp.abs(T[-1, -1]) < tol, status, 1)

    T_new = T[:-1, :]
    jit_delete = jit(jnp.delete, static_argnames=['assume_unique_indices'])
    T = jnp.delete(T_new, av, 1, assume_unique_indices=True)

    # phase 2
    T, basis, status = _solve_simplex(T, n, basis, maxiter, tol, 2, bland)

    solution = jnp.zeros(n + m)
    solution = solution.at[basis[:n]].set(T[:n, -1])
    x = solution[:m]

    return x, status
