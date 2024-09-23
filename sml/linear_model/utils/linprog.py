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


def _pivot_col(T, tol=1e-9, bland=False):
    # 创建掩码数组
    mask = T[-1, :-1] >= -tol

    # 检查掩码数组是否全被掩盖
    all_masked = jnp.all(mask)

    # 定义根据 Bland's 规则选择第一个未被掩盖元素的函数
    bland_first_col = jnp.argmin(jnp.where(mask, jnp.inf, jnp.arange(T.shape[1] - 1)))
    # 定义根据最小值选择列的函数
    ma = jnp.where(mask, jnp.inf, T[-1, :-1])
    min_col = jnp.argmin(ma)

    result = jnp.where(bland, bland_first_col, min_col)

    # 处理全被掩盖的情况
    valid = ~all_masked
    result = jnp.where(all_masked, 0, result)

    return valid, result


# 由于非密态下q存在nan，而密态下nan都为0,已解决
def _pivot_row(T, basis, pivcol, phase, tol=1e-9, bland=False):

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


# def _apply_pivot(T, basis, pivrow, pivcol, tol=1e-9):
#     pivrow = jnp.int32(pivrow)
#     basis = basis.at[pivrow].set(pivcol)
#     pivval = T[pivrow, pivcol]
#     T = T.at[pivrow].set(T[pivrow] / pivval)

#     # 向量化更新所有行
#     pivrow_vector = T[pivrow]
#     scalar = T[:, pivcol].reshape(-1, 1)  # 获取每行的标量，形状为 (n, 1)

#     # 使用矩阵减法进行批量更新，避免循环
#     updated_T = T - scalar * pivrow_vector

#     # 由于主元行已经被更新过，因此我们需要恢复该行
#     updated_T = updated_T.at[pivrow].set(T[pivrow])
#     # updated_T = T
#     return updated_T, basis


def _apply_pivot(T, basis, pivrow, pivcol, tol=1e-9):
    # 将 pivrow 和 pivcol 转换为 int32 类型
    pivrow = jnp.int32(pivrow)
    pivcol = jnp.int32(pivcol)

    # 更新 basis 数组
    basis = basis.at[pivrow].set(pivcol)

    # 生成 one-hot 向量
    pivrow_one_hot = jax.nn.one_hot(pivrow, T.shape[0])
    pivcol_one_hot = jax.nn.one_hot(pivcol, T.shape[1])

    # 计算主元值
    pivval = jnp.dot(pivrow_one_hot, jnp.dot(T, pivcol_one_hot))

    # 更新主元行
    # T = T.at[pivrow].set(T[pivrow] / pivval)
    updated_row = T[pivrow] / pivval
    T = pivrow_one_hot[:, None] * updated_row + T * (1 - pivrow_one_hot[:, None])

    # 计算需要更新的行
    scalar = jnp.dot(T, pivcol_one_hot).reshape(-1, 1)  # 获取每行的标量，形状为 (n, 1)

    # 使用矩阵减法进行批量更新，避免循环
    updated_T = T - scalar * T[pivrow]

    # 由于主元行已经被更新过，因此我们需要恢复该行
    # updated_T = updated_T.at[pivrow].set(T[pivrow])
    row_restore_matrix = pivrow_one_hot[:, None] * T[pivrow]
    updated_T = row_restore_matrix + updated_T * (1 - pivrow_one_hot[:, None])

    return updated_T, basis


def _solve_simplex(
    T,
    n,
    basis,
    maxiter=1000,
    tol=1e-9,
    phase=2,
    bland=False,
    nit0=0,
):
    # 删除callback参数，删除postsolve_args参数
    nit = nit0
    status = 0
    message = ''
    complete = False

    def func_row(T, pivrow, basis, tol, nit):
        return T, pivrow, basis, tol, nit

    phase_is_2 = phase == 2
    func_row_T, func_row_pivrow, func_row_basis, func_row_tol, func_row_nit = func_row(
        T, 0, basis, tol, nit
    )
    # T = jnp.where(phase_is_2, func_row_T, T)
    # pivrow = jnp.where(phase_is_2, func_row_pivrow, 0)
    # basis = jnp.where(phase_is_2, func_row_basis, basis)
    # tol = jnp.where(phase_is_2, func_row_tol, tol)

    # nit = jnp.where(phase_is_2, func_row_nit, nit)

    pivcol = 0
    pivrow = 0

    for _ in range(maxiter):
        pivcol_found, pivcol = _pivot_col(T, tol, bland)
        pivrow_found = False

        def cal_pivcol_found_True(
            T, basis, pivcol, phase, tol, bland, status, complete
        ):
            pivrow_found, pivrow = _pivot_row(T, basis, pivcol, phase, tol, bland)

            pivrow_isnot_found = pivrow_found == False
            status = jnp.where(pivrow_isnot_found, 3, status)
            complete = jnp.where(pivrow_isnot_found, True, complete)

            return pivrow_found, pivrow, status, complete

        pivcol_isnot_found = pivcol_found == False
        pivcol = jnp.where(pivcol_isnot_found, 0, pivcol)
        pivrow = jnp.where(pivcol_isnot_found, 0, pivrow)
        status = jnp.where(pivcol_isnot_found, 0, status)
        complete = jnp.where(pivcol_isnot_found, True, complete)

        pivcol_is_found = pivcol_found == True
        pivrow_found_True, pivrow_True, status_True, complete_True = (
            cal_pivcol_found_True(T, basis, pivcol, phase, tol, bland, status, complete)
        )
        pivrow_found = jnp.where(pivcol_is_found, pivrow_found_True, pivrow_found)
        pivrow = jnp.where(pivcol_is_found, pivrow_True, pivrow)
        status = jnp.where(pivcol_is_found, status_True, status)
        complete = jnp.where(pivcol_is_found, complete_True, complete)

        # 瓶颈在这里
        def cal_ifnot_complete(T, basis, nit, status, complete, maxiter):
            nit_greater_maxiter = nit >= maxiter
            status = jnp.where(nit_greater_maxiter, 1, status)
            complete = jnp.where(nit_greater_maxiter, True, complete)

            apply_T, apply_basis = _apply_pivot(T, basis, pivrow, pivcol, tol)
            # apply_T = T
            # apply_basis = basis

            T = jnp.where(nit_greater_maxiter, T, apply_T)
            basis = jnp.where(nit_greater_maxiter, basis, apply_basis)
            nit = jnp.where(nit_greater_maxiter, nit, nit + 1)

            return T, basis, nit, status, complete

        complete_is_False = complete == False
        (
            ifnot_complete_T,
            ifnot_complete_basis,
            ifnot_complete_nit,
            ifnot_complete_status,
            ifnot_complete_complete,
        ) = cal_ifnot_complete(T, basis, nit, status, complete, maxiter)

        T = jnp.where(complete_is_False, ifnot_complete_T, T)

        basis = jnp.where(complete_is_False, ifnot_complete_basis, basis)
        # basis = jnp.where(complete_is_False, basis, basis)

        nit = jnp.where(complete_is_False, ifnot_complete_nit, nit)
        status = jnp.where(complete_is_False, ifnot_complete_status, status)
        complete = jnp.where(complete_is_False, ifnot_complete_complete, complete)

    return T, basis, nit, status


def _linprog_simplex(
    c, A, b, c0=0, maxiter=1000, tol=1, disp=False, bland=False, **unknown_options
):
    # 删除参数callback, postsolve_args
    status = 0
    messages = {
        0: "Optimization terminated successfully.",
        1: "Iteration limit reached.",
        2: "Optimization failed. Unable to find a feasible" " starting point.",
        3: "Optimization failed. The problem appears to be unbounded.",
        4: "Optimization failed. Singular matrix encountered.",
    }

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
    # T, basis, nit1, status = _solve_simplex(
    #     T, n, basis, maxiter=maxiter, tol=tol, phase=1, bland=bland
    # )
    nit1 = 1

    nit2 = nit1

    status = jnp.where(jnp.abs(T[-1, -1]) < tol, status, 2)

    original_shape = T.shape

    T_new = T[:-1, :]
    jit_delete = jit(jnp.delete, static_argnames=['assume_unique_indices'])
    # TODO: just use index ?
    T = jnp.delete(T_new, av, 1, assume_unique_indices=True)

    # def recover_tensor(T_recovered, T_shape, T_new_shape):
    #     # 根据保存的形状信息恢复原形状

    #     T_recovered = lax.dynamic_slice(
    #         T_recovered, (0, 0), (T_new_shape[0], T_new_shape[1])
    #     )

    #     return T_recovered

    # # phase 2
    (
        _solve_simplex_T,
        _solve_simplex_basis,
        _solve_simplex_nit2,
        _solve_simplex_status,
    ) = _solve_simplex(T, n, basis, maxiter, tol, 2, bland, nit1)

    status_is_0 = status == 0
    T = jnp.where(status_is_0, _solve_simplex_T, T)
    basis = jnp.where(status_is_0, _solve_simplex_basis, basis)
    nit2 = jnp.where(status_is_0, _solve_simplex_nit2, nit2)
    status = jnp.where(status_is_0, _solve_simplex_status, status)

    solution = jnp.zeros(n + m)
    solution = solution.at[basis[:n]].set(T[:n, -1])
    x = solution[:m]
    return x, status, nit2
