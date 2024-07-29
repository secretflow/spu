# Copyright 2023 Ant Group Co., Ltd.
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
from warnings import warn  
import warnings
from jax import lax
from jax import jit
import jax

# 这个是使用if-else实现的linprog_simplex函数，但是
# 由于jax中if-else和while-loop的限制，需使用lax.cond函数，所以
# 还写了一版使用lax.cond实现的linprog_simplex函数，见test.py
# 但由于test.py中的_pivot_row函数中获得min_rows的时候，数据为tracer
# 报错：jax.errors.ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: traced array with shape int32[].
# 所以暂时无法运行
# 目前完成linprog后可以替换quantile.py的代码，之后完成tests和emulations
# 先去看adaboost 2024.7.11

# 现在的问题是if语句导致的修改矩阵T大小不知道咋处理,jit编译后报错因为都是tracer 7.14
# 解决办法是删除if判断
# 但是出现新的错误,不会处理


# from collections import namedtuple

# _LPProblem = namedtuple('_LPProblem',
#                         'c A_ub b_ub A_eq b_eq bounds x0 integrality')
# _LPProblem.__new__.__defaults__ = (None,) * 7  # make c the only required arg


def _pivot_col(T, tol=1e-9, bland=False):
    # 创建掩码数组
    mask = T[-1, :-1] >= -tol

    # 定义根据 Bland's 规则选择第一个未被掩盖元素的函数
    def bland_func(_):
        return jnp.argmin(jnp.where(mask, jnp.inf, jnp.arange(T.shape[1] - 1)))
    
    # 定义根据最小值选择列的函数
    def min_func(_):
        ma = jnp.where(mask, jnp.inf, T[-1, :-1])
        return jnp.argmin(ma)
    
    # 检查掩码数组是否全被掩盖
    all_masked = jnp.all(mask)

    # 使用 jax.lax.cond 根据条件选择函数
    result = lax.cond(bland, bland_func, min_func, operand=None)
    
    # 将返回值转换为浮点数类型，以匹配 NaN 的类型
    result = result
    
    # 使用 jax.lax.cond 处理全被掩盖的情况
    valid = ~all_masked

    result = lax.cond(all_masked, lambda _: 0, lambda _: result, operand=None)
    
    return valid, result

def _pivot_row(T, basis, pivcol, phase, tol=1e-9, bland=False):  
    # pivcol = lax.cond(jnp.isnan(pivcol), lambda _: jnp.nan, lambda _: jnp.float32(pivcol), operand=None)
    # pivcol = jnp.floor(pivcol).astype(jnp.int32)
    # if jnp.isnan(pivcol):
    #     pivcol = jnp.nan
    # else:
    #     pivcol = pivcol.astype(jnp.int32)
    # k = lax.cond(phase == 1, lambda _: 2, lambda _: 1, operand=None)
    # k = 1
    def true_mask_func():
        mask = T[:-2, pivcol] <= tol
        ma = jnp.where(mask, jnp.inf, T[:-2, pivcol])
        mb = jnp.where(mask, jnp.inf, T[:-2, -1])

        q = mb / ma
        # 选择最小比值的行
        min_rows = jnp.nanargmin(q)
        all_masked = jnp.all(mask)
        return min_rows, all_masked
    
    def false_mask_func():
        mask = T[:-1, pivcol] <= tol
        ma = jnp.where(mask, jnp.inf, T[:-1, pivcol])
        mb = jnp.where(mask, jnp.inf, T[:-1, -1])

        q = mb / ma
        # 选择最小比值的行
        min_rows = jnp.nanargmin(q)
        all_masked = jnp.all(mask)
        return min_rows, all_masked
    
    min_rows, all_masked = lax.cond(phase==1, true_mask_func, false_mask_func)

    # 定义处理全被掩盖情况的函数
    def all_masked_func(_):
        return 0

    # 定义选择最小比值行的函数
    def bland_func(_):
        return min_rows
        # return min_rows[jnp.argmin(jnp.take(basis, min_rows))]
    
    def min_row_func(_):
        return min_rows

    # 检查掩码数组是否全被掩盖
    # all_masked = jnp.all(mask)
    has_valid_row = min_rows.size > 0
    
    row = lax.cond(bland, bland_func, min_row_func, operand=None)
    
    row = row

    # 使用 jax.lax.cond 处理全被掩盖的情况
    row = lax.cond(all_masked, all_masked_func, lambda _: row, operand=None)
    
    # 使用 jax.lax.cond 处理没有满足条件的行的情况
    row = lax.cond(has_valid_row, lambda r: r, lambda _: 0, row)
    
    return ~all_masked & has_valid_row, row

def _apply_pivot(T, basis, pivrow, pivcol, tol=1e-9):

    basis = basis.at[pivrow].set(pivcol)
    pivval = T[pivrow, pivcol]
    T = T.at[pivrow].set(T[pivrow] / pivval)

    def update_row(irow, T, pivrow, pivcol):
        pivrow_vector = T[pivrow]  # shape (n,)
        scalar = T[irow, pivcol]   # shape ()
        # print(f"pivrow_vector shape: {pivrow_vector.shape}, scalar shape: {scalar.shape}")
        updated_row = T[irow] - pivrow_vector * scalar
        T = T.at[irow].set(updated_row)
        return T
    
    def not_update_row(irow, T, pivrow, pivcol):
        return T

    # Update all other rows
    def condition(state):
        irow, T, pivrow, pivcol= state
        return irow < T.shape[0]

    def body(state):
        irow, T, pivrow, pivcol = state
        T = lax.cond(irow != pivrow,
                    # lambda _: update_row(irow, T, pivrow, pivcol),
                    # lambda _: T,
                    lambda _: update_row(irow, T, pivrow, pivcol),
                    lambda _: not_update_row(irow, T, pivrow, pivcol),
                    operand = None)
        return irow + 1, T, pivrow, pivcol
            
    state = 0, T, pivrow, pivcol
    irow, T, pivrow, pivcol = lax.while_loop(cond_fun=condition, body_fun=body, init_val=state)
    
    # if jnp.isclose(pivval, tol, atol=0, rtol=1e4):
    #     message = (
    #         f"The pivot operation produces a pivot value of:{pivval: .1e}, "
    #         "which is only slightly greater than the specified "
    #         f"tolerance{tol: .1e}. This may lead to issues regarding the "
    #         "numerical stability of the simplex method. "
    #         "Removing redundant constraints, changing the pivot strategy "
    #         "via Bland's rule or increasing the tolerance may "
    #         "help reduce the issue.")
    #     warn(message, stacklevel=5)

    return T, basis

def _solve_simplex(T, n, basis,
                   maxiter=1000, tol=1e-9, phase=2, bland=False, nit0=0,
                   ):
    # 删除callback参数，删除postsolve_args参数
    nit = nit0
    status = 0
    message =''
    complete = False
    # a,b = T_new_shape
    # assert phase in [1, 2],"Argument 'phase' to _solve_simplex must be 1 or 2"
    m = lax.cond(phase == 1, lambda _: T.shape[1]-2, lambda _: T.shape[1]-1, operand=None)
    # print(m)
    def func_col(T, pivrow, basis, tol, nit):
        def body_fun(carry):
            col, pivrow, T, basis, nit, found = carry
            
            def apply_pivot_and_update(T, basis, pivrow, col, nit, tol):
                pivcol = col
                T, basis = _apply_pivot(T, basis, pivrow, pivcol, tol)
                nit = nit + 1
                return T, basis, nit
            
            T, basis, nit = lax.cond(abs(T[pivrow, col]) > tol,
                            lambda _: apply_pivot_and_update(T, basis, pivrow, col, nit, tol),
                            lambda _: (T, basis, nit),
                            (T, basis, nit))
            
            found = found | (jnp.abs(T[pivrow, col]) > tol)
            
            return col + 1, pivrow, T, basis, nit, found

        def cond_fun(carry):
            col, pivrow, T, basis, nit, found = carry
            return jnp.logical_and(col < T.shape[1] - 1, ~found)

        _, _, T, basis, nit, _ = lax.while_loop(cond_fun, body_fun, (0, pivrow, T, basis, nit, False))

        return T, basis, nit
      
    def func_row(T, pivrow, basis, tol, nit):
        def body_fun(carry):
            T, pivrow, basis, tol, nit = carry
            T, basis, nit = lax.cond(basis[pivrow] > T.shape[1] - 2,
                            lambda _: func_col(T, pivrow, basis, tol, nit),
                            lambda _: (T, basis, nit),
                            (T, basis, nit))
            return T, pivrow + 1, basis, tol, nit        
            
        def cond_fun(carry):
            T, pivrow, basis, tol, nit = carry
            return pivrow < basis.size
          
        T, pivrow, basis, tol, nit = lax.while_loop(cond_fun, body_fun, (T, pivrow, basis, tol, nit))
        return T, pivrow, basis, tol, nit
          
    T, pivrow, basis, tol, nit = lax.cond(phase == 2,
                                          lambda _: func_row(T, 0, basis, tol, nit),
                                          lambda _: (T, 0, basis, tol, nit), 
                                          (T, 0, basis, tol, nit))
    
    def cond_ifnot_complete(carry):
        T, basis, pivcol, pivrow, phase, tol, bland, status, complete, nit, maxiter = carry
        return ~complete
    # 这个complete不会更新,一直为False
    
    def body_ifnot_complete(carry):
        T, basis, pivcol, pivrow, phase, tol, bland, status, complete, nit, maxiter = carry
        pivcol_found, pivcol = _pivot_col(T, tol, bland)
        pivrow_found = False
        def cal_pivcol_found_True(T, basis, pivcol, phase, tol, bland, status, complete):
            pivrow_found, pivrow = _pivot_row(T, basis, pivcol, phase, tol, bland)
            status, complete = lax.cond(pivrow_found==False,
                                        lambda _: (3, True),
                                        lambda _: (status, complete),
                                        None)
            return pivrow_found, pivrow, status, complete
        
        pivcol, pivrow, status, complete = lax.cond(pivcol_found==False,
                                                lambda _: (0, 0, 0, True),
                                                lambda _: (pivcol, pivrow, status, complete),
                                                None)
        pivrow_found, pivrow, status, complete = lax.cond(pivcol_found==True,
                                                          lambda _: cal_pivcol_found_True(T, basis, pivcol, phase, tol, bland, status, complete),
                                                          lambda _: (pivrow_found, pivrow, status, complete),
                                                          None)
        def cal_ifnot_complete(T, basis, nit, status, complete, maxiter):
            status, complete = lax.cond(nit >= maxiter, lambda _: (1, True), lambda _: (status, complete), None)
            T, basis = lax.cond(nit < maxiter, lambda _: _apply_pivot(T, basis, pivrow, pivcol, tol), lambda _: (T, basis), None)
            nit = lax.cond(nit < maxiter, lambda _: nit+1, lambda _: nit, None)
            return T, basis, nit, status, complete
        
        T, basis, nit, status, complete = lax.cond(complete==False,
                                                  lambda _: cal_ifnot_complete(T, basis, nit, status, complete, maxiter),
                                                  lambda _: (T, basis, nit, status, complete),
                                                  None)
        return T, basis, pivcol, pivrow, phase, tol, bland, status, complete, nit, maxiter
    
    T, basis, pivcol, pivrow, phase, tol, bland, status, complete, nit, maxiter = lax.while_loop(cond_ifnot_complete,
                                                                                                 body_ifnot_complete,
                                                                                                 (T, basis, 0, 0, phase, tol, bland, status, complete, nit, maxiter))
    return T, basis, nit, status
    # while not complete:
    #     pivcol_found, pivcol = _pivot_col(T, tol, bland)
    #     print(pivcol_found)
    #     pivrow_found = False
    #     def cal_pivcol_found_True(T, basis, pivcol, phase, tol, bland, status, complete):
    #         pivrow_found, pivrow = _pivot_row(T, basis, pivcol, phase, tol, bland)
    #         status, complete = lax.cond(pivrow_found==False,
    #                                     lambda _: (3, True),
    #                                     lambda _: (status, complete),
    #                                     None)
    #         return pivrow_found, pivrow, status, complete
        
    #     pivcol, pivrow, status, complete = lax.cond(pivcol_found==False,
    #                                             lambda _: (0, 0, 0, True),
    #                                             lambda _: (pivcol, pivrow, status, complete),
    #                                             None)
    #     pivrow_found, pivrow, status, complete = lax.cond(pivcol_found==True,
    #                                                       lambda _: cal_pivcol_found_True(T, basis, pivcol, phase, tol, bland, status, complete),
    #                                                       lambda _: (pivrow_found, pivrow, status, complete),
    #                                                       None)
    #     def cal_ifnot_complete(T, basis, nit, status, complete, maxiter):
    #         status, complete = lax.cond(nit >= maxiter, lambda _: (1, True), lambda _: (status, complete), None)
    #         T, basis = lax.cond(nit < maxiter, lambda _: _apply_pivot(T, basis, pivrow, pivcol, tol), lambda _: (T, basis), None)
    #         nit = lax.cond(nit < maxiter, lambda _: nit+1, lambda _: nit, None)
    #         return T, basis, nit, status, complete
    #     print(T)
    #     print("---------------------------------------------------")
    #     T, basis, nit, status, complete = lax.cond(complete==False,
    #                                               lambda _: cal_ifnot_complete(T, basis, nit, status, complete, maxiter),
    #                                               lambda _: (T, basis, nit, status, complete),
    #                                               None)
    
    # return T, basis, nit, status


def _linprog_simplex(c, A, b, c0=0, 
                     maxiter=1000, tol=1, disp=False, bland=False,
                     **unknown_options):
    # 删除参数callback, postsolve_args
    status = 0
    messages = {0: "Optimization terminated successfully.",
                1: "Iteration limit reached.",
                2: "Optimization failed. Unable to find a feasible"
                   " starting point.",
                3: "Optimization failed. The problem appears to be unbounded.",
                4: "Optimization failed. Singular matrix encountered."}

    n, m = A.shape
    
    # All constraints must have b >= 0.
    is_negative_constraint = jnp.less(b, 0)
    A = jnp.where(is_negative_constraint[:, None], A * -1, A)
    b = jnp.where(is_negative_constraint, b * -1, b)
    
    av = jnp.arange(n) + m
    # print(av)
    basis = av.copy()
    
    row_constraints = jnp.hstack((A, jnp.eye(n), b[:, jnp.newaxis]))
    row_objective = jnp.hstack((c, jnp.zeros(n), c0))
    row_pseudo_objective = -row_constraints.sum(axis=0)
    row_pseudo_objective = row_pseudo_objective.at[av].set(0)
    T = jnp.vstack((row_constraints, row_objective, row_pseudo_objective))
    # print(T)
    # print(n)
    # print(basis)
    # phase 1
    T, basis, nit1, status = _solve_simplex(T, n, basis, maxiter=maxiter, tol=tol, phase=1, bland=bland)
    # print(T)
    nit2 = nit1

    def if_abs_true(status):
        return status
    def if_abs_false(status):
        status = 2
        return status
    status = lax.cond(jnp.abs(T[-1, -1]) < tol,if_abs_true,if_abs_false,(status))
    # messages[2] = (
    #     "Phase 1 of the simplex method failed to find a feasible "
    #     "solution. The pseudo-objective function evaluates to {0:.1e} "
    #     "which exceeds the required tolerance of {1} for a solution to be "
    #     "considered 'close enough' to zero to be a basic solution. "
    #     "Consider increasing the tolerance to be greater than {0:.1e}. "
    #     "If this tolerance is unacceptably  large the problem may be "
    #     "infeasible.".format(abs(T[-1, -1]), tol)
    # )
    # def modify_tensor(T, av, tol):
    #     if (abs(T[-1, -1]) < tol):
    #         T = T[:-1, :]
    #         T = jnp.delete(T, av, 1)
    #     return T
    
    # jit_modify_tensor = jit(modify_tensor, static_argnums=(0,))
    # print(jnp.any(T>0))
    # av = tuple(av.tolist())
    # print(av)
    # T = jit_modify_tensor(T, av, tol)
    
    # av = av.item()
    # print(av) 
    # print("av",av.shape)

    original_shape = T.shape
    # print(original_shape)
    T_new = T[:-1, :]
    jit_delete = jit(jnp.delete, static_argnames=['assume_unique_indices'])
    T = jnp.delete(T_new, av, 1, assume_unique_indices=True)
    # ndicator = jnp.ones(T.shape[1], dtype=int)
    # print("avdsa",av)
    # def true_fn(T):
    #     T_new = T[:-1, :]
    #     jit_delete = jit(jnp.delete, static_argnames=['assume_unique_indices'])
    #     T_new = jnp.delete(T_new, av, 1, assume_unique_indices=True)
    #     # 保存有效部分的形状信息
    #     # T_new_shape = jnp.array(T_new.shape)
    #     T_new_shape = jnp.array([original_shape[0]-1, original_shape[1]-len(av)])
    #     padding = [(0, original_shape[0] - T_new.shape[0]), (0, original_shape[1] - T_new.shape[1])]
    #     T_padded = jnp.pad(T_new, padding, mode='constant')
    #     return T_padded, jnp.array(original_shape), T_new_shape
    #     # indicator = indicator.at[av].set(0)

    # def false_fn(T):
    #     return T, jnp.array(original_shape), jnp.array(original_shape)

    # T, T_shape, T_new_shape = lax.cond(abs(T[-1, -1]) < tol, true_fn, false_fn, T)
    
    # return T_modified, T_shape, T_new_shape
    # print(T_new_shape)
    def recover_tensor(T_recovered, T_shape, T_new_shape):
        # 根据保存的形状信息恢复原形状
        # rows_to_keep = abs(T_new_shape[0] - T_shape[0])
        # cols_to_keep = abs(T_new_shape[1] - T_shape[1])

        # 使用lax.dynamic_slice来保留左上角部分
        # print(T_recovered)
        T_recovered = lax.dynamic_slice(T_recovered, (0, 0), (T_new_shape[0], T_new_shape[1]))
        # print("T_recovered",T_recovered.shape)
        return T_recovered
  
    # T, T_shape, T_new_shape = modify_tensor(T, av, tol)
    # print(T_new_shape)
    # T = recover_tensor(T, T_shape, T_new_shape)
    # # phase 2
    T, basis, nit2, status = lax.cond(status == 0,
                                    lambda _: _solve_simplex(T, n, basis, maxiter, tol, 2, bland, nit1),
                                    lambda _: (T, basis, nit2, status),
                                    None)
    
    solution = jnp.zeros(n+m)
    solution = solution.at[basis[:n]].set(T[:n, -1])
    # status = status.astype(int)
    x = solution[:m]
    return x, status, nit2



# if __name__ == "__main__":
    # T = jnp.array([ 
    #     [ 1.,  1.,  0.,  1.,  0.,  0.,  4.],
    #     [ 2.,  1., -1.,  0.,  1.,  0.,  3.],
    #     [-1.,  2.,  1.,  0.,  0.,  1.,  2.],
    #     [-2., -1.,  2.,  0.,  0.,  0.,  0.],
    #     [-2., -4., -0.,  0.,  0.,  0., -9.],
    # ])
    # n = 3
    # basis = jnp.array([3,4,5])  # 假设初始的基变量索引
    
    # # pivcol_found, pivcol = _pivot_col(T)
    # # print(pivcol_found)
    # # print(pivcol)
    # pivcol = 1
    # phase = 1
    # pivrow = 2
    
    # # pivrow_found, pivrow = _pivot_row(T, basis, pivcol, phase)    
    # # print(pivrow_found)
    # # print(pivrow)
    
    # # T, basis = _apply_pivot(T, basis, pivrow, pivcol)
    # # print(T)
    # # print(basis)
    
    
    
    # T, basis, nit1, status = _solve_simplex(T, n, basis,phase=1)
    # print(T)
    # print(nit1)

#     # 处理结果
#     print("Final T matrix:")
#     print(T_final)
#     print("Final basis:", basis_final)
#     print("Number of iterations:", nit_final)
#     print("Status:", status_final)
    
    # T, basis = _apply_pivot(T, basis, 1, 0)
    # print(T)
    # print(basis)

    # A_eq = jnp.eye(3)
    # b_eq = jnp.ones(3)
    # c = jnp.array([1, 2, 3])
    # # n,m = A_eq.shape
    # result = _linprog_simplex(c, A_eq, b_eq)
    
    # print(result)
    # print(result[0]@c)

    # from scipy.optimize import linprog
    # result = linprog(c, A_eq=A_eq, b_eq=b_eq, method='simplex')
    # print(result)
