# Copyright 2020 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Limited-Memory Broyden-Fletcher-Goldfarb-Shanno minimization algorithm."""
from typing import Any, Callable, NamedTuple, Optional, Union
from functools import partial

import jax
from jax import jit
import jax.numpy as jnp
from jax import lax
from .line_search import line_search  

_dot = partial(jnp.dot, precision=lax.Precision.HIGHEST)

Array = Any

class LBFGSResults(NamedTuple):
  k: Union[int, Array]
  nfev: Union[int, Array]
  ngev: Union[int, Array]
  x_k: Array
  f_k: Array
  g_k: Array
  s_history: Array
  y_history: Array
  rho_history: Array
  gamma: Union[float, Array]
  ls_status: Union[int, Array]


def _minimize_lbfgs(
    fun: Callable,
    x0: Array,
    maxiter: Optional[float] = None,
    norm=jnp.inf,
    maxcor: int = 15,
    maxls: int = 20,
):
  if not isinstance(x0, jnp.ndarray):
    x0 = jnp.array(x0)
  d = len(x0)
  dtype = jnp.dtype(x0)

  if maxiter is None:
    maxiter = d * 100


  def cond_fun(state: LBFGSResults):
    return state.k<maxiter
  def body_fun(state: LBFGSResults):
    # find search direction
    p_k = _two_loop_recursion(state)

    # line search
    ls_results = line_search(
      f=fun,
      xk=state.x_k,
      pk=p_k,
      old_fval=state.f_k,
      gfk=state.g_k,
      maxiter=maxls,
    )

    # evaluate at next iterate
    s_k = ls_results.a_k.astype(p_k.dtype) * p_k
    x_kp1 = state.x_k + s_k
    f_kp1 = ls_results.f_k
    g_kp1 = ls_results.g_k
    y_k = g_kp1 - state.g_k
    rho_k_inv = jnp.real(_dot(y_k, s_k))
    rho_k = jnp.reciprocal(rho_k_inv).astype(y_k.dtype)
    gamma = rho_k_inv / jnp.real(_dot(jnp.conj(y_k), y_k))

    state = state._replace(
      k=state.k + 1,
      nfev=state.nfev + ls_results.nfev,
      ngev=state.ngev + ls_results.ngev,
      x_k=x_kp1.astype(state.x_k.dtype),
      f_k=f_kp1.astype(state.f_k.dtype),
      g_k=g_kp1.astype(state.g_k.dtype),
      s_history=_update_history_vectors(history=state.s_history, new=s_k),
      y_history=_update_history_vectors(history=state.y_history, new=y_k),
      rho_history=_update_history_scalars(history=state.rho_history, new=rho_k),
      gamma=gamma.astype(state.g_k.dtype),
      ls_status=ls_results.status,
    )

    return state


  # initial evaluation
  f_0, g_0 = jax.value_and_grad(fun)(x0)
  # 通过 jit 函数执行 cond_fun
  # cond_fun_jit = jit(body_fun)
  state_initial = LBFGSResults(
    k=0,
    nfev=1,
    ngev=1,
    x_k=x0,
    f_k=f_0,
    g_k=g_0,
    s_history=jnp.zeros((maxcor, d), dtype=dtype),
    y_history=jnp.zeros((maxcor, d), dtype=dtype),
    rho_history=jnp.zeros((maxcor,), dtype=dtype),
    gamma=1.,
    ls_status=0,
  )

  return lax.while_loop(cond_fun, body_fun, state_initial)


def _two_loop_recursion(state: LBFGSResults):
  dtype = state.rho_history.dtype
  his_size = len(state.rho_history)
  curr_size = jnp.where(state.k < his_size, state.k, his_size)
  q = -jnp.conj(state.g_k)
  a_his = jnp.zeros_like(state.rho_history)

  def body_fun1(j, carry):
    i = his_size - 1 - j
    _q, _a_his = carry
    a_i = state.rho_history[i] * _dot(jnp.conj(state.s_history[i]), _q).real.astype(dtype)
    _a_his = _a_his.at[i].set(a_i)
    _q = _q - a_i * jnp.conj(state.y_history[i])
    return _q, _a_his

  q, a_his = lax.fori_loop(0, curr_size, body_fun1, (q, a_his))
  q = state.gamma * q

  def body_fun2(j, _q):
    i = his_size - curr_size + j
    b_i = state.rho_history[i] * _dot(state.y_history[i], _q).real.astype(dtype)
    _q = _q + (a_his[i] - b_i) * state.s_history[i]
    return _q

  q = lax.fori_loop(0, curr_size, body_fun2, q)
  return q


def _update_history_vectors(history, new):
  # TODO(Jakob-Unfried) use rolling buffer instead? See #6053
  return jnp.roll(history, -1, axis=0).at[-1, :].set(new)


def _update_history_scalars(history, new):
  # TODO(Jakob-Unfried) use rolling buffer instead? See #6053
  return jnp.roll(history, -1, axis=0).at[-1].set(new)
