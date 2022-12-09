# Copyright 2022 Ant Group Co., Ltd.
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


"""
Custom jax optimizer implementations. 
Refer to https://github.com/google/jax/blob/main/jax/example_libraries/optimizers.py
"""

import jax.numpy as jnp
from jax.example_libraries import optimizers
from jax.example_libraries.optimizers import make_schedule


@optimizers.optimizer
def amsgrad(step_size, b1=0.9, b2=0.999, eps=1e-8):
    """Construct optimizer triple for amsgrad.

    Reference: https://paperswithcode.com/method/amsgrad

    Args:
      step_size: positive scalar, or a callable representing a step size schedule
        that maps the iteration index to a positive scalar.
      b1: optional, a positive scalar value for beta_1, the exponential decay rate
        for the first moment estimates (default 0.9).
      b2: optional, a positive scalar value for beta_2, the exponential decay rate
        for the second moment estimates (default 0.999).
      eps: optional, a positive scalar value for epsilon, a small constant for
        numerical stability (default 1e-8).

    Returns:
      An (init_fun, update_fun, get_params) triple.
    """

    step_size = make_schedule(step_size)

    def init(x0):
        m0 = jnp.zeros_like(x0)
        v0 = jnp.zeros_like(x0)
        vhat0 = jnp.zeros_like(x0)
        return x0, m0, v0, vhat0

    def update(i, g, state):
        x, m, v, vhat = state
        m = (1 - b1) * g + b1 * m
        v = (1 - b2) * jnp.square(g) + b2 * v
        vhat = jnp.maximum(vhat, v)
        x = x - step_size(i) * m / (jnp.sqrt(vhat) + eps)
        return x, m, v, vhat

    def get_params(state):
        x, _, _, _ = state
        return x

    return init, update, get_params
