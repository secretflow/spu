# Copyright 2021 Ant Group Co., Ltd.
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


import unittest

import jax.numpy as jnp
import jax
import numpy as np

import spu.utils.simulation as ppsim
import spu.spu_pb2 as spu_pb2
import spu.intrinsic as si

scale = 1.0
power = 1.9


def upcast(x):
    return x.astype(np.float32)


def downcast(x):
    return x.astype(np.float16)


def starting_mu(labels: np.ndarray) -> np.ndarray:
    return (labels + jnp.mean(labels)) / 2.0


def baseline_irls_calculate_partials(
    x: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray = None,
    weight: np.ndarray = None,
    model: np.ndarray = None,
    start_mu: np.ndarray = None,
):
    y = y.reshape((-1, 1))

    if offset is not None:
        offset = offset.reshape((-1, 1))

    if weight is not None:
        weight = weight.reshape((-1, 1))

    if model is not None:
        model = model.reshape((-1, 1))
        eta = jnp.matmul(x, model)
        if offset is not None:
            mu = jnp.exp(eta + offset)
        else:
            mu = jnp.exp(eta)
    else:
        # for correctness, start_mu should be provided
        mu = start_mu
        eta = jnp.log(mu)
        if offset is not None:
            eta = eta - offset

    v = jnp.power(mu, power)
    g_gradient = 1 / mu

    # v = upcast(v)
    # g_gradient = upcast(g_gradient)

    if weight is not None:
        W_diag = weight / scale / (v * g_gradient) / g_gradient
    else:
        W_diag = 1 / scale / (v * g_gradient) / g_gradient
    Z = eta + (y - mu) * g_gradient

    # W_diag = downcast(W_diag)
    # Z = downcast(Z)
    # x = downcast(x)

    XTW = jnp.transpose(x * W_diag.reshape(-1, 1))

    J = jnp.matmul(XTW, x)
    XTWZ = jnp.matmul(XTW, Z)
    return J, XTWZ


def _irls_calculate_partials(
    x: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray = None,
    weight: np.ndarray = None,
    model: np.ndarray = None,
    start_mu: np.ndarray = None,
):
    y = y.reshape((-1, 1))

    if offset is not None:
        offset = offset.reshape((-1, 1))

    if weight is not None:
        weight = weight.reshape((-1, 1))

    if model is not None:
        model = model.reshape((-1, 1))
        eta = jnp.matmul(x, model)
        if offset is not None:
            mu = jnp.exp(eta + offset)
        else:
            mu = jnp.exp(eta)
    else:
        # for correctness, start_mu should be provided
        mu = start_mu
        eta = jnp.log(mu)
        if offset is not None:
            eta = eta - offset

    v = jnp.power(mu, power)
    g_gradient = 1 / mu

    # v = upcast(v)
    # g_gradient = upcast(g_gradient)

    if weight is not None:
        W_diag = weight / scale / (v * g_gradient) / g_gradient
    else:
        W_diag = 1 / scale / (v * g_gradient) / g_gradient
    Z = eta + (y - mu) * g_gradient

    W_diag = downcast(W_diag)
    Z = downcast(Z)
    x = downcast(x)

    XTW = jnp.transpose(x * W_diag.reshape(-1, 1))

    J = jnp.matmul(XTW, x)
    XTWZ = jnp.matmul(XTW, Z)
    return J, XTWZ


def new_irls_calculate_partials(
    x: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray = None,
    weight: np.ndarray = None,
    model: np.ndarray = None,
    start_mu: np.ndarray = None,
):
    y = y.reshape((-1, 1))

    if offset is not None:
        offset = offset.reshape((-1, 1))

    if weight is not None:
        weight = weight.reshape((-1, 1))

    if model is not None:
        model = model.reshape((-1, 1))
        eta = jnp.matmul(x, model)
        if offset is not None:
            mu = jnp.exp(eta + offset)
        else:
            mu = jnp.exp(eta)
    else:
        # for correctness, start_mu should be provided
        mu = start_mu
        eta = jnp.log(mu)
        if offset is not None:
            eta = eta - offset

    # v = jnp.power(mu, power)
    g_gradient = 1 / mu

    new_mu = upcast(mu)
    v_g_gradient_g_gradient = jnp.power(new_mu, power - 2)

    # v = upcast(v)
    # g_gradient = upcast(g_gradient)

    if weight is not None:
        W_diag = weight / scale / v_g_gradient_g_gradient
    else:
        W_diag = 1 / scale / v_g_gradient_g_gradient

    Z = eta + (y - mu) * g_gradient

    W_diag = downcast(W_diag)
    Z = downcast(Z)
    x = downcast(x)

    XTW = jnp.transpose(x * W_diag.reshape(-1, 1))

    J = jnp.matmul(XTW, x)
    XTWZ = jnp.matmul(XTW, Z)
    return J, XTWZ


if __name__ == "__main__":
    """
    You can modify the code below for debug purpose only.
    Please DONT commit it unless it will cause build break.
    """
    config = spu_pb2.RuntimeConfig(protocol=spu_pb2.ProtocolKind.ABY3)
    config.enable_hal_profile = True
    config.enable_pphlo_profile = True

    sim = ppsim.Simulator(3, config)
    copts = spu_pb2.CompilerOptions()
    # Tweak compiler options
    copts.disable_div_sqrt_rewrite = True
    copts.enable_pretty_print = True
    copts.pretty_print_dump_dir = "/home/haoqi.whq/SF-playground/ppu/ppdump"
    copts.xla_pp_kind = 2

    x = np.random.randn(4, 5).astype(np.float16)
    y = np.random.randn(4, 1).astype(np.float16)
    y = np.abs(y)
    # fn = lambda x, y: jax.nn.relu(x) * jax.nn.relu(jnp.exp(x + y))
    fn = lambda x, y: jnp.matmul(x, y)

    spu_fn = ppsim.sim_jax(sim, _irls_calculate_partials, copts=copts)
    z = spu_fn(
        x.astype(np.float32),
        y.astype(np.float32),
        start_mu=starting_mu(y).astype(np.float32),
    )

    spu_fn = ppsim.sim_jax(sim, baseline_irls_calculate_partials, copts=copts)
    z = spu_fn(
        x.astype(np.float32),
        y.astype(np.float32),
        start_mu=starting_mu(y).astype(np.float32),
    )

    new_spu_fn = ppsim.sim_jax(sim, new_irls_calculate_partials, copts=copts)
    new_z = new_spu_fn(x, y, start_mu=starting_mu(y).astype(np.float16))

    print(spu_fn.pphlo)

    print(f"spu out = {z}")
    print(f"new spu out = {new_z}")
    print(f"cpu out = {_irls_calculate_partials(x, y, start_mu=starting_mu(y))}")
