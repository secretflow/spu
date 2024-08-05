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


"""
Custom loss function. With in the backward computation, extend the gradients
"""
from functools import partial
from typing import Any, Callable, List, NamedTuple, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import custom_vjp, vmap
from jax.nn import (
    elu,
    gelu,
    leaky_relu,
    log_softmax,
    normalize,
    relu,
    selu,
    sigmoid,
    softmax,
    softplus,
)
from jax.nn.initializers import glorot_normal, normal, ones, zeros


@custom_vjp
def custom_mse_loss(y, label):
    return jnp.sum((y - label) ** 2)


def custom_mse_loss_fwd(y, label):
    print("Custom loss forward", "=" * 20)
    # TODO: weight reduction
    loss = custom_mse_loss(y, label)
    print(
        "y type: {0}, label type: {1}, loss type: {2}".format(
            y.dtype, label.dtype, loss.dtype
        )
    )
    return loss, 2.0 * (y - label)


def custom_mse_loss_bwd(res, g):
    print("Custom loss backward", "=" * 20)
    loss_lo = res
    # TODO: loss gradient extension, this does not ensure the input x is positive.
    loss_hi = loss_lo
    # loss_hi = loss_lo
    print(
        "loss_lo type: {0}, loss_hi type: {1}, g type: {2}".format(
            loss_lo.dtype, loss_hi.dtype, g.dtype
        )
    )

    # TODO: gradient extension?
    return loss_hi, None


custom_mse_loss.defvjp(custom_mse_loss_fwd, custom_mse_loss_bwd)
"""
Custom relu computation
"""


@custom_vjp
def custom_relu(x):
    return relu(x)


def custom_relu_fwd(x):
    print("Custom relu forward", "=" * 20)
    # TODO: weight reduction
    pred = x > 0
    print("pred: {0}, x type: {1}".format(pred.dtype, x.dtype))
    return jax.lax.select(pred, x, jax.lax.full_like(x, 0)), pred


def custom_relu_bwd(res, g):
    print("Custom relu backward", "=" * 20)
    pred = res
    # TODO: activation & weight extension
    # x_hi = share_extension_prim(x_lo)
    # loss_hi = loss_lo
    print("pred: {0}, g type: {1}".format(pred.dtype, g.dtype))

    # TODO: gradient extension?
    print(g)
    return (jax.lax.select(pred, g, jax.lax.full_like(g, 0)),)


custom_relu.defvjp(custom_relu_fwd, custom_relu_bwd)
"""
Custom Dense layer
"""


@custom_vjp
def custom_Dot(inputs, W):
    # W_lo = W.astype(np.float32)
    # print("inputs type: {0}, W_lo type: {1}, W type: {2}".format(inputs.dtype, W_lo.dtype, W.dtype))
    return jnp.dot(inputs, W)


# https://jax.readthedocs.io/en/latest/jax.html#jax.linearize
def custom_Dot_fwd(inputs, W):
    print("Custom Dot forward", "=" * 20)
    # TODO: add share reduction protocol
    W_lo = W
    # W_lo = W.astype(FX32)
    outputs = custom_Dot(inputs, W_lo)
    print(
        "inputs type: {0}, W_lo type: {1}, W type: {2}, output type: {3}".format(
            inputs.dtype, W_lo.dtype, W.dtype, outputs.dtype
        )
    )
    # outputs, dot_jvp = jax.linearize(jnp.dot, inputs, W)
    return outputs, (inputs, W)


def custom_Dot_bwd(res, g):
    print("Custom Dot backward", "=" * 20)
    inputs_lo, W_hi = res
    # inputs_hi = share_extension_prim(inputs_lo)
    # inputs_hi = inputs_lo.astype(np.float32)
    inputs_hi = inputs_lo
    print(
        "inputs_lo type: {0}, inputs_hi type: {1}, g type: {2}, W_hi type: {3}".format(
            inputs_lo.dtype, inputs_hi.dtype, g.dtype, W_hi.dtype
        )
    )

    print("g: ", g.shape)
    print("inputs: ", inputs_hi.shape)
    print("W_hi: ", W_hi.shape)

    # https://github.com/google/jax/issues/2303
    # grad_fn = grad(lambda x, W: jnp.sum(jnp.dot(x, W)), argnums=(0, 1))
    # grad_inputs, grad_W = grad_fn(inputs_hi, W_hi)
    # return jnp.dot(g, W_hi.T), jnp.dot(inputs_hi.T, g)

    # https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#vjps-in-jax-code
    _, grad_fn = jax.vjp(jnp.dot, inputs_hi, W_hi)
    return grad_fn(g)


custom_Dot.defvjp(custom_Dot_fwd, custom_Dot_bwd)


def custom_Dense(out_dim, W_init=glorot_normal(), b_init=normal()):
    """Layer constructor function for a dense (fully-connected) layer."""

    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        k1, k2 = jax.random.split(rng)
        W, b = W_init(k1, (input_shape[-1], out_dim)), b_init(k2, (out_dim,))
        return output_shape, (W, b)

    def apply_fun(params, inputs, **kwargs):
        W, b = params
        # b_lo = b.astype(FXP32)
        b_lo = b
        return custom_Dot(inputs, W) + b_lo

    return init_fun, apply_fun
