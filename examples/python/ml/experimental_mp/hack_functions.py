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

from flax.linen.linear import Array
from typing import Any, Callable, Dict, Optional, Tuple, Union
from contextlib import contextmanager
import jax.nn as jnn
import jax.numpy as jnp
from functools import partial


def gelu_poly(
    x: Array,
    axis: Optional[Union[int, Tuple[int, ...]]] = -1,
    where: Optional[Array] = None,
    initial: Optional[Array] = None,
) -> Array:
    b0 = x < -4.0
    b1 = x < -1.95
    b2 = x > 3.0
    b3 = b1 ^ b2 ^ True  # x in [-1.95, 3.0]
    b4 = b0 ^ b1  # x in [-4, -1.95]

    # seg1 = a[3] * x^3 + a[2] * x^2 + a[1] * x + a[0]
    # seg2 = b[6] * x^6 + b[4] * x^4 + b[2] * x^2 + b[1] * x + b[0]
    a_coeffs = jnp.array(
        [
            -0.5054031199708174,
            -0.42226581151983866,
            -0.11807612951181953,
            -0.011034134030615728,
        ]
    )
    b_coeffs = jnp.array(
        [
            0.008526321541038084,
            0.5,
            0.3603292692789629,
            0.0,
            -0.037688200365904236,
            0.0,
            0.0018067462606141187,
        ]
    )
    x2 = jnp.square(x)
    x3 = jnp.multiply(x, x2)
    x4 = jnp.square(x2)
    x6 = jnp.square(x3)

    seg1 = a_coeffs[3] * x3 + a_coeffs[2] * x2 + a_coeffs[1] * x + a_coeffs[0]
    seg2 = (
        b_coeffs[6] * x6
        + b_coeffs[4] * x4
        + b_coeffs[2] * x2
        + b_coeffs[1] * x
        + b_coeffs[0]
    )

    ret = b2 * x + b4 * seg1 + b3 * seg2

    return ret


def gelu_quad(
    x: Array,
    axis: Optional[Union[int, Tuple[int, ...]]] = -1,
    where: Optional[Array] = None,
    initial: Optional[Array] = None,
) -> Array:
    ret = 0.125 * jnp.square(x) + 0.25 * x + 0.5
    return ret


def puma_softmax(
    x: Array,
    axis: Optional[Union[int, Tuple[int, ...]]] = -1,
    where: Optional[Array] = None,
    initial: Optional[Array] = None,
) -> Array:
    x_max = jnp.max(x, axis, where=where, initial=initial, keepdims=True)
    x = x - x_max

    # exp on large negative is clipped to zero
    b = x > -14
    # for the sake of precision
    x = x.astype(jnp.float32)
    nexp = jnp.exp(x) * b

    divisor = jnp.sum(nexp, axis, where=where, keepdims=True)

    return nexp / divisor


def softmax_2relu(scores, eps=1e-12):
    relu = jnn.relu(scores)
    reduce_dim = scores.shape[-1]
    out = (relu + eps / reduce_dim) / (jnp.sum(relu, axis=-1, keepdims=True) + eps)
    return out


def softmax_2quad(scores, attention_mask_zero_one, axis):
    scores = (scores + 5) ** 2
    scores *= attention_mask_zero_one
    scores = scores / jnp.sum(scores, axis=axis, keepdims=True)
    return scores


ACT2FN = {
    "raw": partial(jnn.gelu, approximate=True),
    "quad": gelu_quad,
    "puma": gelu_poly,
}
ACT2SFN = {
    "raw": jnn.softmax,
    "puma": puma_softmax,
    "2relu": softmax_2relu,
    "2quad": softmax_2quad,
}


@contextmanager
def hack_gelu_context(msg: str, enabled: bool = False):
    if not enabled:
        yield
        return
    # hijack some target functions
    print(f"Using gelu: {msg}")
    raw_gelu = jnn.gelu
    jnn.gelu = ACT2FN[msg]
    yield
    # recover back
    jnn.gelu = raw_gelu


@contextmanager
def hack_softmax_context(msg: str, enabled: bool = False):
    if not enabled:
        yield
        return
    # hijack some target functions
    print(f"Using softmax: {msg}")
    raw_softmax = jnn.softmax
    jnn.softmax = ACT2SFN[msg]
    yield
    # recover back
    jnn.softmax = raw_softmax
