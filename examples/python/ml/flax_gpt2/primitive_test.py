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

# Start nodes.
# > bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/ml/flax_gpt2/3pc.json up
#
# Run this example script.
# > bazel run -c opt //examples/python/ml/flax_gpt2:flax_gpt2

import argparse
import json
import jax
import jax.numpy as jnp
import jax.nn as jnn

import numpy as np
import spu.utils.distributed as ppd

import flax.linen as fnn
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, Tuple, Union
import spu.spu_pb2 as spu_pb2

Array = Any
copts = spu_pb2.CompilerOptions()
copts.enable_pretty_print = False
copts.xla_pp_kind = 2
# enable x / broadcast(y) -> x * broadcast(1/y)
copts.enable_optimize_denominator_with_broadcast = True

VECTOR_LEN = 1024 * 1024

parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="examples/python/ml/flax_gpt2/3pc.json")
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])

def hack_softmax(x: Array,
            axis: Optional[Union[int, Tuple[int, ...]]] = -1,
            where: Optional[Array] = None,
            initial: Optional[Array] = None) -> Array:

    x_max = jnp.max(x, axis, where=where, initial=initial, keepdims=True)
    x = x - x_max

    # exp on large negative is clipped to zero
    b = x > -14
    nexp = jnp.exp(x) * b

    divisor = jnp.sum(nexp, axis, where=where, keepdims=True)

    return nexp / divisor

@contextmanager
def hack_softmax_context(msg: str, enabled: bool = False):
    if not enabled:
        yield
        return
    # hijack some target functions
    raw_softmax = jnn.softmax
    jnn.softmax = hack_softmax
    yield
    # recover back
    jnn.softmax = raw_softmax

def hack_gelu(x: Array,
            axis: Optional[Union[int, Tuple[int, ...]]] = -1,
            where: Optional[Array] = None,
            initial: Optional[Array] = None,
            approximate: bool = True) -> Array:

    b0 = x < -4.0
    b1 = x < -1.95
    b2 = x > 3.0
    b3 = b1 ^ b2 ^ True # x in [-1.95, 3.0]
    b4 = b0 ^ b1 # x in [-4, -1.95] 

    # seg1 = a[3] * x^3 + a[2] * x^2 + a[1] * x + a[0]
    # seg2 = b[6] * x^6 + b[4] * x^4 + b[2] * x^2 + b[1] * x + b[0]
    a_coeffs = jnp.array([-0.5054031199708174, -0.42226581151983866, -0.11807612951181953, -0.011034134030615728])
    b_coeffs = jnp.array([0.008526321541038084,  0.5, 0.3603292692789629, 0.0, -0.037688200365904236, 0.0, 0.0018067462606141187])
    x2 = jnp.square(x)
    x3 = jnp.multiply(x, x2)
    x4 = jnp.square(x2)
    x6 = jnp.square(x3)

    seg1 = a_coeffs[3] * x3 + a_coeffs[2] * x2 + a_coeffs[1] * x + a_coeffs[0]
    seg2 = b_coeffs[6] * x6 + b_coeffs[4] * x4 + b_coeffs[2] * x2 + b_coeffs[1] * x + b_coeffs[0]

    ret = b2 * x + b4 * seg1 + b3 * seg2

    return ret

@contextmanager
def hack_gelu_context(msg: str, enabled: bool = False):
    if not enabled:
        yield
        return
    # hijack some target functions
    raw_gelu = jnn.gelu
    jnn.gelu = hack_gelu
    yield
    # recover back
    jnn.gelu = raw_gelu
    
def show_l2(secret: float, plain: float):
    print("Stastic.")
    print(f"l2: {jnp.linalg.norm(secret - plain)}")
    print(f"medium: {jnp.median(secret - plain)}")
    print(f"max abs: {jnp.max(jnp.abs(secret - plain))}")
    print(f"min abs: {jnp.min(jnp.abs(secret - plain))}")

'''
Primitive test.
'''
def msb(x):
    return x > 0
def eqz(x):
    return x == 0
def a2b(x):
    return x ^ 1
def b2a(x):
    return (x ^ 1) + 1
def exp(x):
    return jnp.exp(x)
def relu(x):
    return jnn.relu(x)
def gelu(x):
    with hack_gelu_context("hijack jax gelu", enabled=True):
        return jnn.gelu(x)
def softmax(x):
    with hack_softmax_context("hijack jax softmax", enabled=True):
        return jnn.softmax(x)
def sigmoid(x):
    return jnn.sigmoid(x)

def sample_input():
    np.random.seed()
    x = np.random.randint(-10, 10, VECTOR_LEN)
    return x

def exp_msb():
    x = sample_input()
    x_spu = ppd.device("P1")(lambda x: x)(x)
    msb_spu = ppd.device("SPU")(msb)(x_spu)
    msb_spu = ppd.get(msb_spu)
    show_l2(msb_spu, (x > 0).astype(np.float64))

# def exp_eqz():
#     x = sample_input()
#     x_spu = ppd.device("P1")(lambda x: x)(x)
#     eqz_spu = ppd.device("SPU")(eqz)(x_spu)

def exp_ppa():
    x = sample_input()
    x_spu = ppd.device("P1")(lambda x: x)(x)
    b2a_spu = ppd.device("SPU")(b2a)(x_spu)
    b2a_spu = ppd.get(b2a_spu)
    show_l2(b2a_spu, (x ^ 1).astype(np.float64) + 1)

def exp_a2b():
    x = sample_input()
    x_spu = ppd.device("P1")(lambda x: x)(x)
    a2b_spu = ppd.device("SPU")(a2b)(x_spu)
    a2b_spu = ppd.get(a2b_spu)
    show_l2(a2b_spu, (x ^ 1).astype(np.float64))
    
def exp_exp():
    x = sample_input()
    x_spu = ppd.device("P1")(lambda x: x)(x)
    exp_spu = ppd.device("SPU")(exp)(x_spu)
    exp_spu = ppd.get(exp_spu)
    show_l2(exp_spu, jnp.exp(x))
    
def exp_relu():
    x = sample_input()
    x_spu = ppd.device("P1")(lambda x: x)(x)
    b2a_spu = ppd.device("SPU")(relu)(x_spu)
    show_l2(x_spu, x)
    
def exp_gelu():
    x = sample_input()
    x_spu = ppd.device("P1")(lambda x: x)(x)
    b2a_spu = ppd.device("SPU")(gelu)(x_spu)
    show_l2(x_spu, x)
    
def exp_softmax():
    x = sample_input()
    x_spu = ppd.device("P1")(lambda x: x)(x)
    b2a_spu = ppd.device("SPU")(softmax)(x_spu)
    show_l2(x_spu, x)
    
def exp_sigmoid():
    x = sample_input()
    x_spu = ppd.device("P1")(lambda x: x)(x)
    b2a_spu = ppd.device("SPU")(sigmoid)(x_spu)
    b2a_spu = ppd.get(b2a_spu)
    show_l2(x_spu, x)

if __name__ == '__main__':
    # exp_eqz()
    
    exp_msb()
    exp_ppa()
    exp_a2b()

    # exp_exp()
    # exp_relu()
    # exp_gelu()
    # exp_softmax()
    # exp_sigmoid()