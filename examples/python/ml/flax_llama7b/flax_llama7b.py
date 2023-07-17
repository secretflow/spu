# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Start nodes.
# > bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/ml/flax_llama7b/3pc.json up
#
# Run this example script.
# > bazel run -c opt //examples/python/ml/flax_llama7b:flax_llama

import argparse
import json
import jax
import jax.numpy as jnp
import jax.nn as jnn
import flax.linen as nn
from flax.linen.linear import Array
from typing import Any, Optional, Tuple, Union
import torch

from transformers import LlamaTokenizer, LlamaForCausalLM
from EasyLM.models.llama.llama_model import LLaMAConfig, FlaxLLaMAForCausalLM
from transformers import AutoTokenizer, FlaxGPT2LMHeadModel, GPT2Config

import spu.utils.distributed as ppd
from contextlib import contextmanager
import spu.intrinsic as intrinsic
import spu.spu_pb2 as spu_pb2

parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="examples/python/ml/flax_llama7b/3pc.json")
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])

copts = spu_pb2.CompilerOptions()
copts.enable_pretty_print = False
copts.xla_pp_kind = 2
# enable x / broadcast(y) -> x * broadcast(1/y)
copts.enable_optimize_denominator_with_broadcast = True

model_path = 'openlm-research/open_llama_7b'
tokenizer = LlamaTokenizer.from_pretrained(model_path)
pretrained_model = FlaxLLaMAForCausalLM.from_pretrained(model_path, from_pt=True)

def hack_softmax(x: Array,
            axis: Optional[Union[int, Tuple[int, ...]]] = -1,
            where: Optional[Array] = None,
            initial: Optional[Array] = None) -> Array:

    x_max = jnp.max(x, axis, where=where, initial=initial, keepdims=True)
    x = x - x_max

    # exp on large negative is clipped to zero
    b = x > -14
    nexp = jnp.exp(x)

    divisor = jnp.sum(nexp, axis, where=where, keepdims=True)

    return b * (nexp / divisor)

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

def hack_gelu(x: Array) -> Array:

    b0 = x < -4.0
    b1 = x < -1.95
    b2 = x > 3.0
    b3 = b1 ^ b2 ^ True # x in [-1.95, 3.0]
    b4 = b0 ^ b1 # x in [-4, -1.95)

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



# greedy search
# ref: https://huggingface.co/blog/how-to-generate
def text_generation(input_ids, params, token_num=8):
    config = LLaMAConfig()
    model = FlaxLLaMAForCausalLM(config=config)
    for _ in range(token_num):
        outputs = model(input_ids=input_ids, params=params)
        next_token_logits = outputs[0][0, -1, :]
        next_token = jnp.argmax(next_token_logits)
        input_ids = jnp.concatenate([input_ids, jnp.array([[next_token]])], axis=1)
    return input_ids


def run_on_cpu():
    # encode context the generation is conditioned on
    inputs_ids = tokenizer.encode(
        'Hello, my dog is cute and', return_tensors='jax'
        )

    outputs_ids = text_generation(inputs_ids, pretrained_model.params)
    return outputs_ids


def run_on_spu():
    # encode context the generation is conditioned on
    inputs_ids = tokenizer.encode(
        'Hello, my dog is cute and', return_tensors='jax'
        )

    # enabled=True, turn on hijacking function; enabled=False, turn off hijacking.
    with hack_softmax_context("hack exp of softmax", enabled=True), hack_gelu_context("hack gelu", enabled=True):
        input_ids = ppd.device("P1")(lambda x: x)(inputs_ids)
        params = ppd.device("P2")(lambda x: x)(pretrained_model.params)
        outputs_ids = ppd.device("SPU")(
            text_generation, copts=copts
            )(input_ids, params)
        outputs_ids = ppd.get(outputs_ids)

    return outputs_ids


if __name__ == '__main__':
    print('\n------\nRun on CPU')
    outputs_ids = run_on_cpu()
    print(tokenizer.decode(outputs_ids[0], skip_special_tokens=True))
    print('\n------\nRun on SPU')
    outputs_ids = run_on_spu()
    print(tokenizer.decode(outputs_ids[0], skip_special_tokens=True))
