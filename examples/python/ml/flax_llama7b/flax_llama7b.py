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
# > bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/ml/flax_llama/3pc.json" up
# Run this example script.
# > bazel run -c opt //examples/python/ml/flax_llama7b -- --config `pwd`/examples/python/ml/flax_llama7b/3pc.json

import argparse
import json
import jax
import jax.numpy as jnp
import jax.nn as jnn
import flax.linen as nn
from flax.linen.linear import Array
from typing import Any, Optional, Tuple, Union
from transformers import LlamaTokenizer
from EasyLM.models.llama.llama_model import LLaMAConfig, FlaxLLaMAForCausalLM
import spu.utils.distributed as ppd
from contextlib import contextmanager
import spu.intrinsic as intrinsic
import spu.spu_pb2 as spu_pb2

parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="examples/python/ml/flax_llama/3pc.json")
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])

copts = spu_pb2.CompilerOptions()
copts.enable_pretty_print = False
copts.xla_pp_kind = 2
# enable x / broadcast(y) -> x * broadcast(1/y)
copts.enable_optimize_denominator_with_broadcast = True

model_path = 'path-to-flax-llama7b'

tokenizer = LlamaTokenizer.from_pretrained(model_path)
tokenizer.pad_token_id = tokenizer.eos_token_id
config = LLaMAConfig()
pretrained_model = FlaxLLaMAForCausalLM.from_pretrained(model_path, config=config)


def hack_softmax(
    x: Array,
    axis: Optional[Union[int, Tuple[int, ...]]] = -1,
    where: Optional[Array] = None,
    initial: Optional[Array] = None,
) -> Array:
    x_max = jnp.max(x, axis, where=where, initial=initial, keepdims=True)
    x = x - x_max
    # exp on large negative is clipped to zero
    b = x > -14
    nexp = jnp.exp(x) * b
    divisor = jnp.sum(nexp, axis, where=where, keepdims=True)
    return nexp / divisor


@contextmanager
def hack_softmax_context(msg: str, enabled: bool = True):
    if not enabled:
        yield
        return
    # hijack some target functions
    raw_softmax = jnn.softmax
    jnn.softmax = hack_softmax
    yield
    # recover back
    jnn.softmax = raw_softmax


def hack_silu(x: Array) -> Array:
    b0 = x < -8.0
    b1 = x < -4.0
    b2 = x > 4.0
    b3 = b1 ^ b2 ^ True  # x in [-4.0, 4.0)
    b4 = b0 ^ b1  # x in [-8.0, -4.0)
    # seg1 =  a[2] * x^2 + a[1] * x + a[0]
    # seg2 = b[6] * x^6 + b[4] * x^4 + b[2] * x^2 + b[0]
    a_coeffs = jnp.array(
        [-0.3067541139982155, -0.0819767021525476, -0.0055465625580307]
    )
    b_coeffs = jnp.array(
        [
            0.0085064025895951,
            0.5,
            0.2281430841728270,
            -0.011113046708173,
            0.0002743776353465,
        ]
    )
    x2 = jnp.square(x)
    x4 = jnp.square(x2)
    x6 = x2 * x4
    seg1 = a_coeffs[2] * x2 + a_coeffs[1] * x + a_coeffs[0]
    seg2 = (
        b_coeffs[4] * x6
        + b_coeffs[3] * x4
        + b_coeffs[2] * x2
        + b_coeffs[1] * x
        + b_coeffs[0]
    )
    ret = b2 * x + b4 * seg1 + b3 * seg2
    return ret


@contextmanager
def hack_silu_context(msg: str, enabled: bool = True):
    if not enabled:
        yield
        return
    # hijack some target functions
    raw_silu = nn.silu
    nn.silu = hack_silu
    yield
    # recover back
    nn.silu = raw_silu


# greedy search
# ref: https://huggingface.co/blog/how-to-generate
def text_generation(input_ids, params, token_num=1):
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
        'Q: What is the largest animal?\nA:', return_tensors='jax'
    )
    outputs_ids = text_generation(inputs_ids, pretrained_model.params)
    return outputs_ids


def run_on_spu():
    # encode context the generation is conditioned on
    input_ids = tokenizer.encode(
        'Q: What is the largest animal?\nA:', return_tensors='jax'
    )
    with hack_softmax_context("hack exp of softmax", enabled=True), hack_silu_context(
        "hack silu", enabled=True
    ):
        params = ppd.device("P2")(lambda x: x)(pretrained_model.params)
        input_ids = ppd.device("P1")(lambda x: x)(input_ids)
        outputs_ids = ppd.device("SPU")(text_generation, copts=copts)(input_ids, params)
        outputs_ids = ppd.get(outputs_ids)
    return outputs_ids


if __name__ == '__main__':
    print('\n------\nRun on CPU')
    outputs_ids = run_on_cpu()
    print(tokenizer.decode(outputs_ids[0], skip_special_tokens=True))
    print('\n------\nRun on SPU')
    outputs_ids = run_on_spu()
    print(tokenizer.decode(outputs_ids[0], skip_special_tokens=True))
