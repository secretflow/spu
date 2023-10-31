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
# > bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/ml/flax_llama_split/3pc.json" up
# Run this example script.
# > bazel run -c opt //examples/python/ml/flax_llama7b -- --config `pwd`/examples/python/ml/flax_llama_split/3pc.json
import time
import argparse
import json
import jax
import jax.numpy as jnp
import jax.nn as jnn
import flax.linen as nn
from flax.linen.linear import Array
from typing import Any, Optional, Tuple, Union
from transformers import LlamaTokenizer
from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.models.llama.llama_model import FlaxLLaMAForCausalLM
from EasyLM.models.llama.llama_model_splited_transformer import (
    FlaxLLaMAForCausalLMClient,
    FlaxLLaMAForCausalLMServer,
    FlaxLLaMAModule,
    FlaxLLaMAForCausalLMMid,
    LLaMAConfig,
)


import spu.utils.distributed as ppd
from contextlib import contextmanager
import spu.spu_pb2 as spu_pb2

from flax.linen.linear import Array
from typing import Any, Optional, Tuple, Union

parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument(
    "-c", "--config", default="examples/python/ml/flax_llama_split/3pc.json"
)
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])

copts = spu_pb2.CompilerOptions()
copts.enable_pretty_print = False
copts.xla_pp_kind = 2
# enable x / broadcast(y) -> x * broadcast(1/y)
copts.enable_optimize_denominator_with_broadcast = True

# model_path = 'path-to-flax-llama7b'

model_path = "params::path-to-flax-llama7b-checkpoint"
tokenizer_path = "path-to-flax-llama7b"
tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
tokenizer.pad_token_id = tokenizer.eos_token_id
config = LLaMAConfig()
# pretrained_model = FlaxLLaMAForCausalLM.from_pretrained(model_path, config=config)
with jax.default_device(jax.devices("cpu")[0]):
    # llama_config = LLaMAConfig.load_config(FLAGS.load_llama_config)
    _, params = StreamingCheckpointer.load_trainstate_checkpoint(
        model_path, disallow_trainstate=True
    )
client_params_dict = {
    "transformer": {
        "wte": params['params']["transformer"]["wte"],
        "ln_f": params['params']["transformer"]["ln_f"],
        "h": {str(i): params['params']["transformer"]["h"][str(i)] for i in range(2)},
    }
}

mid_params_dict = {
    "transformer": {
        "h": {str(i): params['params']["transformer"]["h"][str(i)] for i in range(2, 3)}
    }
}

server_params_dict = {
    "transformer": {
        "ln_f": params['params']["transformer"]["ln_f"],
        "h": {
            str(i): params['params']["transformer"]["h"][str(i)]
            for i in range(3, len(params['params']["transformer"]["h"]))
        },
    },
    "lm_head": {
        "kernel": params['params']["lm_head"]["kernel"],
    },
}


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
# for embedding generation
def embeding_generation(input_ids, params):
    config = LLaMAConfig()
    model = FlaxLLaMAForCausalLMClient(config=config)
    smasheddata, attention_mask, position_ids = model(
        input_ids=input_ids, params=params
    )
    del model
    return smasheddata, attention_mask, position_ids


def mid_generation(input_ids, params, attention_mask, position_ids):
    config = LLaMAConfig()
    _model = FlaxLLaMAForCausalLMMid(config=config)

    _smasheddata = _model(
        input_ids=input_ids,
        params=params,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )

    return _smasheddata, attention_mask, position_ids


def server_generation(input_ids, params, attention_mask, position_ids):
    config = LLaMAConfig()
    _model = FlaxLLaMAForCausalLMServer(config=config)

    _smasheddata = _model(
        input_ids=input_ids,
        params=params,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )

    return _smasheddata


def run_on_cpu(token_num=9):
    input_ids = tokenizer.encode(
        'Q: What is the largest animal?\nA:', return_tensors='jax'
    )
    for _ in range(token_num):
        smasheddata, attention_mask, position_ids = embeding_generation(
            input_ids=input_ids, params=client_params_dict
        )

        _smasheddata, attention_mask, position_ids = mid_generation(
            input_ids=smasheddata,
            params=mid_params_dict,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        outputs = server_generation(
            input_ids=_smasheddata,
            params=server_params_dict,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        next_token_logits = outputs[0][0, -1, :]
        next_token = jnp.argmax(next_token_logits)

        input_ids = jnp.concatenate([input_ids, jnp.array([[next_token]])], axis=1)

    return input_ids


def run_on_spu(token_num=9):
    # encode context the generation is conditioned on
    input_ids = tokenizer.encode(
        'Q: What is the largest animal?\nA:', return_tensors='jax'
    )
    for _ in range(token_num):
        smasheddata, attention_mask, position_ids = embeding_generation(
            input_ids=input_ids, params=client_params_dict
        )
        with hack_softmax_context(
            "hack exp of softmax", enabled=True
        ), hack_silu_context("hack silu", enabled=True):
            _input_ids = ppd.device("P1")(lambda x: x)(smasheddata)
            _params = ppd.device("P2")(lambda x: x)(mid_params_dict)

            _smasheddata, attention_mask, position_ids = ppd.device("SPU")(
                mid_generation
            )(_input_ids, _params, attention_mask, position_ids)

            _smasheddata, attention_mask, position_ids = (
                ppd.get(_smasheddata),
                ppd.get(attention_mask),
                ppd.get(position_ids),
            )

        outputs = server_generation(
            input_ids=_smasheddata,
            params=server_params_dict,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        next_token_logits = outputs[0][0, -1, :]
        next_token = jnp.argmax(next_token_logits)

        input_ids = jnp.concatenate([input_ids, jnp.array([[next_token]])], axis=1)

    return input_ids


if __name__ == '__main__':
    print('\n------\nRun on CPU')
    start_time = time.time()
    outputs_ids = run_on_cpu()
    print(tokenizer.decode(outputs_ids[0], skip_special_tokens=True))
    end_time = time.time()
    print(f"generate on CPU: {end_time - start_time} seconds")

    print('\n------\nRun on SPU')
    start_time = time.time()
    outputs_ids = run_on_spu()
    print(tokenizer.decode(outputs_ids[0], skip_special_tokens=True))
    end_time = time.time()
    print(f"generate  on SPU: {end_time - start_time} seconds")
