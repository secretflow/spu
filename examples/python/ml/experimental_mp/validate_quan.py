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
# > bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/ml/experimental_mp/3pc.json up
#
# Run this example script.
# > bazel run -c opt //examples/python/ml/experimental_mp:validate_quan

import argparse
import json

import jax.numpy as jnp
from transformers import AutoTokenizer, FlaxGPT2LMHeadModel, GPT2Config

import spu.utils.distributed as ppd
import spu.spu_pb2 as spu_pb2

copts = spu_pb2.CompilerOptions()
copts.enable_pretty_print = True
copts.pretty_print_dump_dir = "ppdump"
copts.xla_pp_kind = 2  # HTML format

parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument(
    "-c", "--config", default="examples/python/ml/experimental_mp/3pc.json"
)
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])


from flax import linen as nn
import jax
import jax.numpy as jnp


def softmax(x, axis: -1):
    x_max = jnp.max(x, axis, keepdims=True)
    x_norm = (x - x_max).astype(jnp.float32)
    unnormalized = jnp.exp(x_norm)
    result = unnormalized / jnp.sum(unnormalized, axis, keepdims=True)
    return result


def mat_test():
    key = jax.random.PRNGKey(0)

    dtype = jnp.float16
    # 拆分随机键以生成新的键
    key, subkey = jax.random.split(key)

    x = jax.random.uniform(key, shape=(8, 10), dtype=jnp.float32).astype(dtype)
    y = jax.random.uniform(key, shape=(768, 768 * 4), dtype=jnp.float32).astype(dtype)

    # standard matmul
    fn = lambda x, y: jnp.matmul(x, y)

    # quantized matmul
    fn = lambda x, y: jnp.clip(jnp.matmul(x, y).astype(jnp.int32) * 0.25, -127, 128)

    # quantized softmax
    fn = lambda x, y: softmax(x, -1)

    x_spu = ppd.device("P1")(lambda x: x)(x)
    y_spu = ppd.device("P2")(lambda x: x)(y)
    outputs = ppd.device("SPU")(fn, copts=copts)(x_spu, y_spu)
    print(ppd.get(outputs))


if __name__ == '__main__':
    print('\n------\nRun on SPU')
    mat_test()
