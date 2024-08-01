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
import jax.numpy as jnp
import numpy as np
import spu.utils.distributed as ppd

VECTOR_LEN = 1024 * 1024

parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="examples/python/ml/flax_gpt2/3pc.json")
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])

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

def sample_input():
    np.random.seed()
    x = np.random.randint(-10, 10, VECTOR_LEN)
    return x

def exp_msb():
    x = sample_input()
    x_spu = ppd.device("P1")(lambda x: x)(x)
    msb_spu = ppd.device("SPU")(msb)(x_spu)

def exp_eqz():
    x = sample_input()
    x_spu = ppd.device("P1")(lambda x: x)(x)
    eqz_spu = ppd.device("SPU")(eqz)(x_spu)

def exp_ppa():
    x = sample_input()
    x_spu = ppd.device("P1")(lambda x: x)(x)
    b2a_spu = ppd.device("SPU")(b2a)(x_spu)

def exp_a2b():
    x = sample_input()
    x_spu = ppd.device("P1")(lambda x: x)(x)
    b2a_spu = ppd.device("SPU")(a2b)(x_spu)


if __name__ == '__main__':
    exp_msb()
    exp_eqz()
    exp_ppa()
    exp_a2b()
