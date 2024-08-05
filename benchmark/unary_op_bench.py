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

import argparse
import json

import jax.numpy as jnp
import numpy as np
import time

import spu.utils.distributed as ppd

parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="../examples/python/conf/3pc.json")
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])


@ppd.device("P1")
def rand_from_alice():
    np.random.seed()
    return np.random.rand(10000000)


@ppd.device("SPU")
def foo(x):
    return jnp.max(x)


x = rand_from_alice()

# x & y will be automatically fetch by SPU (as secret shares)
# z will be evaluated as a SPU function.
t = time.time()
z = foo(x)
elapsed = time.time() - t

print(f'elapsed time = {elapsed}')

print(f"x = {ppd.get(x)}")
print(f"z = {ppd.get(z)}")
