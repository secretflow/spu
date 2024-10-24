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


# Start nodes.
# > bazel run -c opt //examples/python/utils:nodectl -- up
#
# Run this example script.
# > bazel run -c opt //examples/python:millionaire


import argparse
import json
import logging

import jax.numpy as jnp
import numpy as np

import spu.utils.distributed as ppd

parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="examples/python/conf/3pc.json")
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])


@ppd.device("P1")
def rand_from_alice():
    logging.info("make a private variable on P1, it's known only for P1.")
    np.random.seed()
    return np.random.randint(100, size=(4,))


@ppd.device("P2")
def rand_from_bob():
    logging.info("make a private variable on P2, it's known only for P2.")
    np.random.seed()
    return np.random.randint(100, size=(4,))


@ppd.device("SPU")
def compare(x, y):
    logging.info("compute the max of two parameter, unknown for all parties.")
    return jnp.maximum(x, y)


x = rand_from_alice()
y = rand_from_bob()

# x & y will be automatically fetch by SPU (as secret shares)
# z will be evaluated as an SPU function.
z = compare(x, y)

print(f"z({type(z)} is a device object ref, we can not access it directly.")
print(f"use ppd.get to get the object from device to this host")
print(f"x = {ppd.get(x)}")
print(f"y = {ppd.get(y)}")
print(f"z = {ppd.get(z)}")
