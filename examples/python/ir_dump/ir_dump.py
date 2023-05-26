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
# > bazel run -c opt //examples/python/utils:nodectl -- up
#
# Run this example script.
# > bazel run -c opt //examples/python/ir_dump:ir_dump


import argparse
import json
import logging
import os

import jax.numpy as jnp
import numpy as np

import spu.utils.distributed as ppd
import spu.spu_pb2 as spu_pb2

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description="distributed driver.")
parser.add_argument("-c", "--config", default="examples/python/conf/3pc.json")
parser.add_argument("-d", "--dir", default="ppdump")
args = parser.parse_args()

with open(args.config, "r") as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])


# enable dump ir to custom path
dump_path = os.path.join(os.path.expanduser("~"), args.dir)
logging.info(f"Dump path: {dump_path}")
# refer to spu.proto for more detailed configuration
copts = spu_pb2.CompilerOptions()
copts.enable_pretty_print = True
copts.pretty_print_dump_dir = dump_path
copts.xla_pp_kind = 2


def func(x, y):
    """
    Any custom function that consists of operators that SPU currently supports.
    Here, we define a `max` function use jax.numpy.
    """
    return jnp.maximum(x, y)


def get_data(seed=123):
    """
    Any IO function that loads the data.
    """
    np.random.seed(seed)
    data = np.random.randn(3, 4)
    return data


def main():
    # CPU plaintext version
    x = get_data(1)
    y = get_data(2)

    res = func(x, y)
    logging.info(f"\nx: {x}\ny: {y}\nres: {res}")

    # SPU secure version
    x_spu = ppd.device("P1")(lambda x: x)(x)
    y_spu = ppd.device("P2")(lambda x: x)(y)

    res_spu = ppd.device("SPU")(func, copts=copts)(x_spu, y_spu)
    logging.info(f"\nx: {ppd.get(x_spu)}\ny: {ppd.get(y_spu)}\nres: {ppd.get(res_spu)}")

    assert jnp.allclose(res, ppd.get(res_spu), 1e-4)


if __name__ == "__main__":
    main()
