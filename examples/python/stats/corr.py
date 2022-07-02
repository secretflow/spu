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
# > bazel run //examples/python/stats:corr

import argparse
import json

import jax
import jax.numpy as jnp

import examples.python.utils.dataset_utils as dsutil
import spu.binding.util.distributed as ppd

parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="examples/python/conf/3pc.json")
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])


@ppd.device("P1")
def load_feature_r1():
    # TODO: pre-process for constant column
    x, _ = dsutil.breast_cancer(slice(None, 15))
    return dsutil.standardize(x)


@ppd.device("P2")
def load_feature_r2():
    # TODO: pre-process for constant column
    x, _ = dsutil.breast_cancer(slice(15, None))
    return dsutil.standardize(x)


def run_spu():
    x1 = load_feature_r1()
    x2 = load_feature_r2()

    @ppd.device("SPU")
    def XTX(x1, x2):
        x = jnp.concatenate((x1, x2), axis=1)
        return jnp.matmul(x.transpose(), x), x.shape[0]

    ss_xtx, rows = XTX(x1, x2)
    corr = ppd.get(ss_xtx) / ppd.get(rows)
    print(corr)


def run_origin():
    x, _ = dsutil.breast_cancer()
    std_x = dsutil.standardize(x)
    corr = jnp.matmul(std_x.transpose(), std_x) / (x.shape[0] - 1)
    print(corr)


if __name__ == '__main__':
    print("run_origin:")
    run_origin()
    print("run_spu:")
    run_spu()
