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
# > bazel run -c opt //examples/python/stats:woe


import argparse
import json
from tkinter.messagebox import NO

import jax
import numpy as np
import pandas as pd
from jax import numpy as jnp
from scipy.stats import norm

import examples.python.utils.dataset_utils as dsutil
import spu.utils.distributed as ppd

parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="examples/python/conf/3pc.json")
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = json.load(file)
    ppd.init(conf["nodes"], conf["devices"])

BUCKETS_SIZE = 10


def build_select_map(x):
    smap = np.zeros((x.shape[1] * BUCKETS_SIZE, x.shape[0]))
    for col in range(x.shape[1]):
        idxs = pd.qcut(x[:, col], BUCKETS_SIZE, labels=False)
        for sample in range(idxs.shape[0]):
            bucket = idxs[sample]
            smap[col * BUCKETS_SIZE + bucket, sample] = 1.0
    return smap, np.sum(smap, axis=1)


def load_select_map_r2():
    x, _ = dsutil.breast_cancer(slice(15, None))
    return build_select_map(x)


def woe_calc(total_counts, positive_counts, positive_label_cnt, negative_label_cnt):
    assert total_counts.shape == positive_counts.shape

    negative_counts = total_counts - positive_counts
    woe = np.zeros(total_counts.shape)

    def calc(p, n, pl, nl):
        import math

        if p == 0 or n == 0:
            positive_distrib = (p + 0.5) / pl
            negative_distrib = (n + 0.5) / nl
            return math.log(positive_distrib / negative_distrib)
        else:
            positive_distrib = np.double(p) / pl
            negative_distrib = np.double(n) / nl
            return math.log(positive_distrib / negative_distrib)

    for idx in range(total_counts.shape[0]):
        woe[idx] = calc(
            positive_counts[idx],
            negative_counts[idx],
            positive_label_cnt,
            negative_label_cnt,
        )

    return woe


def woe_calc_for_master():
    x, y = dsutil.breast_cancer(slice(None, 15))

    total_counts = np.zeros(x.shape[1] * BUCKETS_SIZE)
    positive_counts = np.zeros(x.shape[1] * BUCKETS_SIZE)
    for col in range(x.shape[1]):
        idxs = pd.qcut(x[:, col], BUCKETS_SIZE, labels=False)
        for sample in range(idxs.shape[0]):
            bucket = idxs[sample]
            total_counts[col * BUCKETS_SIZE + bucket] += 1
            positive_counts[col * BUCKETS_SIZE + bucket] += y[sample]

    positive_label_cnt = np.sum(y)
    negative_label_cnt = y.shape[0] - positive_label_cnt

    woe = woe_calc(
        total_counts, positive_counts, positive_label_cnt, negative_label_cnt
    )

    return woe


def woe_calc_for_peer(totals, positives):
    total_counts = np.around(totals)
    positive_counts = np.around(positives)

    _, y = dsutil.breast_cancer()
    positive_label_cnt = np.sum(y)
    negative_label_cnt = y.shape[0] - positive_label_cnt

    woe = woe_calc(
        total_counts, positive_counts, positive_label_cnt, negative_label_cnt
    )

    return woe


def ssbuckercounter(s2, y):
    return jnp.matmul(s2, y).flatten()


def run_spu():
    s2, t2 = ppd.device("P2")(load_select_map_r2)()
    _, y = ppd.device("P1")(dsutil.breast_cancer)()
    positive_counts_r2 = ppd.device("SPU")(ssbuckercounter)(s2, y)
    woe_r1 = ppd.device("P1")(woe_calc_for_master)()
    woe_r2 = ppd.device("P1")(woe_calc_for_peer)(t2, positive_counts_r2)

    print(ppd.get(woe_r1))
    print(ppd.get(woe_r2))
    # TODO: woe categories split points.


def run_origin():
    x, _ = dsutil.breast_cancer()
    smap = build_select_map(x[:, 0:1])
    print(smap)


if __name__ == "__main__":
    print("run_origin:")
    run_origin()
    print("run_spu:")
    run_spu()
