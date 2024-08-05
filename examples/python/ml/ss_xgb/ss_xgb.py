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
# > bazel run -c opt //examples/python/ml/ss_xgb:ss_xgb

import argparse
import json
from statistics import mode
import time

from sklearn.metrics import roc_auc_score

from typing import Any, Dict, List, Tuple
import jax.numpy as jnp
import numpy as np
import pandas as pd
from functools import reduce

import examples.python.utils.dataset_utils as dsutil
import examples.python.utils.appr_sigmoid as Sigmoid
import spu.utils.distributed as ppd

from spu.utils.distributed import PYU, SPU

parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="examples/python/conf/3pc.json")
# use small dataset for this example
parser.add_argument(
    "-d", "--dataset_config", default="examples/python/conf/ds_breast_cancer_basic.json"
)

args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = json.load(file)


with open(args.dataset_config, "r") as f:
    dataset_config = json.load(f)

ppd.init(conf["nodes"], conf["devices"])


class XgbModel:
    def __init__(self) -> None:
        # List[PYU.Object of XgbTree], owned by pyu, only knows split value if feature belong to this pyu.
        self.trees = list()
        # List[SPU.Object of np.array], owned by spu and not reveal to any one
        self.weights = list()
        # TODO how to ser/der ?

    class XgbTree:
        def __init__(self) -> None:
            self.split_features = list()
            self.split_values = list()

        def insert_split_node(self, feature: int, value: float):
            assert isinstance(feature, int), f"feature {feature}"
            assert isinstance(value, float), f"value {value}"
            self.split_features.append(feature)
            self.split_values.append(value)


####### Xgb SPU functions #######
def sigmoid(x):
    return Sigmoid.sr_sig(x)


def compute_obj(G: np.ndarray, H: np.ndarray):
    # lambda == 0.5
    return G * (G / (H + 0.5))


def compute_weight(G: float, H: float):
    # lambda == 0.5
    # learn rate = 0.3
    return -((G / (H + 0.5)) * 0.3)


def get_weight(context: Dict[str, Any], s: np.ndarray):
    g_sum = (context['g'] * s).sum(axis=1)
    h_sum = (context['h'] * s).sum(axis=1)
    return compute_weight(g_sum, h_sum)


def compute_gh(y: np.ndarray, pred: np.ndarray):
    yhat = sigmoid(pred)
    g = yhat - y
    h = yhat * (1 - yhat)
    return g, h


def spu_global_setup(buckets_map: List[np.ndarray], y: np.ndarray) -> Dict[str, Any]:
    context = dict()
    context['buckets_map'] = jnp.concatenate(buckets_map, axis=1)
    context['y'] = y

    return context


def spu_tree_setup(context: Dict[str, Any], pred: np.ndarray) -> Dict[str, Any]:
    gh = compute_gh(context['y'], pred)
    context['g'] = gh[0]
    context['h'] = gh[1]

    return context


def find_best_split_bucket(
    context: Dict[str, Any], l_nodes_s: List[np.ndarray], last_level: bool
):
    if 'cache' in context:
        GL_cache = context['cache'][0]
        HL_cache = context['cache'][1]
        assert len(GL_cache) == len(l_nodes_s)
    else:
        # root level, no cache
        GL_cache = None
        HL_cache = None

    level_GL = list()
    level_HL = list()
    for idx in range(len(l_nodes_s)):
        sg = context['g'] * l_nodes_s[idx]
        sh = context['h'] * l_nodes_s[idx]
        lchild_GL = jnp.matmul(sg, context['buckets_map'])
        lchild_HL = jnp.matmul(sh, context['buckets_map'])
        level_GL.append(lchild_GL)
        level_HL.append(lchild_HL)
        if GL_cache is not None:
            level_GL.append(GL_cache[idx] - lchild_GL)
            level_HL.append(HL_cache[idx] - lchild_HL)

    if not last_level:
        context['cache'] = (level_GL, level_HL)
    elif 'cache' in context:
        del context['cache']

    GL = jnp.concatenate(level_GL, axis=0)
    HL = jnp.concatenate(level_HL, axis=0)

    GR = GL[:, -1].reshape(-1, 1) - GL
    HR = HL[:, -1].reshape(-1, 1) - HL

    obj_l = compute_obj(GL, HL)
    obj_r = compute_obj(GR, HR)

    obj = obj_l[:, -1].reshape(-1, 1)

    # gamma == 0
    gain = obj_l + obj_r - obj

    split_buckets = jnp.argmax(gain, axis=1)

    return split_buckets, context


def init_pred(base: float, samples: int):
    shape = (1, samples)
    return jnp.full(shape, base)


def root_select(samples: int):
    return jnp.ones((1, samples))


def get_child_select(nodes_s: np.ndarray, lchilds_s: np.ndarray):
    childs_s = list()
    for current, lchile in zip(nodes_s, lchilds_s):
        ls = current * lchile
        rs = current - ls
        childs_s.append(ls)
        childs_s.append(rs)
    return childs_s


def predict_tree_weight(selects: List[np.ndarray], weights: np.ndarray):
    # select = jnp.prod(selects, axis=0)
    # prod requires ndarray or scalar arguments, got <class 'list'> at position 0.
    # jnp.prod has little different with np.prod.
    select = selects[0]
    for i in range(1, len(selects)):
        select = select * selects[i]
    assert (
        select.shape[1] == weights.shape[0]
    ), f"select {select.shape}, weights {weights.shape}"

    return jnp.matmul(select, weights)


def do_leaf(context: Dict[str, Any], ss: List[np.ndarray]):
    s = jnp.concatenate(ss, axis=0)
    return get_weight(context, s)


####### Xgb PYU functions #######
def predict_weight_select(x: np.ndarray, tree: XgbModel.XgbTree):
    split_nodes = len(tree.split_features)

    select = np.zeros((x.shape[0], split_nodes + 1), dtype=np.int8)
    # should parallel in c++
    for r in range(x.shape[0]):
        row = x[r, :]
        idxs = list()
        idxs.append(0)
        while len(idxs):
            idx = idxs.pop(0)
            if idx < split_nodes:
                f = tree.split_features[idx]
                v = tree.split_values[idx]
                if f == -1:
                    idxs.append(idx * 2 + 1)
                    idxs.append(idx * 2 + 2)
                else:
                    if row[f] <= v:
                        idxs.append(idx * 2 + 1)
                    else:
                        idxs.append(idx * 2 + 2)
            else:
                leaf_idx = idx - split_nodes
                select[r, leaf_idx] = 1

    return select


def build_maps(context: Dict[str, Any], x: np.ndarray):
    buckets_map = np.zeros(
        (x.shape[0], x.shape[1] * context['buckets']), dtype=np.bool_
    )
    context['order_map'] = np.zeros((x.shape[0], x.shape[1]), dtype=np.int8)
    context['split_points'] = list()
    for f in range(x.shape[1]):
        bins, split_point = pd.qcut(
            x[:, f], context['buckets'], labels=False, duplicates='drop', retbins=True
        )
        context['order_map'][:, f] = bins
        sum_bin = None
        for b in range(split_point.size - 1):
            bin = np.flatnonzero(bins == b)
            if sum_bin is None:
                sum_bin = bin
            else:
                sum_bin = np.concatenate((sum_bin, bin), axis=None)
            buckets_map[sum_bin, f * context['buckets'] + b] = 1
        # todo: remove empty bin
        context['split_points'].append(list(np.delete(split_point, (0,))))
    return buckets_map


def pyu_global_setup(x: np.ndarray, buckets: int):
    context = dict()
    context['buckets'] = buckets
    buckets_map = build_maps(context, x)

    return buckets_map, context


def pyu_tree_setup(context: Dict[str, Any]):
    context['tree'] = XgbModel.XgbTree()
    return context


def tree_finish(context: Dict[str, Any]) -> XgbModel.XgbTree:
    return context['tree']


def do_split(context: Dict[str, Any], split_bucket: int):
    if split_bucket == -1:
        context['tree'].insert_split_node(-1, float("inf"))
        return context
    else:
        # todo: not work if build_maps remove empty bin
        feature = int(split_bucket / context['buckets'])
        split_point_idx = split_bucket % context['buckets']
        context['tree'].insert_split_node(
            feature, context['split_points'][feature][split_point_idx]
        )
        # lchild' select
        ls = (
            (context['order_map'][:, feature] <= split_point_idx)
            .astype(np.int8)
            .reshape(1, context['order_map'].shape[0])
        )

        return ls, context


class SSXgb:
    def __init__(self, spu: SPU) -> None:
        self.spu = spu

    def _update_predict_tree(
        self,
        pred: SPU.Object,
        dataset: List[PYU.Object],
        tree: List[XgbModel.XgbTree],
        weight: SPU.Object,
    ):
        assert len(tree) == len(dataset)

        weight_selects = list()
        for idx in range(len(dataset)):
            assert tree[idx].device == dataset[idx].device
            weight_selects.append(
                dataset[idx].device(predict_weight_select)(dataset[idx], tree[idx])
            )

        current = self.spu(predict_tree_weight)(weight_selects, weight)
        if pred:
            return self.spu(lambda x, y: x + y)(pred, current)
        else:
            return current

    def train(
        self,
        trees: int,
        depth: int,
        buckets: int,
        dataset: List[PYU.Object],
        y: PYU.Object,
    ):
        self.trees = trees
        self.depth = depth
        self.buckets = buckets
        self.pyus = [data.device for data in dataset]
        self.samples = dataset[0].shape[0]

        # todo: subsample / colsample
        buckets_map = list()
        self.buckets_size = list()
        self.pyus_context = list()
        for r in range(len(self.pyus)):
            m, context = self.pyus[r](pyu_global_setup)(dataset[r], self.buckets)
            buckets_map.append(m)
            self.buckets_size.append(m.shape[1])
            self.pyus_context.append(context)

        self.spu_context = self.spu(spu_global_setup)(buckets_map, y)
        del buckets_map

        pred = self.spu(init_pred, static_argnums=(0, 1))(0, self.samples)
        model = XgbModel()
        while len(model.trees) < self.trees:
            for r in range(len(self.pyus)):
                self.pyus_context[r] = self.pyus[r](pyu_tree_setup)(
                    self.pyus_context[r]
                )
            self.spu_context = self.spu(spu_tree_setup)(self.spu_context, pred)

            tree, weight = self.train_tree()
            model.trees.append(tree)
            model.weights.append(weight)

            if len(model.trees) < self.trees:
                pred = self._update_predict_tree(pred, dataset, tree, weight)

        return model

    def _split_rank(self, split_bucket: int):
        pre_end_pos = 0
        for r in range(len(self.buckets_size)):
            current_end_pod = pre_end_pos + self.buckets_size[r]
            if split_bucket < current_end_pod:
                return r, split_bucket - pre_end_pos
            pre_end_pos += self.buckets_size[r]

        assert False, "should not be here"

    def train_level(self, nodes_s, level):
        last_level = level == (self.depth - 1)
        l_nodes_s = [nodes_s[idx] for idx in range(len(nodes_s)) if idx % 2 == 0]
        spu_split_buckets, self.spu_context = self.spu(
            find_best_split_bucket, static_argnums=(2,)
        )(self.spu_context, l_nodes_s, last_level)
        split_buckets = list(ppd.get(spu_split_buckets))
        assert len(split_buckets) == len(nodes_s)
        lchilds_s = list()
        for s in split_buckets:
            split_rank, split_rank_idx = self._split_rank(s)
            for r in range(len(self.pyus)):
                if r == split_rank:
                    lchild_s, self.pyus_context[r] = self.pyus[r](do_split)(
                        self.pyus_context[r], split_rank_idx
                    )
                    lchilds_s.append(lchild_s)
                else:
                    self.pyus_context[r] = self.pyus[r](do_split)(
                        self.pyus_context[r], -1
                    )
        assert len(lchilds_s) == len(split_buckets)
        childs_s = self.spu(get_child_select)(nodes_s, lchilds_s)
        return childs_s

    def train_tree(self):
        root_s = self.spu(root_select, static_argnums=(0,))(self.samples)
        nodes_s = (root_s,)
        for level in range(self.depth + 1):
            if level < self.depth:
                # split nodes
                nodes_s = self.train_level(nodes_s, level)
            else:
                # leaf nodes
                weight = self.spu(do_leaf)(self.spu_context, nodes_s)

        tree = [
            self.pyus[r](tree_finish)(self.pyus_context[r])
            for r in range(len(self.pyus))
        ]
        return tree, weight

    def predict(self, dataset: List[PYU.Object], model: XgbModel) -> SPU.Object:
        if len(model.trees) == 0:
            return None
        assert len(dataset) == len(model.trees[0])
        pred = None
        for idx in range(len(model.trees)):
            pred = self._update_predict_tree(
                pred, dataset, model.trees[idx], model.weights[idx]
            )

        return self.spu(sigmoid)(pred)


def main():
    x1, x2, y = dsutil.load_dataset_by_config(dataset_config)
    x1, y = ppd.device("P1")(dsutil.load_feature_r1)(x1, y)
    x2 = ppd.device("P2")(dsutil.load_feature_r2)(x2)

    dataset = [x1, x2]

    start = time.time()
    ss_xgb = SSXgb(ppd.device("SPU"))
    model = ss_xgb.train(3, 3, 4, dataset, y)
    train_time = time.time() - start
    print(f"train time {train_time}")

    start = time.time()
    yhat = ppd.get(ss_xgb.predict(dataset, model))
    predict_time = time.time() - start
    print(f"predict time {predict_time}")
    score = roc_auc_score(ppd.get(y), yhat)
    print(f"auc {score}")
    return score, train_time, predict_time


if __name__ == '__main__':
    main()
