# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
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
# > bazel run -c opt //examples/python/ml/ss_lr:ss_lr

import argparse
import json
import time
import logging
from enum import Enum
from sklearn.metrics import roc_auc_score, explained_variance_score
from typing import Dict, List

import examples.python.utils.dataset_utils as dsutil

import jax.numpy as jnp
import numpy as np
import examples.python.utils.appr_sigmoid as Sigmoid
import spu.utils.distributed as ppd


def sigmoid(x):
    return Sigmoid.t1_sig(x)


class RegType(Enum):
    Linear = 'linear'
    Logistic = 'logistic'


class Penalty(Enum):
    NONE = 'None'
    L1 = 'l1'  # not supported
    L2 = 'l2'


def place_dataset(xs: List[np.ndarray], y: np.ndarray):
    assert xs[0].shape[0] == y.shape[0], "x/y not aligned"
    assert len(y.shape) == 1 or y.shape[1] == 1, "y should be list or 1D array"

    x = jnp.concatenate(xs, axis=1)
    y = y.reshape((y.shape[0], 1))
    return {
        'x': x,
        'y': y,
    }


def predict(
    dataset: Dict[str, np.ndarray],
    w: np.ndarray,
    reg_type: str,
):
    x = dataset['x']
    num_feat = x.shape[1]
    samples = x.shape[0]
    assert w.shape[0] == num_feat + 1, f"w shape is mismatch to x={x.shape}"
    assert len(w.shape) == 1 or w.shape[1] == 1, "w should be list or 1D array"
    w.reshape((w.shape[0], 1))

    bias = w[-1, 0]
    w = jnp.resize(w, (num_feat, 1))

    pred = jnp.matmul(x, w) + bias

    if reg_type == RegType.Logistic:
        pred = sigmoid(pred)
    return pred


def epoch_update_w(
    dataset: Dict[str, np.ndarray],
    w: np.ndarray,
    learning_rate: float,
    l2_norm: float,
    reg_type: RegType,
    penalty: Penalty,
    total_batch: int,
    batch_size: int,
) -> np.ndarray:
    """
    update weights on dataset in one iteration.

    Args:
        dataset: input datasets.
        w: base model weights.
        learning_rate: controls how much to change the model in one epoch.
        batch_size: how many samples use in one calculation.
        reg_type: Linear or Logistic regression.
        penalty: The penalty (aka regularization term) to be used.
        l2_norm: L2 regularization term.

    Return:
        W after update.
    """
    x = dataset['x']
    y = dataset['y']
    assert x.shape[0] >= total_batch * batch_size, "total batch is too large"
    num_feat = x.shape[1]
    assert w.shape[0] == num_feat + 1, "w shape is mismatch to x"
    assert len(w.shape) == 1 or w.shape[1] == 1, "w should be list or 1D array"
    w = w.reshape((w.shape[0], 1))

    for idx in range(total_batch):
        begin = idx * batch_size
        end = (idx + 1) * batch_size
        # padding one col for bias in w
        x_slice = jnp.concatenate((x[begin:end, :], jnp.ones((batch_size, 1))), axis=1)
        y_slice = y[begin:end, :]

        pred = jnp.matmul(x_slice, w)
        if reg_type == RegType.Logistic:
            pred = sigmoid(pred)

        err = pred - y_slice
        grad = jnp.matmul(jnp.transpose(x_slice), err)

        if penalty == Penalty.L2:
            w_with_zero_bias = jnp.resize(w, (num_feat, 1))
            w_with_zero_bias = jnp.concatenate(
                (w_with_zero_bias, jnp.zeros((1, 1))),
                axis=0,
            )
            grad = grad + w_with_zero_bias * l2_norm

        step = (learning_rate * grad) / batch_size

        w = w - step

    return w


class SSLR:
    def __init__(self, spu: ppd.SPU) -> None:
        self.spu = spu

    def fit(
        self,
        xs: List[ppd.PYU.Object],
        y: ppd.PYU.Object,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        reg_type: str,
        penalty: str,
        l2_norm: float,
    ) -> ppd.SPU.Object:
        # prepare parameters.
        assert epochs > 0, f"epochs should >0"
        assert learning_rate > 0, f"learning_rate should >0"
        assert batch_size > 0, f"batch_size should >0"
        assert penalty != 'l1', "not support L1 penalty for now"
        if penalty == Penalty.L2:
            assert l2_norm > 0, f"l2_norm should >0 if use L2 penalty"
        assert reg_type in [
            e.value for e in RegType
        ], f"reg_type should in {[e.value for e in RegType]}, but got {reg_type}"
        assert penalty in [
            e.value for e in Penalty
        ], f"penalty should in {[e.value for e in Penalty]}, but got {reg_type}"

        num_sample = xs[0].shape[0]
        num_feat = sum(x.shape[1] for x in xs)
        batch_size = min(batch_size, num_sample)
        total_batch = int(num_sample / batch_size)
        penalty = Penalty(penalty)
        reg_type = RegType(reg_type)

        # prepare dataset on ppd.SPU
        start = time.time()
        spu_ds = self.spu(place_dataset)(xs, y)
        logging.info(f"infeed times: {time.time() - start}s")

        # init weights on ppd.SPU
        def init_w(base: float, num_feat: int) -> np.ndarray:
            # last one is bias
            return jnp.full((num_feat + 1, 1), base, dtype=jnp.float32)

        start = time.time()
        spu_w = self.spu(init_w, static_argnums=(0, 1))(0, num_feat)
        logging.info(f"Init times: {time.time() - start}s")

        # do train on ppd.SPU
        epoch_idx = 0
        while epoch_idx < epochs:
            epoch_idx += 1
            start = time.time()
            spu_w = self.spu(epoch_update_w, static_argnums=(2, 3, 4, 5, 6, 7))(
                spu_ds,
                spu_w,
                learning_rate,
                l2_norm,
                reg_type,  # 4
                penalty,  # 5
                total_batch,  # 6
                batch_size,  # 7
            )

            logging.info(f"epoch {epoch_idx} times: {time.time() - start}s")

        return spu_w

    def predict(
        self,
        xs: List[ppd.PYU.Object],
        y: ppd.PYU.Object,
        w: ppd.SPU.Object,
        reg_type: str,
    ) -> ppd.SPU.Object:
        assert reg_type in [
            e.value for e in RegType
        ], f"reg_type should in {[e.value for e in RegType]}, but got {reg_type}"

        ds = self.spu(place_dataset)(xs, y)
        yhat = self.spu(predict, static_argnums=(2,))(
            ds,
            w,
            RegType(reg_type),
        )
        return yhat


parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="examples/python/conf/3pc.json")
parser.add_argument(
    "-d",
    "--dataset_config",
    default="examples/python/conf/ds_mock_regression_basic.json",
)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--learning_rate", default=0.1, type=float)
parser.add_argument("--batch_size", default=1024, type=int)
parser.add_argument("--penalty", default="None", choices=["None", "L1", "L2"])
parser.add_argument("--l2_norm", default=0.0, type=float)
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = json.load(file)

with open(args.dataset_config, "r") as f:
    dataset_config = json.load(f)

ppd.init(conf["nodes"], conf["devices"])

REGTYPE = None


def train():
    if dataset_config["problem_type"] == "regression":
        REGTYPE = RegType.Linear.value
    elif dataset_config["problem_type"] == "classification":
        REGTYPE = RegType.Logistic.value
    x1, x2, y = dsutil.load_dataset_by_config(dataset_config)
    x1, y = ppd.device("P1")(dsutil.load_feature_r1)(x1, y)
    x2 = ppd.device("P2")(dsutil.load_feature_r2)(x2)

    xs = [x1, x2]

    start = time.time()
    sslr = SSLR(ppd.device("SPU"))
    model = sslr.fit(
        xs,
        y,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        reg_type=REGTYPE,
        penalty=args.penalty,
        l2_norm=args.l2_norm,
    )
    train_time = time.time() - start
    print(f"train time {train_time}")

    start = time.time()
    yhat = ppd.get(sslr.predict(xs=xs, y=y, w=model, reg_type=REGTYPE))
    predict_time = time.time() - start
    print(f"predict time {predict_time}")

    if REGTYPE == RegType.Linear.value:
        score = explained_variance_score(ppd.get(y), ppd.get(yhat))
        print(f"explained variance score {score}")
    else:
        score = roc_auc_score(ppd.get(y), ppd.get(yhat))
        print(f"auc {score}")
    return score, train_time, predict_time


if __name__ == '__main__':
    train()
