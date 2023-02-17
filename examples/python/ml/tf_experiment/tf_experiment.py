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


# Linear regression using GradientTape
# based on https://sanjayasubedi.com.np/deeplearning/tensorflow-2-linear-regression-from-scratch/

import argparse
import json

import numpy as np
import tensorflow as tf
from sklearn import metrics
import spu.utils.distributed as ppd

# This example is to show tf program could be converted to XLA IR and run by SPU.
# Start nodes.
# > bazel run -c opt //examples/python/utils:nodectl -- up
#
# Run this example script.
# > bazel run -c opt //examples/python/ml/tf_experiment:tf_experiment
# This example is tf counterpart to jax_lr


def sigmoid(x):
    return 1 / (1 + tf.exp(-x))


def predict(x, w):
    return sigmoid(tf.linalg.matvec(x, w))


def loss(x, y, w):
    pred = predict(x, w)
    label_prob = pred * y + (1 - pred) * (1 - y)
    # return -tf.reduce_mean(tf.math.log(label_prob))
    return -tf.reduce_sum(tf.math.log(label_prob))


class LogitRegression:
    def __init__(self, n_epochs=3, n_iters=10, step_size=0.1):
        self.n_epochs = n_epochs
        self.n_iters = n_iters
        self.step_size = step_size
        self.w = None

    def fit_auto_grad(self, x1, x2, label):
        feature = tf.concat((x1, x2), axis=1)
        if self.w is None:
            self.w = tf.Variable(np.zeros([feature.shape[1]], dtype=np.float64))

        remainder = feature.shape[0] % self.n_iters

        xs = tf.split(
            feature[:-remainder,],
            self.n_iters,
            axis=0,
        )
        ys = tf.split(
            label[:-remainder,],
            self.n_iters,
            axis=0,
        )

        for _ in range(self.n_epochs):
            for x, y in zip(xs, ys):
                with tf.GradientTape() as t:
                    current_loss = loss(x, y, self.w)

                (dw,) = t.gradient(current_loss, [self.w])
                self.w.assign_sub(dw * self.step_size)

        return self.w

    # w -= alpha * x.t * (sigmoid(x * w) - y)
    def fit_manual_grad(self, x1, x2, label):
        feature = tf.concat((x1, x2), axis=1)
        if self.w is None:
            self.w = tf.Variable(np.zeros([feature.shape[1]], dtype=np.float64))

        remainder = feature.shape[0] % self.n_iters

        xs = tf.split(
            feature[:-remainder,],
            self.n_iters,
            axis=0,
        )
        ys = tf.split(
            label[:-remainder,],
            self.n_iters,
            axis=0,
        )

        for _ in range(self.n_epochs):
            for x, y in zip(xs, ys):
                pred = predict(x, self.w)
                err = pred - y
                dw = tf.linalg.matvec(tf.transpose(x), err)
                self.w.assign_sub(dw * self.step_size)

        return self.w

    # NOTE(junfeng): at this moment, SPU doesn't support stateful computations with TF frontend,
    # so self.w is removed and tf.Variable is replaced accordingly.
    def fit_manual_grad_no_captures(self, x1, x2, label):
        feature = tf.concat((x1, x2), axis=1)
        w = tf.constant(np.zeros([feature.shape[1]], dtype=np.float64))

        remainder = feature.shape[0] % self.n_iters

        xs = tf.split(
            feature[:-remainder,],
            self.n_iters,
            axis=0,
        )
        ys = tf.split(
            label[:-remainder,],
            self.n_iters,
            axis=0,
        )

        for _ in range(self.n_epochs):
            for x, y in zip(xs, ys):
                pred = predict(x, w)
                err = pred - y
                dw = tf.linalg.matvec(tf.transpose(x), err)
                w -= dw * self.step_size

        return w


def breast_cancer(
    col_slicer=slice(None, None, None),
    train: bool = True,
    *,
    normalize: bool = True,
):
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    ds = load_breast_cancer()
    x, y = ds['data'], ds['target']

    # only difference to dsutil.breast_cancer
    y = y.astype(dtype=np.float64)

    if normalize:
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    if train:
        x_ = x_train
        y_ = y_train
    else:
        x_ = x_test
        y_ = y_test
    x_ = x_[:, col_slicer]
    return x_, y_


def compile_to_xla(fn, *args, **kwargs):
    """
    This method demonstrate the method to run tf function on spu via tf->xla->spu path.
    It's not supported on SPU yet.
    """
    tf_fn = tf.function(fn, jit_compile=True, experimental_relax_shapes=True)
    xla = tf_fn.experimental_get_compiler_ir(*args, **kwargs)(
        stage="hlo",  # "hlo_serialized"
    )
    cf = tf_fn.get_concrete_function(*args, **kwargs)

    return xla, cf


parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="examples/python/conf/3pc.json")
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"], framework=ppd.Framework.EXP_TF)


def tf_to_xla():
    x, y = breast_cancer()
    x1, x2 = x[:, :15], x[:, 15:]

    print("\n--- direct run ---")
    params = LogitRegression().fit_auto_grad(x1, x2, y)
    print(f"params: {params}")

    print("\n--- tf.function run ---")
    params = tf.function(
        tf.autograph.experimental.do_not_convert(LogitRegression().fit_auto_grad),
        jit_compile=True,
        experimental_relax_shapes=True,
    )(x1, x2, y)
    print(f"params: {params}")

    auc = metrics.roc_auc_score(y, predict(x, params))
    print(f"AUC={auc}")

    print("\n--- tf compilation ---")
    xla, cf = compile_to_xla(LogitRegression().fit_auto_grad, x1, x2, y)
    print(f"xla: {xla}")
    print(f"cf: {cf}")


import time


def run_fit_manual_grad_cpu():
    print('Run on CPU\n------\n')
    x, y = breast_cancer()
    x1, x2 = x[:, :15], x[:, 15:]
    start_ts = time.time()
    params = LogitRegression().fit_manual_grad(x1, x2, y)
    end_ts = time.time()
    x_test, y_test = breast_cancer(slice(None, None, None), False)
    auc = metrics.roc_auc_score(y_test, predict(x_test, params))
    print(f"AUC(cpu)={auc}, time={end_ts-start_ts}")


def run_fit_manual_grad_spu():
    print('Run on SPU\n------\n')
    x1, y = ppd.device("P1")(breast_cancer)(slice(None, 15), True)
    x2, _ = ppd.device("P2")(breast_cancer)(slice(15, None), True)
    start_ts = time.time()
    W = ppd.device('SPU')(LogitRegression().fit_manual_grad_no_captures)(x1, x2, y)
    end_ts = time.time()
    W_r = ppd.get(W)
    x_test, y_test = breast_cancer(slice(None, None, None), False)

    score = metrics.roc_auc_score(y_test, predict(x_test, W_r.astype(dtype=np.float64)))
    print(
        "AUC(spu)={}".format(score),
        f"time={end_ts-start_ts}",
    )
    return score


if __name__ == '__main__':
    # tf_to_xla()
    run_fit_manual_grad_cpu()
    run_fit_manual_grad_spu()
