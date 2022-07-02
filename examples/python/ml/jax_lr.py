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
# > bazel run //examples/python/ml:jax_lr


import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np
from sklearn import metrics

import examples.python.utils.dataset_utils as dsutil
import spu.binding.util.distributed as ppd


# FIXME: For un-normalized data, grad(sigmoid) is likely to overflow, either with exp/tanh or taylor series
# https://stackoverflow.com/questions/68290850/jax-autograd-of-a-sigmoid-always-returns-nan
def sigmoid(x):
    # return 0.5 * (jnp.tanh(x / 2) + 1)
    return 1 / (1 + jnp.exp(-x))


def predict(x, w, b):
    return sigmoid(jnp.matmul(x, w) + b)


def loss(x, y, w, b):
    pred = predict(x, w, b)
    label_prob = pred * y + (1 - pred) * (1 - y)
    return -jnp.mean(jnp.log(label_prob))


class LogitRegression:
    def __init__(self, n_epochs=10, n_iters=10, step_size=0.1):
        self.n_epochs = n_epochs
        self.n_iters = n_iters
        self.step_size = step_size

    def fit_auto_grad(self, feature, label):
        w = jnp.zeros(feature.shape[1])
        b = 0.0

        xs = jnp.array_split(feature, self.n_iters, axis=0)
        ys = jnp.array_split(label, self.n_iters, axis=0)

        def body_fun(_, loop_carry):
            w_, b_ = loop_carry
            for (x, y) in zip(xs, ys):
                grad = jax.grad(loss, argnums=(2, 3))(x, y, w_, b_)
                w_ -= grad[0] * self.step_size
                b_ -= grad[1] * self.step_size

            return w_, b_

        return jax.lax.fori_loop(0, self.n_epochs, body_fun, (w, b))

    def fit_manual_grad(self, feature, label):
        w = jnp.zeros(feature.shape[1])
        b = 0.0

        xs = jnp.array_split(feature, self.n_iters, axis=0)
        ys = jnp.array_split(label, self.n_iters, axis=0)

        def body_fun(_, loop_carry):
            w_, b_ = loop_carry
            for (x, y) in zip(xs, ys):
                pred = predict(x, w_, b_)
                err = pred - y
                w_ -= jnp.matmul(jnp.transpose(x), err) / y.shape[0] * self.step_size
                b_ -= jnp.mean(err) * self.step_size

            return w_, b_

        return jax.lax.fori_loop(0, self.n_epochs, body_fun, (w, b))


parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="examples/python/conf/3pc.json")
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])


def run_on_cpu():
    x_train, y_train = dsutil.breast_cancer(slice(None, None, None), True)

    lr = LogitRegression()

    w0, b0 = jax.jit(lr.fit_auto_grad)(x_train, y_train)
    print(w0, b0)

    x_test, y_test = dsutil.breast_cancer(slice(None, None, None), False)
    print(
        "AUC(cpu, auto_grad)={}".format(
            metrics.roc_auc_score(y_test, predict(x_test, w0, b0))
        )
    )

    w1, b1 = jax.jit(lr.fit_manual_grad)(x_train, y_train)
    print(w1, b1)
    print(
        "AUC(cpu, manual_grad)={}".format(
            metrics.roc_auc_score(y_test, predict(x_test, w1, b1))
        )
    )


def run_on_spu():
    @ppd.device("SPU")
    def train(x1, x2, y):
        x = jnp.concatenate((x1, x2), axis=1)
        lr = LogitRegression()
        return lr.fit_auto_grad(x, y)

    x1, y = ppd.device("P1")(dsutil.breast_cancer)(slice(None, 15), True)
    x2, _ = ppd.device("P2")(dsutil.breast_cancer)(slice(15, None), True)
    W, b = train(x1, x2, y)

    W_r, b_r = ppd.get(W), ppd.get(b)
    print(W_r, b_r)

    x_test, y_test = dsutil.breast_cancer(slice(None, None, None), False)
    print(
        "AUC(spu)={}".format(metrics.roc_auc_score(y_test, predict(x_test, W_r, b_r)))
    )


if __name__ == "__main__":
    print('Run on CPU\n------\n')
    run_on_cpu()
    print('Run on SPU\n------\n')
    run_on_spu()
