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
# > bazel run -c opt //examples/python/utils:nodectl -- -c examples/python/conf/2pc_semi2k.json up
#
# Run this example script.
# > bazel run -c opt //examples/python/ml/jax_lr:jax_lr


import argparse
import json

import jax
import jax.numpy as jnp
from sklearn import metrics

import examples.python.utils.dataset_utils as dsutil
import spu
import spu.utils.distributed as ppd


# FIXME: For un-normalized data, grad(sigmoid) is likely to overflow, either with exp/tanh or taylor series
# https://stackoverflow.com/questions/68290850/jax-autograd-of-a-sigmoid-always-returns-nan
def sigmoid(x):
    # return 0.5 * (jnp.tanh(x / 2) + 1)
    return 1 / (1 + jnp.exp(-x))


def predict(x, w, b):
    return sigmoid(jnp.matmul(x, w) + b)


def loss(x, y, w, b, use_cache):
    if use_cache:
        w = spu.experimental.make_cached_var(w)
        b = spu.experimental.make_cached_var(b)
    pred = predict(x, w, b)
    label_prob = pred * y + (1 - pred) * (1 - y)

    if use_cache:
        w = spu.experimental.drop_cached_var(w, label_prob)
        b = spu.experimental.drop_cached_var(b, label_prob)

    return -jnp.mean(jnp.log(label_prob))


class LogitRegression:
    def __init__(self, n_epochs=10, n_iters=10, step_size=0.1):
        self.n_epochs = n_epochs
        self.n_iters = n_iters
        self.step_size = step_size

    def fit_auto_grad(self, feature, label, use_cache=False):
        w = jnp.zeros(feature.shape[1])
        b = 0.0

        if use_cache:
            feature = spu.experimental.make_cached_var(feature)

        xs = jnp.array_split(feature, self.n_iters, axis=0)
        ys = jnp.array_split(label, self.n_iters, axis=0)

        def body_fun(_, loop_carry):
            w_, b_ = loop_carry
            for x, y in zip(xs, ys):
                grad = jax.grad(loss, argnums=(2, 3))(x, y, w_, b_, use_cache)
                w_ -= grad[0] * self.step_size
                b_ -= grad[1] * self.step_size

            return w_, b_

        ret = jax.lax.fori_loop(0, self.n_epochs, body_fun, (w, b))

        if use_cache:
            feature = spu.experimental.drop_cached_var(feature, *ret)

        return ret

    def fit_manual_grad(self, feature, label, use_cache=False):
        w = jnp.zeros(feature.shape[1])
        b = 0.0

        if use_cache:
            feature = spu.experimental.make_cached_var(feature)

        xs = jnp.array_split(feature, self.n_iters, axis=0)
        ys = jnp.array_split(label, self.n_iters, axis=0)

        def body_fun(_, loop_carry):
            w_, b_ = loop_carry
            for x, y in zip(xs, ys):
                pred = predict(x, w_, b_)
                err = pred - y
                w_ -= jnp.matmul(jnp.transpose(x), err) / y.shape[0] * self.step_size
                b_ -= jnp.mean(err) * self.step_size

            return w_, b_

        ret = jax.lax.fori_loop(0, self.n_epochs, body_fun, (w, b))

        if use_cache:
            feature = spu.experimental.drop_cached_var(feature, *ret)

        return ret


def run_on_cpu(x_train, y_train):
    lr = LogitRegression()

    w0, b0 = jax.jit(lr.fit_auto_grad)(x_train, y_train)
    print(w0, b0)

    w1, b1 = jax.jit(lr.fit_manual_grad)(x_train, y_train)

    return [w0, w1], [b0, b1]


SPU_OBJECT_META_PATH = "/tmp/driver_spu_jax_lr_object.txt"

import cloudpickle as pickle


def save_and_load_model(x_test, y_test, W, b):
    # 1. save metadata and spu objects.
    meta = ppd.save((W, b))
    with open(SPU_OBJECT_META_PATH, "wb") as f:
        pickle.dump(meta, f)

    # 2. load metadata and spu objects.
    with open(SPU_OBJECT_META_PATH, "rb") as f:
        meta_ = pickle.load(f)
    W_, b_ = ppd.load(meta_)

    W_r, b_r = ppd.get(W_), ppd.get(b_)
    print(W_r, b_r)

    score = metrics.roc_auc_score(y_test, predict(x_test, W_r, b_r))
    print("AUC(save_and_load_model)={}".format(score))

    return score


def compute_score(x_test, y_test, W_r, b_r, type):
    score = metrics.roc_auc_score(y_test, predict(x_test, W_r, b_r))
    print(f"AUC({type})={score}")
    return score


def run_on_spu(x, y, use_cache=False, auto_grad=False):
    @ppd.device("SPU")
    def train(x1, x2, y):
        x = jnp.concatenate((x1, x2), axis=1)
        lr = LogitRegression()
        if auto_grad:
            return lr.fit_auto_grad(x, y, use_cache)
        else:
            return lr.fit_manual_grad(x, y, use_cache)

    x1 = ppd.device("P1")(lambda x: x[:, :50])(x)
    x2 = ppd.device("P2")(lambda x: x[:, 50:])(x)
    y = ppd.device("P1")(lambda x: x)(y)
    W, b = train(x1, x2, y)

    return W, b


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='distributed driver.')
    parser.add_argument(
        "-c", "--config", default="examples/python/conf/2pc_semi2k.json"
    )
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        conf = json.load(file)

    ppd.init(conf["nodes"], conf["devices"])

    x, y = dsutil.mock_classification(10 * 10000, 100, 0.0, 42)

    print('Run on CPU\n------\n')
    w, b = run_on_cpu(x, y)
    compute_score(x, y, w[0], b[0], 'cpu, auto_grad')
    compute_score(x, y, w[1], b[1], 'cpu, manual_grad')
    print('Run on SPU\n------\n')
    # without cache
    # total send bytes 2376240800, recv bytes 2376240800
    w, b = run_on_spu(x, y)
    w_r, b_r = ppd.get(w), ppd.get(b)
    compute_score(x, y, w_r, b_r, 'spu')
    save_and_load_model(x, y, w, b)

    print('Run on SPU with cache\n------\n')
    # with semi2k beaver cache
    # total send bytes 856240800, recv bytes 856240800
    # Reduced communication bytes by 64%
    w, b = run_on_spu(x, y, True)
    w_r, b_r = ppd.get(w), ppd.get(b)
    compute_score(x, y, w_r, b_r, 'spu_cached')

    print('Run on SPU auto_grad\n------\n')
    w, b = run_on_spu(x, y, True, True)
    w_r, b_r = ppd.get(w), ppd.get(b)
    compute_score(x, y, w_r, b_r, 'spu, auto_grad')
