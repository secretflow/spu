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


import unittest

import jax
import jax.numpy as jnp
import numpy as np
import sklearn
from absl.testing import absltest, parameterized
from sklearn import metrics
from sklearn.datasets import load_breast_cancer

import spu.utils.simulation as ppsim
import spu.spu_pb2 as spu_pb2


# Note: for un-normalized data, grad(sigmoid) is likely to overflow, either with exp/tanh or taylor series
# https://stackoverflow.com/questions/68290850/jax-autograd-of-a-sigmoid-always-returns-nan
def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def load_dataset():
    ds = load_breast_cancer()
    x, y = ds['data'], ds['target']
    x = normalize(x)
    y = y.astype(dtype=np.float64)
    return x, y


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
            for x, y in zip(xs, ys):
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
            for x, y in zip(xs, ys):
                pred = predict(x, w_, b_)
                err = pred - y
                w_ -= jnp.matmul(jnp.transpose(x), err) / y.shape[0] * self.step_size
                b_ -= jnp.mean(err) * self.step_size

            return w_, b_

        return jax.lax.fori_loop(0, self.n_epochs, body_fun, (w, b))


@parameterized.product(
    wsize=(2, 3),
    prot=(
        spu_pb2.ProtocolKind.SEMI2K,
        spu_pb2.ProtocolKind.ABY3,
        spu_pb2.ProtocolKind.CHEETAH,
    ),
    field=(spu_pb2.FieldType.FM64, spu_pb2.FieldType.FM128),
)
class UnitTests(parameterized.TestCase):
    def test_sslr(self, wsize, prot, field):
        if prot == spu_pb2.ProtocolKind.ABY3 and wsize != 3:
            return
        if prot == spu_pb2.ProtocolKind.CHEETAH and (
            wsize != 2 or field != spu_pb2.FieldType.FM64
        ):
            return

        x, y = load_dataset()
        lr = LogitRegression()
        sim = ppsim.Simulator.simple(wsize, prot, field)
        w1, b1 = ppsim.sim_jax(sim, lr.fit_manual_grad)(x, y)
        auc = metrics.roc_auc_score(y, predict(x, w1, b1))
        self.assertGreater(auc, 0.96)

        # print(w1, b1)
        # print("AUC={}".format(metrics.roc_auc_score(y, predict(x, w1, b1))))


if __name__ == "__main__":
    unittest.main()
