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
# > bazel run //examples/python/ml:flax_mlp

import argparse
import json
from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from sklearn import metrics

import examples.python.utils.dataset_utils as dsutil
import spu.binding.util.distributed as ppd

FEATURES = [30, 15, 8, 1]


class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x


def predict(params, x):
    return MLP(FEATURES).apply(params, x)


def loss_func(params, x, y):
    pred = predict(params, x)

    def mse(y, pred):
        def squared_error(y, y_pred):
            # TODO: check this
            return jnp.multiply(y - y_pred, y - y_pred) / 2.0
            # return jnp.inner(y - y_pred, y - y_pred) / 2.0 # fail, (10, 1) inner (10, 1) -> (10, 10), have to be (10,) inner (10,) -> scalar

        return jnp.mean(squared_error(y, pred))

    return mse(y, pred)


def train_auto_grad(x, y, n_batch=10, n_epochs=10, step_size=0.001):
    model = MLP(FEATURES)
    params = model.init(jax.random.PRNGKey(1), jnp.ones((n_batch, FEATURES[0])))
    xs = jnp.array_split(x, len(x) / n_batch, axis=0)
    ys = jnp.array_split(y, len(y) / n_batch, axis=0)

    def body_fun(_, loop_carry):
        params = loop_carry
        for (x, y) in zip(xs, ys):
            _, grads = jax.value_and_grad(loss_func)(params, x, y)
            params = jax.tree_util.tree_map(
                lambda p, g: p - step_size * g, params, grads
            )
        return params

    params = jax.lax.fori_loop(0, n_epochs, body_fun, params)
    return params


def run_on_cpu():
    x_train, y_train = dsutil.breast_cancer(slice(None, None, None), True)
    train_auto_grad_jit = jax.jit(train_auto_grad)
    params = train_auto_grad_jit(x_train, y_train)

    x_test, y_test = dsutil.breast_cancer(slice(None, None, None), False)
    y_predict = predict(params, x_test)
    print("AUC(cpu)={}".format(metrics.roc_auc_score(y_test, y_predict)))


parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="examples/python/conf/3pc.json")
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])


def run_on_spu():
    @ppd.device("SPU")
    def main(x1, x2, y):
        x = jnp.concatenate((x1, x2), axis=1)
        return train_auto_grad(x, y)

    x1, y = ppd.device("P1")(dsutil.breast_cancer)(slice(None, 15), True)
    x2, _ = ppd.device("P2")(dsutil.breast_cancer)(slice(15, None), True)
    params = main(x1, x2, y)
    params = ppd.get(params)

    x_test, y_test = dsutil.breast_cancer(slice(None, None, None), False)
    y_predict = predict(params, x_test)
    print("AUC(spu)={}".format(metrics.roc_auc_score(y_test, y_predict)))


if __name__ == '__main__':
    print('\n------\nRun on CPU')
    run_on_cpu()
    print('\n------\nRun on SPU')
    run_on_spu()
