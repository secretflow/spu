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
# > bazel run -c opt //examples/python/ml/flax_mlp:flax_mlp

import argparse
import json
from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from sklearn import metrics

import examples.python.utils.dataset_utils as dsutil
import spu.utils.distributed as ppd

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


def train_auto_grad(x, y, params, n_batch=10, n_epochs=10, step_size=0.01):
    xs = jnp.array_split(x, len(x) / n_batch, axis=0)
    ys = jnp.array_split(y, len(y) / n_batch, axis=0)

    def body_fun(_, loop_carry):
        params = loop_carry
        for x, y in zip(xs, ys):
            _, grads = jax.value_and_grad(loss_func)(params, x, y)
            params = jax.tree_util.tree_map(
                lambda p, g: p - step_size * g, params, grads
            )
        return params

    params = jax.lax.fori_loop(0, n_epochs, body_fun, params)
    return params


# Model init is purely public and run on SPU leads to significant accuracy loss, thus hoist out and run in python
def model_init(n_batch=10):
    model = MLP(FEATURES)
    return model.init(jax.random.PRNGKey(1), jnp.ones((n_batch, FEATURES[0])))


def run_on_cpu():
    x_train, y_train = dsutil.breast_cancer(slice(None, None, None), True)
    params = model_init()
    params = jax.jit(train_auto_grad)(x_train, y_train, params)

    x_test, y_test = dsutil.breast_cancer(slice(None, None, None), False)
    y_predict = predict(params, x_test)
    print("AUC(cpu)={}".format(metrics.roc_auc_score(y_test, y_predict)))
    return params


parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="examples/python/conf/3pc.json")
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])

import cloudpickle as pickle
import tempfile


def compute_score(param, type):
    x_test, y_test = dsutil.breast_cancer(slice(None, None, None), False)
    y_predict = predict(param, x_test)
    score = metrics.roc_auc_score(y_test, y_predict)
    print(f"AUC({type})={score}")
    return score


def run_on_spu():
    @ppd.device("SPU")
    def main(x1, x2, y, params):
        x = jnp.concatenate((x1, x2), axis=1)
        return train_auto_grad(x, y, params)

    x1, y = ppd.device("P1")(dsutil.breast_cancer)(slice(None, 15), True)
    x2, _ = ppd.device("P2")(dsutil.breast_cancer)(slice(15, None), True)

    params = model_init()
    params = main(x1, x2, y, params)

    return params


def save_and_load_model():
    # 1. run with spu
    params = run_on_spu()

    # 2. save metadata and spu objects.
    meta = ppd.save(params)

    spu_model_file = tempfile.NamedTemporaryFile()
    spu_model_file_name = spu_model_file.name
    with open(spu_model_file_name, "wb") as f:
        pickle.dump(meta, f)

    # 3. load metadata and spu objects.
    with open(spu_model_file_name, "rb") as f:
        meta_ = pickle.load(f)
    params_ = ppd.load(meta_)
    params_r = ppd.get(params_)

    x_test, y_test = dsutil.breast_cancer(slice(None, None, None), False)
    y_predict = predict(params_r, x_test)
    print(
        "AUC(save_and_load_model)={}".format(metrics.roc_auc_score(y_test, y_predict))
    )


if __name__ == '__main__':
    print('\n------\nRun on CPU')
    p = run_on_cpu()
    compute_score(p, 'cpu')
    print('\n------\nRun on SPU')
    p = ppd.get(run_on_spu())
    compute_score(p, 'spu')
    save_and_load_model()
