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
# > bazel run -c opt //examples/python/ml/jax_svm:jax_svm


import argparse
import json

import jax
import jax.numpy as jnp
from sklearn import metrics

import examples.python.utils.dataset_utils as dsutil
import spu.utils.distributed as ppd


def predict(x, w, b):
    approx = jnp.dot(x, w) - b
    return jnp.sign(approx)


# The jax SVM implementation refers to the numpy version at
# https://github.com/python-engineer/MLfromscratch/blob/master/mlfromscratch/svm.py
class LinearSVM:
    def __init__(self, n_epochs=1000, step_size=0.001, lambda_param=0.01):
        self.n_epochs = n_epochs
        self.step_size = step_size
        self.lambda_param = lambda_param

    def fit(self, feature, label):
        n_samples, n_features = feature.shape

        w = jnp.zeros(n_features)
        b = 0

        xs = jnp.array(feature)
        ys = jnp.array(label)

        def epoch_loop(_, loop_carry):
            wi, bi = loop_carry

            def sample_loop(i, loop_carry):
                wj, bj = loop_carry
                x = xs[i]
                y = ys[i]
                condition = y * (jnp.dot(x, wj) - bj) >= 1
                delta_w = jnp.where(
                    condition,
                    self.step_size * (2 * self.lambda_param * wj),
                    self.step_size * (2 * self.lambda_param * wj - jnp.dot(x, y)),
                )
                delta_b = jnp.where(condition, 0, self.step_size * y)
                wj -= delta_w
                bj -= delta_b
                return wj, bj

            return jax.lax.fori_loop(0, n_samples, sample_loop, (wi, bi))

        return jax.lax.fori_loop(0, self.n_epochs, epoch_loop, (w, b))


parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="examples/python/conf/3pc.json")
parser.add_argument("--n_epochs", default=1000, type=int)
parser.add_argument("--step_size", default=0.001, type=float)
parser.add_argument("--lambda_param", default=0.01, type=float)
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])


def compute_score(w, b, type):
    x_test, y_test = dsutil.breast_cancer(slice(None, None, None), False)
    y_test = jnp.where(y_test <= 0, -1, 1)
    score = metrics.accuracy_score(y_test, predict(x_test, w, b))
    print(f"AUC({type})={score}")
    return score


def run_on_cpu():
    x_train, y_train = dsutil.breast_cancer(slice(None, None, None), True)

    svm = LinearSVM(args.n_epochs, args.step_size, args.lambda_param)
    y_train = jnp.where(y_train <= 0, -1, 1)
    w, b = jax.jit(svm.fit)(x_train, y_train)
    print(w, b)

    return w, b


def run_on_spu():
    @ppd.device("SPU")
    def train(x1, x2, y):
        x = jnp.concatenate((x1, x2), axis=1)
        svm = LinearSVM(args.n_epochs, args.step_size, args.lambda_param)
        y = jnp.where(y <= 0, -1, 1)
        return svm.fit(x, y)

    x1, y = ppd.device("P1")(dsutil.breast_cancer)(slice(None, 15), True)
    x2, _ = ppd.device("P2")(dsutil.breast_cancer)(slice(15, None), True)
    w, b = train(x1, x2, y)

    w_r, b_r = ppd.get(w), ppd.get(b)
    print(w_r, b_r)

    return w_r, b_r


if __name__ == "__main__":
    print('Run on CPU\n------\n')
    w, b = run_on_cpu()
    compute_score(w, b, 'cpu')
    print('Run on SPU\n------\n')
    w, b = run_on_spu()
    compute_score(w, b, 'spu')
