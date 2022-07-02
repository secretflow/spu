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


# Linear regression using GradientTape
# based on https://sanjayasubedi.com.np/deeplearning/tensorflow-2-linear-regression-from-scratch/

import numpy as np
import tensorflow as tf
from sklearn import metrics

# This example is to show tf program could be converted to XLA IR.
# Run this example script.
# > bazel run //examples/python/ml:tf_experiment


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
            feature[
                :-remainder,
            ],
            self.n_iters,
            axis=0,
        )
        ys = tf.split(
            label[
                :-remainder,
            ],
            self.n_iters,
            axis=0,
        )

        for _ in range(self.n_epochs):
            for (x, y) in zip(xs, ys):
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
            feature[
                :-remainder,
            ],
            self.n_iters,
            axis=0,
        )
        ys = tf.split(
            label[
                :-remainder,
            ],
            self.n_iters,
            axis=0,
        )

        for _ in range(self.n_epochs):
            for (x, y) in zip(xs, ys):
                pred = predict(x, self.w)
                err = pred - y
                dw = tf.linalg.matvec(tf.transpose(x), err)
                self.w.assign_sub(dw * self.step_size)

        return self.w


def load_dataset():
    from sklearn.datasets import load_breast_cancer

    ds = load_breast_cancer()
    x, y = ds['data'], ds['target']

    def normalize(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    x = normalize(x)
    y = y.astype(dtype=np.float64)
    return x, y


def compile_to_xla(fn, *args, **kwargs):
    """
    This method demostrate the method to run tf function on spu via tf->xla->spu path.
    It's not supported on SPU yet.
    """
    convert_method = lambda shape, dtype: tf.convert_to_tensor(np.zeros(shape, dtype))

    tf_fn = tf.function(fn, jit_compile=True, experimental_relax_shapes=True)
    xla = tf_fn.experimental_get_compiler_ir(*args, **kwargs)(
        stage="hlo",  # "hlo_serialized"
    )
    cf = tf_fn.get_concrete_function(*args, **kwargs)

    return xla, cf


def run():
    x, y = load_dataset()
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

    # print(LogitRegression().fit_manual_grad(x1, x2, y))
    # w = tf.function(
    #     tf.autograph.experimental.do_not_convert(LogitRegression().fit_manual_grad),
    #     jit_compile=True,
    #     experimental_relax_shapes=True,
    # )(x1, x2, y).numpy()


if __name__ == '__main__':
    run()
