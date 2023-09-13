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


import jax.numpy as jnp
import pandas as pd
import sklearn.linear_model as sk
from sklearn.datasets import load_iris

import sml.utils.emulation as emulation
from sml.linear_model.pla import Perceptron


def emul_perceptron(mode: emulation.Mode.MULTIPROCESS):
    def proc(x, y):
        model = Perceptron(
            max_iter=20,
            eta0=1.0,
            penalty='elasticnet',
            alpha=0.001,
            fit_intercept=True,
            l1_ratio=0.7,
            patience=10,
            batch_size=64,
            early_stop=True,
        )

        return model.fit(x, y).predict(x)

    def load_data():
        iris = load_iris()
        df = pd.DataFrame(
            iris.data,
            columns=['sepal length', 'sepal width', 'petal length', 'petal width'],
        )
        df['label'] = iris.target

        # only use sepal length and sepal width features
        # 100 samples
        data = jnp.array(df.iloc[0:100, [0, 1, -1]])
        x, y = data[:, :-1], data[:, -1]

        # y is -1 or 1
        y = jnp.sign(y)
        y = jnp.where(y <= 0, -1, y)
        y = y.reshape((y.shape[0], 1))

        return x, y

    try:
        # bandwidth and latency only work for docker mode
        emulator = emulation.Emulator(
            emulation.CLUSTER_ABY3_3PC, mode, bandwidth=300, latency=20
        )
        emulator.up()

        # load mock data
        x, y = load_data()
        n_samples = len(y)

        # compare with sklearn
        sk_pla = sk.Perceptron(
            max_iter=20,
            eta0=1.0,
            penalty='elasticnet',
            alpha=0.001,
            l1_ratio=0.7,
            fit_intercept=True,
        )
        result_sk = sk_pla.fit(x, y).predict(x)
        result_sk = result_sk.reshape(result_sk.shape[0], 1)
        acc_sk = jnp.sum((result_sk == y)) / n_samples * 100

        # mark these data to be protected in SPU
        x_spu, y_spu = emulator.seal(x, y)

        # run
        result = emulator.run(proc)(x_spu, y_spu)
        result = result.reshape(result.shape[0], 1)
        acc_ = jnp.sum((result == y)) / n_samples * 100

        # print acc
        print(f"Accuracy in SKlearn: {acc_sk:.2f}%")
        print(f"Accuracy in SPU: {acc_:.2f}%")

    finally:
        emulator.down()


if __name__ == "__main__":
    emul_perceptron(emulation.Mode.MULTIPROCESS)
