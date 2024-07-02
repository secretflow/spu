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
'''
Author: Li Zhihang
Date: 2024-06-16 12:02:32
LastEditTime: 2024-06-22 16:52:19
FilePath: /klaus/spu-klaus/sml/forest/emulations/forest_emul.py
Description: 
'''
import time

import jax.numpy as jnp
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

import sml.utils.emulation as emulation
from sml.forest.forest import RandomForestClassifier as sml_rfc

MAX_DEPTH = 3
CONFIG_FILE = emulation.CLUSTER_ABY3_3PC


def emul_forest(mode=emulation.Mode.MULTIPROCESS):
    def proc_wrapper(
        n_estimators=100,
        max_features=None,
        n_features=200,
        criterion='gini',
        splitter='best',
        max_depth=3,
        bootstrap=True,
        max_samples=None,
        n_labels=3,
        seed=0,
    ):
        rf_custom = sml_rfc(
            n_estimators,
            max_features,
            n_features,
            criterion,
            splitter,
            max_depth,
            bootstrap,
            max_samples,
            n_labels,
            seed,
        )

        def proc(X, y):
            rf_custom_fit = rf_custom.fit(X, y)
            result = rf_custom_fit.predict(X)
            return result

        return proc

    def load_data():
        iris = load_iris()
        iris_data, iris_label = jnp.array(iris.data), jnp.array(iris.target)
        # sorted_features: n_samples * n_features_in
        n_samples, n_features_in = iris_data.shape
        n_labels = len(jnp.unique(iris_label))
        sorted_features = jnp.sort(iris_data, axis=0)
        new_threshold = (sorted_features[:-1, :] + sorted_features[1:, :]) / 2
        new_features = jnp.greater_equal(
            iris_data[:, :], new_threshold[:, jnp.newaxis, :]
        )
        new_features = new_features.transpose([1, 0, 2]).reshape(n_samples, -1)

        X, y = new_features[:, ::3], iris_label[:]
        return X, y

    try:
        # bandwidth and latency only work for docker mode
        emulator = emulation.Emulator(CONFIG_FILE, mode, bandwidth=300, latency=20)
        emulator.up()

        # load mock data
        X, y = load_data()
        n_samples, n_features = X.shape
        n_labels = jnp.unique(y).shape[0]

        # compare with sklearn
        rf = RandomForestClassifier(
            n_estimators=3,
            max_features=None,
            criterion='gini',
            max_depth=MAX_DEPTH,
            bootstrap=False,
            max_samples=None,
        )
        start = time.time()
        rf = rf.fit(X, y)
        score_plain = rf.score(X, y)
        end = time.time()
        print(f"Running time in SKlearn: {end - start:.2f}s")

        # mark these data to be protected in SPU
        X_spu, y_spu = emulator.seal(X, y)

        # run
        proc = proc_wrapper(
            n_estimators=3,
            max_features=0.7,
            n_features=n_features,
            criterion='gini',
            splitter='best',
            max_depth=3,
            bootstrap=False,
            max_samples=None,
            n_labels=n_labels,
            seed=0,
        )
        start = time.time()
        # 不可以使用bootstrap，否则在spu运行的正确率很低
        result = emulator.run(proc)(X_spu, y_spu)
        end = time.time()
        score_encrpted = jnp.sum((result == y)) / n_samples
        print(f"Running time in SPU: {end - start:.2f}s")

        # print acc
        print(f"Accuracy in SKlearn: {score_plain:.2f}")
        print(f"Accuracy in SPU: {score_encrpted:.2f}")

    finally:
        emulator.down()


if __name__ == "__main__":
    emul_forest(emulation.Mode.MULTIPROCESS)
