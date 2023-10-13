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

import time

import jax.numpy as jnp
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

import sml.utils.emulation as emulation
from sml.tree.tree import DecisionTreeClassifier as sml_dtc

MAX_DEPTH = 2
CONFIG_FILE = emulation.CLUSTER_ABY3_3PC


def emul_tree(mode=emulation.Mode.MULTIPROCESS):
    def proc_wrapper(max_depth=2, n_labels=3):
        dt = sml_dtc(max_depth=max_depth, criterion='gini', splitter='best', n_labels=n_labels)
        def proc(X, y):
            dt_fit = dt.fit(X, y)
            result = dt_fit.predict(X)
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
        new_features = (
            jnp.greater_equal(iris_data[:, :], new_threshold[:, jnp.newaxis, :]) + 1 - 1
        )
        new_features = new_features.transpose([1, 0, 2]).reshape(n_samples, -1)

        X, y = new_features, iris_label
        return X, y

    try:
        # bandwidth and latency only work for docker mode
        emulator = emulation.Emulator(CONFIG_FILE, mode, bandwidth=300, latency=20)
        emulator.up()

        # load mock data
        X, y = load_data()
        n_samples = y.shape[0]
        n_labels = jnp.unique(y).shape[0]

        # compare with sklearn
        clf = DecisionTreeClassifier(
            max_depth=MAX_DEPTH, criterion='gini', splitter='best', random_state=None
        )
        start = time.time()
        clf = clf.fit(X, y)
        score_plain = clf.score(X, y)
        end = time.time()
        print(f"Running time in SKlearn: {end - start:.2f}s")

        # mark these data to be protected in SPU
        X_spu, y_spu = emulator.seal(X, y)

        # run
        proc = proc_wrapper(MAX_DEPTH, n_labels)
        start = time.time()
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
    emul_tree(emulation.Mode.MULTIPROCESS)
