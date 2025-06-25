# Copyright 2024 Ant Group Co., Ltd.
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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

import sml.utils.emulation as emulation
from sml.ensemble.adaboost import AdaBoostClassifier as sml_Adaboost
from sml.tree.tree import DecisionTreeClassifier as sml_dtc

MAX_DEPTH = 3
CONFIG_FILE = emulation.CLUSTER_ABY3_3PC


def emul_ada(mode=emulation.Mode.MULTIPROCESS):
    def proc_wrapper(
        estimator,
        n_estimators,
        learning_rate,
        algorithm,
        epsilon,
    ):
        ada_custom = sml_Adaboost(
            estimator=estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm=algorithm,
            epsilon=epsilon,
        )

        def proc(X, y):
            ada_custom_fit = ada_custom.fit(X, y, sample_weight=None)
            result = ada_custom_fit.predict(X)
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
        n_labels = jnp.unique(y).shape[0]

        # compare with sklearn
        base_estimator = DecisionTreeClassifier(max_depth=3)  # 基分类器
        ada = AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=3,
            learning_rate=1.0,
            algorithm="SAMME",
        )

        start = time.time()
        ada = ada.fit(X, y)
        score_plain = ada.score(X, y)
        end = time.time()
        print(f"Running time in SKlearn: {end - start:.2f}s")

        # mark these data to be protected in SPU
        X_spu, y_spu = emulator.seal(X, y)

        # run
        dtc = sml_dtc("gini", "best", 3, 3)
        proc = proc_wrapper(
            estimator=dtc,
            n_estimators=3,
            learning_rate=1.0,
            algorithm="discrete",
            epsilon=1e-5,
        )
        start = time.time()

        result = emulator.run(proc)(X_spu, y_spu)
        end = time.time()
        score_encrpted = jnp.mean((result == y))
        print(f"Running time in SPU: {end - start:.2f}s")

        # print acc
        print(f"Accuracy in SKlearn: {score_plain:.2f}")
        print(f"Accuracy in SPU: {score_encrpted:.2f}")

    finally:
        emulator.down()


if __name__ == "__main__":
    emul_ada(emulation.Mode.MULTIPROCESS)
