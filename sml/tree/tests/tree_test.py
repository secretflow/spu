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


import unittest

import jax.numpy as jnp
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

import spu.spu_pb2 as spu_pb2  # type: ignore
import spu.utils.simulation as spsim
from sml.tree.tree import DecisionTreeClassifier as sml_dtc

MAX_DEPTH = 3


class UnitTests(unittest.TestCase):
    def test_tree(self):
        def proc_wrapper(max_depth=2, n_labels=3):
            dt = sml_dtc("gini", "best", max_depth, n_labels)

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
            new_features = jnp.greater_equal(
                iris_data[:, :], new_threshold[:, jnp.newaxis, :]
            )
            new_features = new_features.transpose([1, 0, 2]).reshape(n_samples, -1)

            X, y = new_features[:, ::3], iris_label[:]
            return X, y

        # bandwidth and latency only work for docker mode
        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64
        )

        # load mock data
        X, y = load_data()
        n_samples = y.shape[0]
        n_labels = jnp.unique(y).shape[0]

        # compare with sklearn
        clf = DecisionTreeClassifier(
            max_depth=MAX_DEPTH, criterion='gini', splitter='best', random_state=None
        )
        clf = clf.fit(X, y)
        score_plain = clf.score(X, y)

        # run
        proc = proc_wrapper(MAX_DEPTH, n_labels)
        result = spsim.sim_jax(sim, proc)(X, y)
        score_encrpted = jnp.sum((result == y)) / n_samples

        # print acc
        print(f"Accuracy in SKlearn: {score_plain:.2f}")
        print(f"Accuracy in SPU: {score_encrpted:.2f}")


if __name__ == "__main__":
    unittest.main()
