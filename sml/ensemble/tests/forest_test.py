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
from sklearn.ensemble import RandomForestClassifier

import spu.spu_pb2 as spu_pb2  # type: ignore
import spu.utils.simulation as spsim
from sml.ensemble.forest import RandomForestClassifier as sml_rfc

MAX_DEPTH = 3


class UnitTests(unittest.TestCase):
    def test_forest(self):
        def proc_wrapper(
            n_estimators,
            max_features,
            criterion,
            splitter,
            max_depth,
            bootstrap,
            max_samples,
            n_labels,
        ):
            rf_custom = sml_rfc(
                n_estimators,
                max_features,
                criterion,
                splitter,
                max_depth,
                bootstrap,
                max_samples,
                n_labels,
            )

            def proc(X, y):
                rf_custom_fit = rf_custom.fit(X, y)

                result = rf_custom_fit.predict(X)
                return result

            return proc

        def load_data():
            iris = load_iris()
            iris_data, iris_label = jnp.array(iris.data), jnp.array(iris.target)
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
        n_labels = jnp.unique(y).shape[0]

        # compare with sklearn
        rf = RandomForestClassifier(
            n_estimators=3,
            max_features="log2",
            criterion='gini',
            max_depth=MAX_DEPTH,
            bootstrap=True,
            max_samples=0.7,
        )
        rf = rf.fit(X, y)
        score_plain = rf.score(X, y)
        # 获取每棵树的预测值
        tree_predictions = jnp.array([tree.predict(X) for tree in rf.estimators_])
  
        # run
        proc = proc_wrapper(
            n_estimators=3,
            max_features="log2",
            criterion='gini',
            splitter='best',
            max_depth=3,
            bootstrap=True,
            max_samples=0.7,
            n_labels=n_labels,
        )

        result = spsim.sim_jax(sim, proc)(X, y)

        score_encrpted = jnp.mean((result == y))

        # print acc
        print(f"Accuracy in SKlearn: {score_plain}")
        print(f"Accuracy in SPU: {score_encrpted}")


if __name__ == "__main__":
    unittest.main()

