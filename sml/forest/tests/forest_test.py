'''
Author: Li Zhihang
Date: 2024-06-16 12:03:08
LastEditTime: 2024-06-22 16:45:43
FilePath: /klaus/spu-klaus/sml/forest/tests/forest_test.py
Description:正确率相差太大：Accuracy in SKlearn: 0.95；Accuracy in SPU: 0.67
'''

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
from sml.forest.forest import RandomForestClassifier as sml_rfc

MAX_DEPTH = 3


class UnitTests(unittest.TestCase):
    def test_forest(self):
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

        # bandwidth and latency only work for docker mode
        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64
        )

        # load mock data
        X, y = load_data()
        n_samples, n_features = X.shape
        n_labels = jnp.unique(y).shape[0]
        print(y)

        # compare with sklearn
        rf = RandomForestClassifier(
            n_estimators=3,
            max_features='log2',
            criterion='gini',
            max_depth=MAX_DEPTH,
            bootstrap=True,
            max_samples=0.7,
        )
        rf = rf.fit(X, y)
        score_plain = rf.score(X, y)
        # 获取每棵树的预测值
        tree_predictions = jnp.array([tree.predict(X) for tree in rf.estimators_])
        # print("sklearn:")
        # print(tree_predictions)
        print(n_features)
        # run
        proc = proc_wrapper(
            n_estimators=3,
            max_features='log2',
            n_features=n_features,
            criterion='gini',
            splitter='best',
            max_depth=3,
            bootstrap=True,
            max_samples=0.7,
            n_labels=n_labels,
            seed=0,
        )
        # 不可以使用bootstrap，否则在spu运行的正确率很低
        result = spsim.sim_jax(sim, proc)(X, y)

        # print(y_sample)
        score_encrpted = jnp.sum((result == y)) / n_samples

        # print acc
        print(f"Accuracy in SKlearn: {score_plain}")
        print(f"Accuracy in SPU: {score_encrpted}")


if __name__ == "__main__":
    unittest.main()
