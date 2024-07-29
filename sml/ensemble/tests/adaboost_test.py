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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


import spu.spu_pb2 as spu_pb2  # type: ignore
import spu.utils.simulation as spsim
from sml.ensemble.adaboost import AdaBoostClassifier as sml_Adaboost

MAX_DEPTH = 3

class UnitTests(unittest.TestCase):
    def test_Ada(self):
        def proc_wrapper(
            estimator = "dtc",
            n_estimators = 10,
            max_depth = MAX_DEPTH,
            learning_rate = 1.0,
            n_classes = 3,
        ):
            ada_custom = sml_Adaboost(
                estimator = "dtc",
                n_estimators = 10,
                max_depth = MAX_DEPTH,
                learning_rate = 1.0,
                n_classes = 3,
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

        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64
        )
        
        
        X, y = load_data()
        n_samples, n_features = X.shape
        n_labels = jnp.unique(y).shape[0]
        
        # compare with sklearn
        base_estimator = DecisionTreeClassifier(max_depth=3)  # 基分类器
        ada = AdaBoostClassifier(estimator=base_estimator, n_estimators=3, learning_rate=1.0, algorithm="SAMME")
        ada = ada.fit(X, y)
        score_plain = ada.score(X, y)
        
        #run
        proc = proc_wrapper(
            estimator = "dtc",
            n_estimators = 3,
            max_depth = 3,
            learning_rate = 1.0,
            n_classes = 3,
        )
        
        result = spsim.sim_jax(sim, proc)(X, y)
        print(result)
        score_encrypted = jnp.mean(result == y)
        
        # print acc
        print(f"Accuracy in SKlearn: {score_plain}")
        print(f"Accuracy in SPU: {score_encrypted}")

if __name__ == '__main__':
    unittest.main()


        
