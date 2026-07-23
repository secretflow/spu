# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import unittest
from collections import defaultdict

import jax.numpy as jnp
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as spsim

# Add the sml directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from sml.neighbors.knn import KNNClassifer


class UnitTests(unittest.TestCase):
    def test_knn(self):
        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64
        )

        # Test fit_predict
        def proc_predict(
            X_train, y_train, X_test, n_classes, n_neighbors=5, weights='uniform'
        ):
            knn_model = KNNClassifer(
                n_neighbors=n_neighbors, weights=weights, n_classes=n_classes
            )
            knn_model.fit(X_train, y_train)

            return knn_model.predict(X_test)

        # 假设有一组样本和对应的标签
        X_train = jnp.array([[1, 2], [2, 3], [3, 4], [5, 1]])
        y_train = jnp.array([-1, 3, 3, -1])

        # 假设有一组测试样本
        X_test = jnp.array([[4, 2], [1, 3]])

        # 获取样本的类别数
        n_classes = len(set(y_train.tolist()))

        # 将y_train映射为从0开始的连续数组
        label_to_int = defaultdict(lambda: len(label_to_int))
        y_train_new = jnp.array([label_to_int[label] for label in y_train.tolist()])

        # 运行模拟器
        
        copts = spu_pb2.CompilerOptions()

        proc_predict.__name__ = f"test_knn_uniform"
        spu_fn_uniform = spsim.sim_jax(sim, proc_predict, static_argnums=(3, 4, 5), copts=copts, pphlo_ref=(None, None)) #test_knn_uniform
        result_uniform = spu_fn_uniform(
            X_train, y_train_new, X_test, n_classes, 3, 'uniform'
        )
        
        copts = spu_pb2.CompilerOptions()
        
        proc_predict.__name__ = f"test_knn_distance"
        spu_fn_distance = spsim.sim_jax(sim, proc_predict, static_argnums=(3, 4, 5), copts=copts, pphlo_ref=(None, None)) #test_knn_distance
        result_distance = spu_fn_distance(
            X_train, y_train_new, X_test, n_classes, 3, 'distance'
        )

        # 再从连续数组映射回原来的标签
        int_to_label = {i: label for label, i in label_to_int.items()}

        # 与sklearn的结果进行比较
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        skip_uniform = False
        try:
            if result_uniform == "skipped":
                skip_uniform = True
        except:
            pass
        if not skip_uniform:
            print(spu_fn_uniform.pphlo)
            result_uniform_np = np.array(result_uniform)
            predictions_uniform = [
                int_to_label[prediction] for prediction in result_uniform_np
            ]
            neigh_uni = KNeighborsClassifier(n_neighbors=3, weights='uniform')
            neigh_uni.fit(X_train, y_train)
            sklearn_predictions = neigh_uni.predict(X_test)

            self.assertEqual(predictions_uniform, sklearn_predictions.tolist())

        skip_distance = False
        try:
            if result_distance == "skipped":
                skip_distance = True
        except:
            pass
        if not skip_distance:
            print(spu_fn_distance.pphlo)
            result_distance_np = np.array(result_distance)
            predictions_distance = [
                int_to_label[prediction] for prediction in result_distance_np
            ]

            neigh_dis = KNeighborsClassifier(n_neighbors=3, weights='distance')
            neigh_dis.fit(X_train, y_train)

            sklearn_predictions = neigh_dis.predict(X_test)

            self.assertEqual(predictions_distance, sklearn_predictions.tolist())


if __name__ == "__main__":
    unittest.main()
