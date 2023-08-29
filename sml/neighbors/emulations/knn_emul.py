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
from collections import defaultdict

import jax.numpy as jnp
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# from sklearn.metrics import roc_auc_score, explained_variance_score

# Add the library directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

import sml.utils.emulation as emulation
from sml.neighbors.knn import KNNClassifer


# TODO: design the enumation framework, just like py.unittest
# all emulation action should begin with `emul_` (for reflection)
def emul_KNN(mode: emulation.Mode.MULTIPROCESS):
    def proc_KNN(X_train, y_train, X_test, n_classes, n_neighbors=5, weights='uniform'):
        knn_model = KNNClassifer(
            n_neighbors=n_neighbors, weights=weights, n_classes=n_classes
        )
        knn_model.fit(X_train, y_train)

        return knn_model.predict(X_test)

    try:
        # bandwidth and latency only work for docker mode
        emulator = emulation.Emulator(
            emulation.CLUSTER_ABY3_3PC, mode, bandwidth=300, latency=20
        )
        emulator.up()
        # 假设有一组样本和对应的标签
        X_train = jnp.array([[1, 2], [2, 3], [3, 4], [5, 1]])
        y_train = jnp.array([0, 3, 3, 0])

        # 假设有一组测试样本
        X_test = jnp.array([[4, 2], [1, 3]])

        # 获取样本的类别数
        n_classes = len(set(y_train.tolist()))
        n_neighbors = 3

        # 将y_train映射为从0开始的连续数组
        label_to_int = defaultdict(lambda: len(label_to_int))
        y_train_new = jnp.array([label_to_int[label] for label in y_train.tolist()])

        (
            X_train_,
            y_train_new_,
            X_test_,
        ) = emulator.seal(X_train, y_train_new, X_test)

        result_uniform = emulator.run(proc_KNN, static_argnums=(3, 4, 5))(
            X_train_, y_train_new_, X_test_, n_classes, n_neighbors, 'uniform'
        )

        result_distance = emulator.run(proc_KNN, static_argnums=(3, 4, 5))(
            X_train_, y_train_new_, X_test_, n_classes, n_neighbors, 'distance'
        )

        # 再从连续数组映射回原来的标签
        int_to_label = {i: label for label, i in label_to_int.items()}
        result_uniform_np = np.array(result_uniform)
        result_distance_np = np.array(result_distance)
        predictions_uniform = [
            int_to_label[prediction] for prediction in result_uniform_np
        ]
        predictions_distance = [
            int_to_label[prediction] for prediction in result_distance_np
        ]

        # 与sklearn的结果进行比较
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)

        neigh_uni = KNeighborsClassifier(n_neighbors=3, weights='uniform')
        neigh_uni.fit(X_train, y_train)

        sklearn_predictions = neigh_uni.predict(X_test)

        assert np.array_equal(predictions_uniform, sklearn_predictions)

        neigh_dis = KNeighborsClassifier(n_neighbors=3, weights='distance')
        neigh_dis.fit(X_train, y_train)

        sklearn_predictions = neigh_dis.predict(X_test)

        assert np.array_equal(predictions_distance, sklearn_predictions)

    finally:
        emulator.down()


if __name__ == "__main__":
    emul_KNN(emulation.Mode.MULTIPROCESS)
