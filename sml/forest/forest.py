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
Author       : error: git config user.email & please set dead value or install git
Date         : 2024-06-15 20:01
LastEditors: Please set LastEditors
LastEditTime: 2024-06-18 19:09:17
FilePath: /klaus/spu-klaus/sml/forest/forest.py
Description  : 基本完成函数编写的工作，目前测试结果基本正确，后面需要完成emul和test
bootstrap有问题,bootstrap后predict不输出1,bootstrap无1(因为不支持jax.random的api)

!最终：bootstrap这个参数，不可用：在明文下bootstrap取样正确，但在forest_test.py时，无法取到标签1，
因为bootstrap不能用，目前没有开发max_samples这个超参数

基本完成，因为不支持jax.random的api，所以bootstrap和select_features都没有使用随机算法，
导致max_features为sqrt和log2的时候误差较大，而sklearn的select_features比较随机，因此误差大
如果不增加n_features这个参数的话，会导致创建动态数组，并对动态数组切分，导致程序报错
'''
import jax.numpy as jnp
from jax import lax

from sml.tree.tree import DecisionTreeClassifier as sml_dtc

# from functools import partial
# from jax import jit
# import jax.random as jdm


# key = jdm.PRNGKey(42)


class RandomForestClassifier:
    """A random forest classifier."""

    def __init__(
        self,
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
    ):
        assert criterion == "gini", "criteria other than gini is not supported."
        assert splitter == "best", "splitter other than best is not supported."
        assert (
            n_estimators is not None and n_estimators > 0
        ), "n_estimators should not be None and must > 0."
        assert (
            max_depth is not None and max_depth > 0
        ), "max_depth should not be None and must > 0."
        assert isinstance(
            bootstrap, bool
        ), "bootstrap should be a boolean value (True or False)"
        assert isinstance(n_features, int), "n_features should be an integer."
        if isinstance(max_features, int):
            assert (
                max_features <= n_features
            ), "max_features should not exceed n_features when it's an integer"
            max_features = jnp.array(n_features, dtype=int)
        elif isinstance(max_features, float):
            assert (
                0 < max_features <= 1
            ), "max_features should be in the range (0, 1] when it's a float"
            max_features = jnp.array((max_features * n_features), dtype=int)
        elif isinstance(max_features, str):
            if max_features == 'sqrt':
                max_features = jnp.array(jnp.sqrt(n_features), dtype=int)
            elif max_features == 'log2':
                max_features = jnp.array(jnp.log2(n_features), dtype=int)
            else:
                max_features = n_features
        else:
            max_features = n_features

        self.seed = seed
        # self.key = key

        self.n_estimators = n_estimators
        self.max_features = max_features
        self.n_features = n_features
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        # max_samples 这个参数在bootstrap为ture才有效，可以为float(0,1],int<总数，默认为none
        self.n_labels = n_labels

        self.trees = []
        self.features_indices = []

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        if isinstance(self.max_samples, int):
            assert (
                self.max_samples <= n_samples
            ), "max_samples should not exceed n_samples when it's an integer"
            max_samples = self.max_samples
        elif isinstance(self.max_samples, float):
            assert (
                0 < self.max_samples <= 1
            ), "max_samples should be in the range (0, 1] when it's a float"
            max_samples = (int)(self.max_samples * n_samples)
        else:
            max_samples = n_samples

        if not self.bootstrap:
            return X, y
        # 使用斐波那契数列变体生成伪随机索引
        indices = jnp.zeros(max_samples, dtype=int)
        a, b = self.seed % n_samples, (self.seed + 1) % n_samples
        for i in range(max_samples):
            indices = indices.at[i].set(b % n_samples)
            a, b = b, (a + b) % n_samples  # 生成斐波那契数列并取模
        # 更新种子值以保证每次调用生成不同的序列
        self.seed += 1
        return X[indices], y[indices]

    # 可以用，但没n选k
    def _select_features(self):
        # if isinstance(self.max_features, int):
        #     assert self.max_features <= n_features, "max_features should not exceed n_features when it's an integer"
        #     max_features = jnp.array(n_features, dtype=int)
        # elif isinstance(self.max_features, float):
        #     assert 0 < self.max_features <= 1, "max_features should be in the range (0, 1] when it's a float"
        #     max_features = jnp.array((self.max_features * n_features), dtype=int)
        # elif isinstance(self.max_features, str):
        #     if self.max_features == 'sqrt':
        #         max_features = jnp.array(jnp.sqrt(n_features), dtype=int)
        #     elif self.max_features == 'log2':
        #         max_features = jnp.array(jnp.log2(n_features), dtype=int)
        #     else:
        #         max_features = n_features
        # else:
        #     max_features = n_features

        selected_indices = self._shuffle_indices(self.n_features, self.max_features)[
            : self.max_features
        ]
        self.seed += 1
        return selected_indices

    def _shuffle_indices(self, n, k):
        # 基于种子的循环洗牌算法，确保不出现重复的索引
        rng = self.seed
        indices = jnp.arange(n)
        # indices = range(n)

        def cond_fun(state):
            i, _, _ = state
            return i < k

        def body_fun(state):
            i, rng, indices = state
            rng = (rng * 48271 + 1) % (2**31 - 1)
            j = i + rng % (n - i)
            # 交换元素以进行洗牌
            indices = self._swap(indices, i, j)
            return (i + 1, rng, indices)

        _, _, shuffled_indices = lax.while_loop(cond_fun, body_fun, (0, rng, indices))
        # selected_indices = shuffled_indices[:k]
        selected_indices = shuffled_indices
        return selected_indices

    def _swap(self, array, i, j):
        # 辅助函数：交换数组中的两个元素
        array = array.at[i].set(array[j])
        array = array.at[j].set(array[i])
        return array

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.trees = []
        self.features_indices = []

        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            features = self._select_features()
            # selected_indices = self._shuffle_indices(n_features)
            print(y_sample)
            tree = sml_dtc(self.criterion, self.splitter, self.max_depth, self.n_labels)
            tree.fit(X_sample[:, features], y_sample)
            self.trees.append(tree)
            self.features_indices.append(features)

        return self

    def predict(self, X):
        tree_predictions = jnp.zeros((X.shape[0], self.n_estimators))

        for i, tree in enumerate(self.trees):
            features = self.features_indices[i]
            print(features)
            tree_predictions = tree_predictions.at[:, i].set(
                tree.predict(X[:, features])
            )
            # print(tree_predictions[:, i])
        # Use majority vote to determine final prediction
        y_pred, _ = jax_mode_row(tree_predictions)
        # return tree_predictions
        return y_pred.ravel()


def jax_mode_row(data):
    # 获取每行的众数

    # 获取数据的形状
    num_rows, num_cols = data.shape

    # 初始化众数和计数的数组
    modes = jnp.zeros(num_rows, dtype=data.dtype)
    counts = jnp.zeros(num_rows, dtype=jnp.int32)

    # 计算每行的众数及其计数
    for row in range(num_rows):
        row_data = data[row, :]
        unique_values, value_counts = jnp.unique(
            row_data, return_counts=True, size=row_data.shape[0]
        )
        max_count_idx = jnp.argmax(value_counts)
        modes = modes.at[row].set(unique_values[max_count_idx])
        counts = counts.at[row].set(value_counts[max_count_idx])

    return modes, counts
