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

import jax.numpy as jnp
from jax import lax
import jax
import math
import random
random.seed(20)

from sml.tree.tree import DecisionTreeClassifier as sml_dtc

class RandomForestClassifier:
    """A random forest classifier based on DecisionTreeClassifier.
    
    Parameters
    ----------
    n_estimators : int
        The number of trees in the forest. Must specify an integer > 0.
    
    max_features : int, float, "auto", "sqrt", "log2", or None.
        The number of features to consider when looking for the best split.
        If it's an integer, must 0 < integer < n_features.
        If it's an float, must 0 < float <= 1.
        
    criterion : {"gini"}, default="gini"
         The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity.
    
    splitter : {"best"}, default="best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split.
        
    max_depth : int
        The maximum depth of the tree. Must specify an integer > 0.
        
    bootstrap : bool
        Whether bootstrap samples are used when building trees.
        
    max_samples : int, float ,None, default=None
        The number of samples to draw from X to train each base estimator.
        This parameter is only valid if bootstrap is ture.
        If it's an integer, must 0 < integer < n_samples.
        If it's an float, must 0 < float <= 1.
    
    n_labels: int
        The max number of labels.
    
    """

    def __init__(
        self,
        n_estimators,
        max_features,
        criterion,
        splitter,
        max_depth,
        bootstrap,
        max_samples,
        n_labels,
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

        self.n_estimators = n_estimators
        self.max_features = max_features
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.max_samples = max_samples
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

        # 实现bootstrap
        population = range(n_samples)
        indices = random.sample(population, max_samples)

        indices = jnp.array(indices)
        return X[indices], y[indices]
    
    def _select_features(self, n, k):
        indices = range(n)
        selected_elements = random.sample(indices, k)
        return selected_elements

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.n_features = n_features
        x = random.random()

        # 取消__init__中n_features参数，这里计算获得max_features
        if isinstance(self.max_features, int):
            assert (
                0 < self.max_features <= self.n_features
            ), "0 < max_features <= n_features when it's an integer"

        elif isinstance(self.max_features, float):
            assert (
                0 < self.max_features <= 1
            ), "max_features should be in the range (0, 1] when it's a float"
            self.max_features = (int)(self.max_features * self.n_features)

        elif isinstance(self.max_features, str):
            if self.max_features == 'sqrt':
                self.max_features = (int)(math.sqrt(self.n_features))

            elif self.max_features == 'log2':
                self.max_features = (int)(math.log2(self.n_features))

            else:
                self.max_features = self.n_features
        else:
            self.max_features = self.n_features
        
        self.trees = []
        self.features_indices = []

        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            features = self._select_features(self.n_features, self.max_features)

            tree = sml_dtc(self.criterion, self.splitter, self.max_depth, self.n_labels)
            tree.fit(X_sample[:, features], y_sample)
            self.trees.append(tree)
            self.features_indices.append(features)

        return self

    def predict(self, X):
        predictions_list = []
        for i, tree in enumerate(self.trees):
            features = self.features_indices[i]
            predictions = tree.predict(X[:, features])
            predictions_list.append(predictions)
        
        tree_predictions = jnp.array(predictions_list).T
            
        y_pred, _ = jax_mode_row(tree_predictions)

        return y_pred.ravel()


def jax_mode_row(data):
    # 获取每行的众数

    num_rows, num_cols = data.shape

    modes_list = []
    counts_list = []

    for row in range(num_rows):
        row_data = data[row, :]
        unique_values, value_counts = jnp.unique(
            row_data, return_counts=True, size=row_data.shape[0]
        )
        max_count_idx = jnp.argmax(value_counts)
        modes_list.append(unique_values[max_count_idx])
        counts_list.append(value_counts[max_count_idx])

    modes = jnp.array(modes_list, dtype=data.dtype)
    counts = jnp.array(counts_list, dtype=jnp.int32)

    return modes, counts
