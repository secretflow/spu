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

import jax
from jax import jit, random
from functools import partial
import jax.numpy as jnp
import numpy as np
from jax import vmap, lax
from collections import defaultdict

class KNN:
    def __init__(self, metric=None, params=None):
        self.X = None
        self.y = None
        # TODO: add support for other metrics
        self.metric = metric if metric is not None else jnp.linalg.norm
        # TODO: add support for other parameters
        self.params = params if params is not None else {}

    def fit(self, X, y):
        self.X = X
        self.y = jnp.array(y)

    def _compute_weights(self, distances, weights):
        if weights == 'uniform':
            return jnp.ones_like(distances)
        elif weights == 'distance':
            return 1.0 / distances
        else:
            raise ValueError("Invalid weight setting. Only 'uniform' and 'distance' are supported.")

    def _predict_single_sample(self, x, n_neighbors, weights, n_classes):
        distances = self.metric(self.X - x, **self.params, axis=1)
        sorted_indices = jnp.argsort(distances)
        k_indices = sorted_indices[:n_neighbors]
        # indices = jnp.arange(n_neighbors)
        # k_indices = jnp.take(sorted_indices, indices)
        k_distances = distances[k_indices]
        k_labels = self.y[k_indices]

        def predict_uniform(_):
            counts = jnp.bincount(k_labels, length=int(n_classes))
            prediction = jnp.argmax(counts)
            return prediction

        def predict_distance(_):
            weights_inner = self._compute_weights(k_distances, weights)
            weighted_counts = jnp.bincount(k_labels, weights=weights_inner, length=int(n_classes))
            prediction = jnp.argmax(weighted_counts)
            return prediction

        return lax.cond(weights == 'uniform',
                        predict_uniform,
                        predict_distance,
                        operand=None)

    def transform(self, X, n_neighbors, weights, n_classes):
        predictions = []
        for x in X:
            prediction = self._predict_single_sample(x, n_neighbors, weights, n_classes)
            predictions.append(prediction)
        return jnp.array(predictions)





