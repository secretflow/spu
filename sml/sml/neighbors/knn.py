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
import jax.numpy as jnp
from jax import lax, vmap

# name -> function
metic_func_map = {}


class KNNClassifer:
    def __init__(
        self,
        n_neighbors=5,
        weights='uniform',
        n_classes=None,
        metric=None,
        metric_params={},
    ):
        """
        K-Nearest Neighbors (KNN) Classifier.

        Parameters:
        - n_neighbors (int): Number of neighbors to use for prediction.
        - weights (str): Weight function used in prediction. Possible values:
            * 'uniform': Uniform weights. All points in each neighborhood are weighted equally.
            * 'distance': Weight points by the inverse of their distance.
        - metric (str, optional): A function that computes distance between two points.
            It should take two arguments and return a scalar. Defaults to the Minkowski metric
            with p=2 (i.e., Euclidean distance). If None, the Minkowski metric with p=2 is used.
        - metric_params (dict, optional): Additional keyword arguments for the metric function.
        - n_classes (int, optional): Number of classes in the dataset. It needs to be provided
            for computation.

        Note:
        If the default metric (Minkowski with p=2) is used and metric_params is None,
        it will default to {'ord': 2}.
        """

        # Validate n_neighbors
        if not isinstance(n_neighbors, int) or n_neighbors <= 0:
            raise ValueError("n_neighbors should be a positive integer.")

        # Validate weights
        if weights not in ['uniform', 'distance']:
            raise ValueError("weights should be either 'uniform' or 'distance'.")

        # Validate n_classes
        if (n_classes is None) or (not isinstance(n_classes, int)) or (n_classes <= 0):
            raise ValueError("n_classes should be a positive integer if provided.")

        self.X = None
        self.y = None
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.n_classes = n_classes
        self.metric = metric

        # If metric is None, default to Minkowski with p=2
        if metric is None:
            self.metric_func = jnp.linalg.norm
            if metric_params is None:
                metric_params = {'ord': 2}
        else:
            # Check if the metric is callable
            if metric not in metic_func_map:
                raise ValueError("The metric should be a valid function name.")
            if not callable(metic_func_map[metric]):
                raise ValueError("The metric should be a callable function.")
            self.metric_func = metic_func_map[metric]

        self.metric_params = metric_params

    def tree_flatten(self):
        static_data = (
            self.n_neighbors,
            self.weights,
            self.metric,
            self.metric_params,
            self.n_classes,
        )
        dynamic_data = (self.X, self.y)
        return dynamic_data, static_data

    @classmethod
    def tree_unflatten(cls, static_data, dynamic_data):
        n_neighbors, weights, metric, metric_params, n_classes = static_data
        X, y = dynamic_data
        ins = cls(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            metric_params=metric_params,
            n_classes=n_classes,
        )
        ins.X = X
        ins.y = y
        return ins

    def fit(self, X, y):
        """
        Fit the KNN classifier.

        Parameters:
        - X (array-like): Training data.
        - y (array-like): Target values.

        Note:
        The y here needs to be converted to an integer starting from 0 before passing in, otherwise may occur unexpected errors.
        """
        self.X = X
        self.y = jnp.array(y)
        return self

    def _compute_weights(self, distances):
        """
        Compute the weights based on the distances and the weight scheme.

        Parameters:
        - distances (array-like): Distances to the neighbors.

        Returns:
        - array-like: Weights for each neighbor.
        """
        if self.weights == 'uniform':
            return jnp.ones_like(distances)
        elif self.weights == 'distance':
            return 1.0 / (distances + 1e-5)
        else:
            raise ValueError(
                "Invalid weight setting. Only 'uniform' and 'distance' are supported."
            )

    def _predict_single_sample(self, x):
        """
        Predict the class label for a single sample.

        Parameters:
        - x (array-like): A single sample.

        Returns:
        - int: Predicted class label.
        """
        distances = self.metric_func(self.X - x, **self.metric_params, axis=1)
        sorted_indices = jnp.argsort(distances)
        k_indices = sorted_indices[: self.n_neighbors]
        k_distances = distances[k_indices]
        k_labels = self.y[k_indices]

        def predict_uniform(_):
            counts = jnp.bincount(k_labels, length=int(self.n_classes))
            prediction = jnp.argmax(counts)
            return prediction

        def predict_distance(_):
            weights_inner = self._compute_weights(k_distances)
            weighted_counts = jnp.bincount(
                k_labels, weights=weights_inner, length=int(self.n_classes)
            )
            prediction = jnp.argmax(weighted_counts)
            return prediction

        return lax.cond(
            self.weights == 'uniform', predict_uniform, predict_distance, operand=None
        )

    def predict(self, X):
        """
        Predict the class labels for the provided data.

        Parameters:
        - X (array-like): Samples to predict.

        Returns:
        - array-like: Predicted class labels for each sample.
        """
        # Use the vectorized function to predict all samples at once
        predictions = vmap(self._predict_single_sample)(X)

        return predictions


jax.tree_util.register_pytree_node_class(KNNClassifer)
