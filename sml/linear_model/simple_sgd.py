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

import numpy as np
import jax.numpy as jnp

from enum import Enum

from ..utils.fxp_approx import sigmoid_sr


class RegType(Enum):
    Linear = 'linear'
    Logistic = 'logistic'


class Penalty(Enum):
    NONE = 'None'
    L1 = 'l1'  # not supported
    L2 = 'l2'


class SGDClassifier:
    def __init__(
        self,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        reg_type: str,
        penalty: str,
        l2_norm: float,
    ):
        # parameter check.
        assert epochs > 0, f"epochs should >0"
        assert learning_rate > 0, f"learning_rate should >0"
        assert batch_size > 0, f"batch_size should >0"
        assert penalty != 'l1', "not support L1 penalty for now"
        if penalty == Penalty.L2:
            assert l2_norm > 0, f"l2_norm should >0 if use L2 penalty"
        assert reg_type in [
            e.value for e in RegType
        ], f"reg_type should in {[e.value for e in RegType]}, but got {reg_type}"
        assert penalty in [
            e.value for e in Penalty
        ], f"penalty should in {[e.value for e in Penalty]}, but got {reg_type}"

        self._epochs = epochs
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._l2_norm = l2_norm
        self._penalty = Penalty(penalty)
        # TODO: reg_type should not be here.
        self._reg_type = RegType(reg_type)

        self._weights = jnp.zeros(())

    def _update_weights(
        self,
        x,  # array-like
        y,  # array-like
        w,  # array-like
        total_batch: int,
        batch_size: int,
    ) -> np.ndarray:
        assert x.shape[0] >= total_batch * batch_size, "total batch is too large"
        num_feat = x.shape[1]
        assert w.shape[0] == num_feat + 1, "w shape is mismatch to x"
        assert len(w.shape) == 1 or w.shape[1] == 1, "w should be list or 1D array"
        w = w.reshape((w.shape[0], 1))

        for idx in range(total_batch):
            begin = idx * batch_size
            end = (idx + 1) * batch_size
            # padding one col for bias in w
            x_slice = jnp.concatenate(
                (x[begin:end, :], jnp.ones((batch_size, 1))), axis=1
            )
            y_slice = y[begin:end, :]

            pred = jnp.matmul(x_slice, w)
            if self._reg_type == RegType.Logistic:
                pred = sigmoid_sr(pred)

            err = pred - y_slice
            grad = jnp.matmul(jnp.transpose(x_slice), err)

            if self._penalty == Penalty.L2:
                w_with_zero_bias = jnp.resize(w, (num_feat, 1))
                w_with_zero_bias = jnp.concatenate(
                    (w_with_zero_bias, jnp.zeros((1, 1))),
                    axis=0,
                )
                grad = grad + w_with_zero_bias * self._l2_norm

            step = (self._learning_rate * grad) / batch_size

            w = w - step

        return w

    def fit(self, x, y):
        """Fit linear model with Stochastic Gradient Descent.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Training data.

        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        assert len(x.shape) == 2, f"expect x to be 2 dimension array, got {x.shape}"

        num_sample = x.shape[0]
        num_feat = x.shape[1]
        batch_size = min(self._batch_size, num_sample)
        total_batch = int(num_sample / batch_size)

        weights = jnp.zeros((num_feat + 1, 1))

        # do train
        for _ in range(self._epochs):
            weights = self._update_weights(
                x,
                y,
                weights,
                total_batch,
                batch_size,
            )

        self._weights = weights
        return self

    def predict_proba(self, x):
        """Probability estimates.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Input data for prediction.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in `self.classes_`.
        """
        num_feat = x.shape[1]
        w = self._weights
        assert w.shape[0] == num_feat + 1, f"w shape is mismatch to x={x.shape}"
        assert len(w.shape) == 1 or w.shape[1] == 1, "w should be list or 1D array"
        w.reshape((w.shape[0], 1))

        bias = w[-1, 0]
        w = jnp.resize(w, (num_feat, 1))

        pred = jnp.matmul(x, w) + bias

        if self._reg_type == RegType.Logistic:
            pred = sigmoid_sr(pred)
        return pred
