# Copyright 2024 Ant Group Co., Ltd.
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

# 不支持early_stop

import copy
import warnings

import jax.numpy as jnp
from jax import lax

from sml.tree.tree import DecisionTreeClassifier as sml_dtc


class AdaBoostClassifier:
    """A adaboost classifier based on DecisionTreeClassifier.

    Parameters
    ----------
    estimator : {"dtc"}, default="dtc"
        Specifies the type of model or algorithm to be used for training.
        Supported estimators are "dtc".

    n_estimators : int
        The number of estimators. Must specify an integer > 0.

    learning_rate : float
        The step size used to update the model weights during training.
        It's an float, must learning_rate > 0.

    algorithm : str (default='discrete')
        The boosting algorithm to use. Only the SAMME discrete algorithm is used in this implementation.
        In scikit-learn, the Real Boosting Algorithm (SAMME.R) will be deprecated.

    epsilon : float (default=1e-5)
        A small positive value used in calculations to avoid division by zero and other numerical issues.
        Must be greater than 0 and less than 0.1.

    """

    def __init__(
        self,
        estimator,
        n_estimators,
        learning_rate,
        algorithm,
        epsilon=1e-5,
    ):
        assert isinstance(
            estimator, sml_dtc
        ), "Estimator other than sml_dtc is not supported."
        assert (
            n_estimators is not None and n_estimators > 0
        ), "n_estimators should not be None and must > 0."
        assert algorithm == "discrete", (
            "Only support SAMME discrete algorithm. "
            "In scikit-learn, the Real Boosting Algorithm (SAMME.R) will be deprecated. "
            "You can refer to the official documentation for more details: "
            "https://github.com/scikit-learn/scikit-learn/issues/26784"
        )
        assert epsilon > 0 and epsilon < 0.1, "epsilon must be > 0 and < 0.1."

        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.epsilon = epsilon

        self.n_classes = estimator.n_labels

        self.estimators_ = []
        self.estimator_weight_ = jnp.zeros(self.n_estimators, dtype=jnp.float32)
        self.estimator_errors_ = jnp.ones(self.n_estimators, dtype=jnp.float32)
        self.estimator_flags_ = jnp.zeros(self.n_estimators, dtype=jnp.bool_)
        self.early_stop = False  # 添加 early_stop 标志

    def _num_samples(self, x):
        """返回x中的样本数量."""
        if hasattr(x, 'fit'):
            # 检查是否是一个estimator
            raise TypeError('Expected sequence or array-like, got estimator')
        if (
            not hasattr(x, '__len__')
            and not hasattr(x, 'shape')
            and not hasattr(x, '__array__')
        ):
            raise TypeError("Expected sequence or array-like, got %s" % type(x))

        if hasattr(x, 'shape'):
            if len(x.shape) == 0:  # scalar
                raise TypeError(
                    "Singleton array %r cannot be considered a valid collection." % x
                )
            return x.shape[0]
        else:
            return len(x)

    def _check_sample_weight(self, sample_weight, X):
        '''
        Description: Validate and process sample weights.

        Parameters:
        - sample_weight: Can be None, a scalar (int or float), or a 1D array-like.
        - X: Input data from which to determine the number of samples.

        Returns:
        - sample_weight: A 1D array of sample weights, one for each sample in X.

        Sample weight scenarios:
        1. None:
           - If sample_weight is None, it will be initialized to an array of ones,
             meaning all samples are equally weighted.
        2. Scalar (int or float):
           - If sample_weight is a scalar, it will be converted to an array where
             each sample's weight is equal to the scalar value.
        3. Array-like:
           - If sample_weight is an array or array-like, it will be converted to a JAX array.
           - The array must be 1D and its length must match the number of samples.
           - If these conditions are not met, an error will be raised.
        '''
        n_samples = self._num_samples(X)

        if sample_weight is None:
            sample_weight = jnp.ones(n_samples, dtype=jnp.float32)
        elif isinstance(sample_weight, (jnp.int32, jnp.float32)):
            sample_weight = jnp.full(n_samples, sample_weight, dtype=jnp.float32)
        else:
            sample_weight = jnp.asarray(sample_weight, dtype=jnp.float32)
            if sample_weight.ndim != 1:
                raise ValueError("Sample weight must be 1D array or scalar")

            if sample_weight.shape[0] != n_samples:
                raise ValueError(
                    "sample_weight.shape == {}, expected {}!".format(
                        sample_weight.shape, (n_samples,)
                    )
                )

        return sample_weight

    def fit(self, X, y, sample_weight=None):
        sample_weight = self._check_sample_weight(
            sample_weight,
            X,
        )
        sample_weight /= sample_weight.sum()

        self.classes = y

        epsilon = self.epsilon

        for iboost in range(self.n_estimators):
            sample_weight = jnp.clip(sample_weight, a_min=epsilon, a_max=None)

            estimator = copy.deepcopy(self.estimator)
            sample_weight, estimator_weight, estimator_error, flag = (
                self._boost_discrete(
                    iboost,
                    X,
                    y,
                    sample_weight,
                    estimator,
                )
            )

            self.estimator_weight_ = self.estimator_weight_.at[iboost].set(
                estimator_weight
            )
            self.estimator_errors_ = self.estimator_errors_.at[iboost].set(
                estimator_error
            )
            self.estimator_flags_ = self.estimator_flags_.at[iboost].set(flag)

            sample_weight_sum = jnp.sum(sample_weight)
            if iboost < self.n_estimators - 1:
                sample_weight /= sample_weight_sum

        return self

    def _boost_discrete(self, iboost, X, y, sample_weight, estimator):
        """Implement a single boost using the SAMME discrete algorithm."""
        self.estimators_.append(estimator)

        n_classes = self.n_classes
        epsilon = self.epsilon

        estimator.fit(X, y, sample_weight=sample_weight)

        y_predict = estimator.predict(X)

        incorrect = y_predict != y
        estimator_error = jnp.mean(
            jnp.average(incorrect, weights=sample_weight, axis=0)
        )
        is_small_error = estimator_error <= epsilon

        self.early_stop = jnp.logical_or(self.early_stop, is_small_error)

        def true_0_fun(sample_weight):
            return sample_weight, 1.0, 0.0, jnp.array(False, dtype=jnp.bool_)

        def false_0_fun(sample_weight, estimator_error, incorrect, n_classes):
            flag = estimator_error < 1.0 - (1.0 / n_classes)
            flag = jnp.where(self.early_stop, jnp.array(False, dtype=jnp.bool_), flag)

            estimator_weight = self.learning_rate * (
                jnp.log((1.0 - estimator_error) / estimator_error)
                + jnp.log(n_classes - 1.0)
            )
            sample_weight_updated = sample_weight * jnp.exp(
                estimator_weight * incorrect
            )

            sample_weight = jnp.where(flag, sample_weight_updated, sample_weight)
            estimator_weight = jnp.where(flag, estimator_weight, 0.0)

            return sample_weight, estimator_weight, estimator_error, flag

        sample_weight_true, estimator_weight_true, estimator_error_true, flag_true = (
            true_0_fun(sample_weight)
        )
        (
            sample_weight_false,
            estimator_weight_false,
            estimator_error_false,
            flag_false,
        ) = false_0_fun(sample_weight, estimator_error, incorrect, n_classes)

        sample_weight = jnp.where(
            is_small_error, sample_weight_true, sample_weight_false
        )
        estimator_weight = jnp.where(
            is_small_error, estimator_weight_true, estimator_weight_false
        )
        estimator_error = jnp.where(
            is_small_error, estimator_error_true, estimator_error_false
        )
        flag = jnp.where(is_small_error, flag_true, flag_false)

        return sample_weight, estimator_weight, estimator_error, flag

    def predict(self, X):
        pred = self.decision_function(X)

        if self.n_classes == 2:
            return self.classes.take(pred > 0, axis=0)

        return self.classes.take(jnp.argmax(pred, axis=1), axis=0)

    def decision_function(self, X):
        n_classes = self.n_classes
        classes = self.classes[:, jnp.newaxis]

        pred = sum(
            jnp.where(
                (estimator.predict(X) == classes).T,
                w,
                -1 / (n_classes - 1) * w,
            )
            * flag
            for estimator, w, flag in zip(
                self.estimators_, self.estimator_weight_, self.estimator_flags_
            )
        )

        weights_flags = self.estimator_weight_ * self.estimator_flags_
        pred /= jnp.sum(weights_flags)

        if n_classes == 2:
            pred[:, 0] *= -1
            return pred.sum(axis=1)
        return pred
