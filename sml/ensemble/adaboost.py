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
import jax.numpy as jnp
from jax import lax
import warnings
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
    
    """
    def __init__(
        self,
        estimator,
        n_estimators,
        learning_rate,
        algorithm,
    ):
        assert isinstance(estimator, sml_dtc), "Estimator other than sml_dtc is not supported."
        assert (
            n_estimators is not None and n_estimators > 0
        ), "n_estimators should not be None and must > 0."
        assert algorithm == "discrete", (
            "Only support SAMME discrete algorithm. "
            "In scikit-learn, the Real Boosting Algorithm (SAMME.R) will be deprecated. "
            "You can refer to the official documentation for more details: "
            "https://github.com/scikit-learn/scikit-learn/issues/26784"
        )

        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        
        self.n_classes = estimator.n_labels
        
        self.estimators_ = []
        self.estimator_weight = jnp.zeros(self.n_estimators, dtype=jnp.float32)
        self.estimator_errors = jnp.ones(self.n_estimators, dtype=jnp.float32)
        self.estimator_flags_ = []
        self.early_stop = False  # 添加 early_stop 标志
    
    def _num_samples(self, x):
        """返回x中的样本数量."""
        if hasattr(x, 'fit'):
            # 检查是否是一个estimator
            raise TypeError('Expected sequence or array-like, got estimator')
        if not hasattr(x, '__len__') and not hasattr(x, 'shape') and not hasattr(x, '__array__'):
            raise TypeError("Expected sequence or array-like, got %s" % type(x))
        
        if hasattr(x, 'shape'):
            if len(x.shape) == 0:  # scalar
                raise TypeError("Singleton array %r cannot be considered a valid collection." % x)
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
            sample_weight, X,
        )
        sample_weight /= sample_weight.sum()

        self.classes = y
        
        epsilon = jnp.finfo(sample_weight.dtype).eps
               
        self.estimator_weight_ = jnp.zeros(self.n_estimators, dtype=jnp.float32)
        self.estimator_errors_ = jnp.ones(self.n_estimators, dtype=jnp.float32)
        
        for iboost in range(self.n_estimators):
            sample_weight = jnp.clip(sample_weight, a_min=epsilon, a_max=None)

            estimator = copy.deepcopy(self.estimator)
            sample_weight, estimator_weight, estimator_error = self._boost_discrete(
                iboost, X, y, sample_weight, estimator,
            )

            self.estimator_weight_ = self.estimator_weight_.at[iboost].set(estimator_weight)
            self.estimator_errors_ = self.estimator_errors_.at[iboost].set(estimator_error)

            sample_weight_sum = jnp.sum(sample_weight)
            if iboost < self.n_estimators - 1:
                sample_weight /= sample_weight_sum
 
        return self
    
    def _boost_discrete(self, iboost, X, y, sample_weight, estimator):
        """Implement a single boost using the SAMME discrete algorithm."""
        self.estimators_.append(estimator)
         
        n_classes = self.n_classes
        
        estimator.fit(X, y, sample_weight=sample_weight)
        
        y_predict = estimator.predict(X)
        
        incorrect = y_predict != y
        estimator_error = jnp.mean(jnp.average(incorrect, weights=sample_weight, axis=0))

        # 判断是否需要提前停止
        # if estimator_error > 0.0:
        #     self.early_stop = True
        self.early_stop = lax.cond(
            estimator_error > 0.0,
            lambda _: jnp.array(True, dtype=jnp.bool_),
            lambda _: jnp.array(False, dtype=jnp.bool_),
            operand=None
        )

        def true_0_fun(sample_weight):
            return sample_weight, 1.0, 0.0, jnp.array(False, dtype=jnp.bool_)
        
        def false_0_fun(sample_weight):
            estimator_weight = self.learning_rate * (
                jnp.log((1.0 - estimator_error) / estimator_error) + jnp.log(n_classes - 1.0)
            )
            def not_last_iboost(sample_weight):
                # Only boost positive weights
                sample_weight *= jnp.exp(estimator_weight * incorrect)
                return sample_weight
            
            def last_iboost(sample_weight):
                return sample_weight
            
            sample_weight = lax.cond(iboost != self.n_estimators - 1,
                                     not_last_iboost, last_iboost, sample_weight)
            
            flag = estimator_error < 1.0 - (1.0 / n_classes)
            flag = lax.cond(
                self.early_stop,
                lambda _: jnp.array(False, dtype=jnp.bool_),
                lambda _: flag,
                operand=None
            )

            return sample_weight, estimator_weight, estimator_error, flag
            
        sample_weight, estimator_weight, estimator_error, flag = lax.cond(
            estimator_error <= 0.0, true_0_fun, false_0_fun, sample_weight
        )
        
        self.estimator_flags_.append(flag)  # 维护 flag 属性
        
        return sample_weight, estimator_weight, estimator_error
        
            
    def predict(self, X):
        pred = self.decision_function(X)
        
        if self.n_classes == 2:
            return self.classes.take(pred > 0, axis=0)
        
        return self.classes.take(jnp.argmax(pred, axis=1), axis=0)
        
    
    def decision_function(self, X):
        n_classes = self.n_classes
        classes = self.classes[:, jnp.newaxis]

        # pred = sum(
        #     jnp.where(
        #         (estimator.predict(X) == classes).T,
        #         w,
        #         -1 / (n_classes - 1) * w,
        #     )
        #     for estimator, w in zip(self.estimators_, self.estimator_weight_)
        # ) 
        # pred /= self.estimator_weight_.sum()
        
        pred = sum(
            jnp.where(
                (estimator.predict(X) == classes).T,
                w,
                -1 / (n_classes - 1) * w,
            ) * flag  # 使用 flag
            for estimator, w, flag in zip(self.estimators_, self.estimator_weight_, self.estimator_flags_)
        )
        
        # pred = sum(
        #     jnp.where(
        #         (estimator.predict(X) == classes).T,
        #         w,
        #         -1 / (n_classes - 1) * w,
        #     ) * flag
        #     for estimator, w, flag in zip(self.estimators_, self.estimator_weight_, self.estimator_flags_)
        #     if not self.early_stop or flag  # 使用 early_stop 进行过滤
        # )
        
        # 将列表转换为 JAX 数组，并进行求和
        weights_flags = jnp.array([w * flag for w, flag in zip(self.estimator_weight_, self.estimator_flags_)])
        pred /= jnp.sum(weights_flags)
        
        # # 计算每个估计器的预测结果
        # predictions = [
        #     jnp.where(
        #         (estimator.predict(X) == classes).T,
        #         w,
        #         -1 / (n_classes - 1) * w,
        #     )
        #     for estimator, w, flag in zip(self.estimators_, self.estimator_weight_, self.estimator_flags_)
        # ]
        
        # # 使用 lax.cond 处理 early_stop 逻辑
        # def apply_flags(predictions, weights_flags):
        #     return sum(p * f for p, f in zip(predictions, weights_flags))
        
        # weights_flags = jnp.array([w * flag for w, flag in zip(self.estimator_weight_, self.estimator_flags_)])
        
        # pred = lax.cond(
        #     self.early_stop,
        #     lambda _: apply_flags(predictions, jnp.array([0] * len(predictions))),
        #     lambda _: apply_flags(predictions, weights_flags),
        #     operand=None
        # )
        
        # pred /= jnp.sum(weights_flags)
        
        if n_classes == 2:
            pred[:, 0] *= -1
            return pred.sum(axis=1)
        return pred
        