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
import warnings
from sml.tree.tree_w import DecisionTreeClassifier as sml_dtc

class AdaBoostClassifier:
    """A adaboost classifier based on DecisionTreeClassifier.
    
    Parameters
    ----------
    estimator : {"dtc"}, default="dtc"
        Specifies the type of model or algorithm to be used for training.
        Supported estimators are "dtc".
    
    n_estimators : int
        The number of estimators. Must specify an integer > 0.
        
    max_depth : int
        The maximum depth of the tree. Must specify an integer > 0.
        
    learning_rate : float 
        The step size used to update the model weights during training.
        It's an float, must learning_rate > 0.
    
    n_classes: int
        The max number of classes.
    
    """
    def __init__(
        self,
        estimator,
        # 默认estimator为决策树,criterion == "gini" splitter == "best"
        n_estimators,
        max_depth,
        learning_rate,
        n_classes,
    ):
        assert estimator == "dtc", "estimator other than dtc is not supported."
        assert (
            n_estimators is not None and n_estimators > 0
        ), "n_estimators should not be None and must > 0."
        assert(
            max_depth is not None and max_depth > 0
        ), "max_depth should not be None and must > 0."
        
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_classes = n_classes
        
        self.estimators_ = []
        self.estimator_weight = jnp.zeros(self.n_estimators, dtype=jnp.float32)
        self.estimator_errors = jnp.ones(self.n_estimators, dtype=jnp.float32)
    
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
    
    def _check_sample_weight(self, sample_weight, X, dtype=None, copy=False, only_non_negative=False):
        '''
        description: 验证样本权重.
        return {*}
        '''   
        # jax默认只支持float32，
        # 如果需要启用 float64 类型，可以设置 jax_enable_x64 配置选项或 JAX_ENABLE_X64 环境变量。
        n_samples = self._num_samples(X)
        if dtype is not None and dtype not in [jnp.float32, jnp.float64]:
            dtype = jnp.float32
            
        if sample_weight is None:
            sample_weight = jnp.ones(n_samples, dtype=dtype)
        elif isinstance(sample_weight, numbers.Number):
            sample_weight = jnp.full(n_samples, sample_weight, dtype=dtype)
        else:
            sample_weight = jnp.asarray(sample_weight, dtype=dtype)
            if sample_weight.ndim != 1:
                raise ValueError("Sample weight must be 1D array or scalar")
            
            if sample_weight.shape[0] != n_samples:
                raise ValueError(
                    "sample_weight.shape == {}, expected {}!".format(
                        sample_weight.shape, (n_samples,)
                    )
                )
        
        if copy:
            sample_weight = jnp.copy(sample_weight)
        
        return sample_weight
    
    def cond_fun(self, iboost, sample_weight, estimator_weight, estimator_error):
        status1 = jnp.logical_and(iboost < self.n_estimators, jnp.all(jnp.isfinite(sample_weight)))
        status2 = jnp.logical_and(estimator_error > 0, jnp.sum(sample_weight) > 0)
        status = jnp.logical_and(status1, status2)
        return status
    
    
    def fit(self, X, y, sample_weight=None):
        sample_weight = self._check_sample_weight(
            sample_weight, X, copy=True, only_non_negative=True
        )
        sample_weight /= sample_weight.sum()

        self.classes = y

        
        epsilon = jnp.finfo(sample_weight.dtype).eps
        
        self.estimator_weight_ = jnp.zeros(self.n_estimators, dtype=jnp.float32)
        self.estimator_errors_ = jnp.ones(self.n_estimators, dtype=jnp.float32)
        
        for iboost in range(self.n_estimators):
            sample_weight = jnp.clip(sample_weight, a_min=epsilon, a_max=None)
            
            sample_weight, estimator_weight, estimator_error = self._boost_discrete(
                iboost, X, y, sample_weight
            )

            self.estimator_weight_ = self.estimator_weight_.at[iboost].set(estimator_weight)
            self.estimator_errors_ = self.estimator_errors_.at[iboost].set(estimator_error)

            sample_weight_sum = jnp.sum(sample_weight)
            def not_last_iboost(sample_weight, sample_weight_sum):
                sample_weight /= sample_weight_sum
                return sample_weight
            def last_iboost(sample_weight, sample_weight_sum):
                return sample_weight
            sample_weight = lax.cond(iboost<self.n_estimators,
                                     lambda : not_last_iboost(sample_weight, sample_weight_sum),
                                     lambda : last_iboost(sample_weight, sample_weight_sum))
 
        return self
    
    def _boost_discrete(self, iboost, X, y, sample_weight):
        """Implement a single boost using the SAMME discrete algorithm."""
        estimator = sml_dtc("gini", "best", self.max_depth, self.n_classes)
        self.estimators_.append(estimator)
        
        n_classes = self.n_classes
        
        estimator.fit(X, y, sample_weight=sample_weight)
        
        y_predict = estimator.predict(X)
        
        incorrect = y_predict != y
        estimator_error = jnp.mean(jnp.average(incorrect, weights=sample_weight, axis=0))

        def true_0_fun(sample_weight):
            return sample_weight, 1.0, 0.0
        
        def false_0_fun(sample_weight):
            estimator_weight = self.learning_rate * (
                jnp.log((1.0 - estimator_error) / estimator_error) + jnp.log(n_classes - 1.0)
            )
            def not_last_iboost(sample_weight):
                # Only boost positive weights
                sample_weight = jnp.exp(
                    jnp.log(sample_weight)
                    + estimator_weight * incorrect * (sample_weight > 0)
                )
                return sample_weight
            
            def last_iboost(sample_weight):
                return sample_weight
            
            sample_weight = lax.cond(iboost != self.n_estimators - 1,
                                     not_last_iboost, last_iboost, sample_weight)
            
            
            return sample_weight, estimator_weight, estimator_error
            
        sample_weight, estimator_weight, estimator_error = lax.cond(
            estimator_error <= 0.0, true_0_fun, false_0_fun, sample_weight
        )
        
        return sample_weight, estimator_weight, estimator_error
        
            
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
            for estimator, w in zip(self.estimators_, self.estimator_weight_)
        ) 
        pred /= self.estimator_weight_.sum()
        
        if n_classes == 2:
            pred[:, 0] *= -1
            return pred.sum(axis=1)
        return pred
        
        
# import jax.numpy as jnp
# from sklearn.datasets import load_iris
# from sklearn.metrics import accuracy_score, classification_report
# def load_data():
#     iris = load_iris()
#     iris_data, iris_label = jnp.array(iris.data), jnp.array(iris.target)
#     # sorted_features: n_samples * n_features_in
#     n_samples, n_features_in = iris_data.shape
#     n_labels = len(jnp.unique(iris_label))
#     sorted_features = jnp.sort(iris_data, axis=0)
#     new_threshold = (sorted_features[:-1, :] + sorted_features[1:, :]) / 2
#     new_features = jnp.greater_equal(
#         iris_data[:, :], new_threshold[:, jnp.newaxis, :]
#     )
#     new_features = new_features.transpose([1, 0, 2]).reshape(n_samples, -1)

#     X, y = new_features[:, ::3], iris_label[:]
#     return X, y

# X,y = load_data()
# n_labels = len(jnp.unique(y))
# model = AdaBoostClassifier(estimator='dtc', n_estimators=50,max_depth=2,learning_rate=0.5, n_classes=n_labels)
# # 训练AdaBoost模型
# model =model.fit(X, y, sample_weight=None)
# # print(model.estimator_weight_)
# print(model.estimator_errors_)
# # 预测测试集
# y_pred = model.predict(X)
# # print(y_pred)

# n_samples, n_features = X.shape
# score_encrypted = jnp.mean(y_pred == y) 
# print(y_pred)
# print(y)
# print(f"Accuracy in SPU: {score_encrypted}")

# # 输出预测结果的准确率和分类报告
# print(f"Accuracy: {accuracy_score(y, y_pred)}")

# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier

# base_estimator = DecisionTreeClassifier(max_depth=2)  # 基分类器
# model = AdaBoostClassifier(estimator=base_estimator, n_estimators=50,learning_rate=0.5,algorithm="SAMME")

# # 训练AdaBoost模型
# model.fit(X, y, sample_weight=None)
# print(model.estimator_errors_)
# # 预测测试集
# y_pred = model.predict(X)
# print(y_pred)
# score_plain = model.score(X, y)
# print(score_plain)
# # 输出预测结果的准确率和分类报告
# print(f"Accuracy: {accuracy_score(y, y_pred)}")
        