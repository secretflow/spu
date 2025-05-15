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

from enum import Enum

import jax
import jax.numpy as jnp
import numpy as np

from ..utils.fxp_approx import SigType, sigmoid
from sml.utils.utils import sml_reveal


class Penalty(Enum):
    NONE = 'None'
    L1 = 'l1'
    L2 = 'l2'
    Elastic = 'elasticnet'


class MultiClass(Enum):
    Binary = 'binary'
    Ovr = 'ovr'
    Multy = 'multinomial'  # not supported yet


class Solver(Enum):
    SGD = 'sgd'


class EarlyStoppingMetric(Enum):
    WEIGHT = 'weight'


class LogisticRegression:
    """
    Logistic Regression (aka logit, MaxEnt) classifier.

    IMPORTANT: Something different between `LogisticRegression` in sklearn:
        1. sigmoid will be computed with approximation
        2. you must define multi_class because we can not inspect y to decision the problem type
        3. Due to the inability to perform data exploration in encrypted state, it is necessary to specify a value for labels.
            For example, binary classification [0,1] ,multi classification [0,1,2].

    Parameters
    ----------
    penalty: Specify the norm of the penalty:
        {'l1', 'l2', 'elasticnet', 'None'}, default='l2'

    solver: Algorithm to use in the optimization problem, default='sgd'.

    multi_class: specify whether and which multi-classification form
        {'binary', 'ovr', 'multinomial'}, default='binary' (current only support binary)
            - binary: binary problem
            - ovr: for each label, will fit a binary problem
            - multinomial: the loss minimised is the multinomial loss that fit across the entire probability distribution

    class_labels: classification labels, default=[0, 1].
        Binary classification labels default=[0, 1].
        Unable to perform data exploration in a confidential state, multiple classification labels need to be specified.

    class_weight: not support yet, for multi-class tasks, default=None

    sig_type: the approximation method for sigmoid function, default='sr'
        for all choices, refer to `SigType`

    C: float, default=1.0
        Inverse of regularization strength; must be a positive float. Like in support vector machines,
        smaller values specify stronger regularization.

    l1_ratio: float, default=0.5
        The Elastic-Net mixing parameter, with 0 <= l1_ratio <= 1. Only used if penalty='elasticnet'.
        Setting l1_ratio=0 is equivalent to using penalty='l2', while setting l1_ratio=1 is equivalent to using penalty='l1'.
        For 0 < l1_ratio <1, the penalty is a combination of L1 and L2.

    epochs, learning_rate, batch_size: hyper-parameters for sgd solver
        epochs: default=20
        learning_rate: default=0.1
        batch_size: default=512

    early_stopping_threshold: float, default=0.0
        The threshold for early stopping, 0 means no early stopping.

    early_stopping_metric: str, default='weight'
        The metric for early stopping.
        'weight': use the max weight change as the metric


    """

    def __init__(
        self,
        penalty: str = 'l2',
        solver: str = 'sgd',
        multi_class: str = 'binary',
        class_labels: list = [0, 1],
        class_weight=None,
        sig_type: str = 'sr',
        C: float = 1.0,
        l1_ratio: float = 0.5,
        epochs: int = 20,
        learning_rate: float = 0.1,
        batch_size: int = 512,
        early_stopping_threshold: float = 0.0,
        early_stopping_metric: str = 'weight',
    ):
        # parameter check.
        assert epochs > 0, f"epochs should >0"
        assert learning_rate > 0, f"learning_rate should >0"
        assert batch_size > 0, f"batch_size should >0"
        assert solver == 'sgd', "only support sgd solver for now"
        assert C > 0, f"C should >0"
        assert early_stopping_threshold >= 0, f"early_stopping_threshold should >0"
        assert early_stopping_metric in [
            e.value for e in EarlyStoppingMetric
        ], f"early_stopping_metric should in {[e.value for e in EarlyStoppingMetric]}, but got {early_stopping_metric}"
        if penalty == Penalty.Elastic:
            assert (
                0 <= l1_ratio <= 1
            ), f"l1_ratio should in `[0, 1]` if use Elastic penalty"
        assert penalty in [
            e.value for e in Penalty
        ], f"penalty should in {[e.value for e in Penalty]}, but got {penalty}"
        assert sig_type in [
            e.value for e in SigType
        ], f"sig_type should in {[e.value for e in SigType]}, but got {sig_type}"
        assert class_weight == None, f"not support class_weight for now"
        assert multi_class in [
            'binary',
            'ovr',
        ], f"only support [binary,ovr] problem for now"

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.C = C
        self.l1_ratio = l1_ratio
        self.penalty = Penalty(penalty)
        self.sig_type = SigType(sig_type)
        self.class_weight = class_weight
        self.multi_class = MultiClass(multi_class)
        self.class_labels = class_labels
        self.early_stopping_threshold = early_stopping_threshold
        self.early_stopping_metric = EarlyStoppingMetric(early_stopping_metric)

        # threshold = 0 means no early stopping
        self._need_early_stopping = self.early_stopping_threshold > 0
        # totally running epochs
        self._running_epochs = 0

        self.weights = jnp.zeros(())

    def _update_weights(
        self,
        x,  # array-like
        y,  # array-like
        w,  # array-like
        total_batch: int,
        batch_size: int,
        pos_class: int,
    ) -> np.ndarray:
        num_feat = x.shape[1]
        assert w.shape[0] == num_feat + 1, "w shape is mismatch to x"
        assert len(w.shape) == 1 or w.shape[1] == 1, "w should be list or 1D array"
        w = w.reshape((w.shape[0], 1))

        mask = y == pos_class
        y = mask

        for idx in range(total_batch):
            begin = idx * batch_size
            end = (idx + 1) * batch_size
            if end > x.shape[0]:
                end = x.shape[0]
            this_batch_size = end - begin
            # padding one col for bias in w
            x_slice = jnp.concatenate(
                (x[begin:end, :], jnp.ones((this_batch_size, 1))), axis=1
            )
            y_slice = y[begin:end, :]

            pred = jnp.matmul(x_slice, w)
            pred = sigmoid(pred, self.sig_type)

            err = pred - y_slice
            grad = jnp.matmul(jnp.transpose(x_slice), err)

            if self.penalty != Penalty.NONE:
                w_with_zero_bias = jnp.resize(w, (num_feat, 1))
                w_with_zero_bias = jnp.concatenate(
                    (w_with_zero_bias, jnp.zeros((1, 1))),
                    axis=0,
                )
            if self.penalty == Penalty.L2:
                reg = w_with_zero_bias * 1.0 / self.C
            elif self.penalty == Penalty.L1:
                reg = jnp.sign(w_with_zero_bias) * 1.0 / self.C
            elif self.penalty == Penalty.Elastic:
                reg = (
                    jnp.sign(w_with_zero_bias) * self.l1_ratio * 1.0 / self.C
                    + w_with_zero_bias * (1 - self.l1_ratio) * 1.0 / self.C
                )
            else:
                reg = 0

            grad = grad + reg

            step = (self.learning_rate * grad) / this_batch_size

            w = w - step

        return w

    def fit(self, x, y):
        """Fit linear model with Stochastic Gradient Descent.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Training data.

        y : ndarray of shape (n_samples,), value MUST between {0,1,...k-1} (k is the number of classes)
            Target values.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        assert len(x.shape) == 2, f"expect x to be 2 dimension array, got {x.shape}"

        num_sample = x.shape[0]
        num_feat = x.shape[1]
        batch_size = min(self.batch_size, num_sample)
        total_batch = int((num_sample + batch_size - 1) / batch_size)
        n_classes = len(self.class_labels)
        _classes = self.class_labels
        if n_classes == 2:
            n_classes = 1
            _classes = _classes[1:]
        _coefs = [None] * n_classes

        # not support class_weight for now
        if isinstance(self.class_weight, dict):
            raise NotImplementedError
        elif self.class_weight == 'balanced':
            raise NotImplementedError

        for i in range(n_classes):
            # TODO: add more init methods
            weights = jnp.zeros((num_feat + 1, 1))
            weights_last = weights
            epoch = 0
            need_stop = False

            def cond_fun(carry):
                weights, weights_last, epoch, need_stop = carry
                return jnp.logical_and(epoch < self.epochs, jnp.logical_not(need_stop))

            def body_fun(carry):
                weights, weights_last, epoch, need_stop = carry
                weights_last = weights
                weights = self._update_weights(
                    x, y, weights, total_batch, batch_size, _classes[i]
                )
                if self._need_early_stopping:
                    if self.early_stopping_metric == EarlyStoppingMetric.WEIGHT:
                        test_criteria = jnp.max(jnp.abs(weights - weights_last))
                        stop = test_criteria < self.early_stopping_threshold
                    else:
                        raise NotImplementedError(
                            f"early_stopping_metric={self.early_stopping_metric} is not supported"
                        )
                    # WARNING: Reveal the need_stop to plaintext here
                    need_stop = sml_reveal(stop)
                else:
                    need_stop = False
                return (weights, weights_last, epoch + 1, need_stop)

            weights, weights_last, epoch, need_stop = jax.lax.while_loop(
                cond_fun, body_fun, (weights, weights_last, epoch, need_stop)
            )

            _coefs[i] = weights
            self._running_epochs = epoch

        self.weights = jnp.array(_coefs)
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
        """
        pred = self.decision_function(x)

        if self.multi_class == MultiClass.Binary:
            prob = sigmoid(pred[0], self.sig_type)
        elif self.multi_class == MultiClass.Ovr:
            preds = [None] * len(pred)
            for i in range(len(pred)):
                prob = sigmoid(pred[i], self.sig_type)
                preds[i] = prob.ravel()
            preds = jnp.transpose(jnp.array(preds))
            prob = preds / preds.sum(axis=1).reshape((preds.shape[0], -1))
            prob = prob.at[:, 0].set(1 - prob.sum(axis=1) + prob[:, 0])
        else:
            raise NotImplementedError

        return prob

    def predict(self, x):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : {array-like, } of shape (n_samples, n_features)
            The data matrix for which we want to get the predictions.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Vector containing the class labels for each sample.
        """
        pred = self.decision_function(x)

        if self.multi_class == MultiClass.Binary:
            label = jnp.select([pred[0] > 0], [1], 0)
        elif self.multi_class == MultiClass.Ovr:
            label = jnp.argmax(jnp.array(pred), axis=0)
        else:
            raise NotImplementedError

        return label.reshape((-1,))

    def decision_function(self, x):
        if self.multi_class in [MultiClass.Binary, MultiClass.Ovr]:
            n_classes = len(self.class_labels)
            preds = [None] * n_classes
            for i in range(n_classes):
                num_feat = x.shape[1]
                w = self.weights[i]
                assert w.shape[0] == num_feat + 1, f"w shape is mismatch to x={x.shape}"
                assert (
                    len(w.shape) == 1 or w.shape[1] == 1
                ), "w should be list or 1D array"
                bias = w[-1, 0]
                w = jnp.resize(w, (num_feat, 1))
                pred = jnp.matmul(x, w) + bias
                preds[i] = pred
            return preds
        else:
            raise NotImplementedError
