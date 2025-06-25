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


import math
from enum import Enum

import jax
import jax.numpy as jnp


class Penalty(Enum):
    NONE = None
    L1 = 'l1'
    L2 = 'l2'
    EN = 'elasticnet'


class Perceptron:
    """Perceptron.

    Parameters
    ----------

    penalty : {'l2','l1','elasticnet',None(NoneType)}, default=None
        The penalty (aka regularization term) to be used.

    alpha : float, default=0.0001
        Constant that multiplies the regularization term if regularization is
        used.

    patience: int, default=10
        How long to wait after last time loss improved.

    l1_ratio : float, default=0.15
        The Elastic Net mixing parameter, with `0 <= l1_ratio <= 1`.
        `l1_ratio=0` corresponds to L2 penalty, `l1_ratio=1` to L1.
        Only used if `penalty='elasticnet'`.

    fit_intercept : bool, default=True
        Whether the intercept should be estimated or not. If False, the
        data is assumed to be already centered.

    max_iter : int, default=1000
        The maximum number of passes over the training data (aka epochs).
        It only impacts the behavior in the ``fit`` method, and not the
        :meth:`partial_fit` method.

    eta0 : float, default=1
        Constant by which the updates are multiplied.

    batch_size : int, default=-1
        The batch size is a number of samples processed before the model is updated.
        if batch_size = -1, all samples will be processed in an iteration.

    early_stop : bool, default=True
        When the loss function of the model no longer decreases, the early stop method
        can avoid the overtraining of the model if True.
    """

    def __init__(
        self,
        penalty=None,
        alpha=0.0001,
        patience=10,
        l1_ratio=0.15,
        fit_intercept=True,
        max_iter=1000,
        eta0=1.0,
        batch_size=-1,
        early_stop=True,
    ):
        # parameter check.
        assert max_iter > 0, f"max_iter should >0"
        assert eta0 > 0, f"eta0 should >0"
        assert alpha > 0, f"alpha should >0"
        assert patience > 0, f"patience should >0"
        assert batch_size > 0 or batch_size == -1, f"batch size should >0 or -1"
        assert penalty in [
            e.value for e in Penalty
        ], f"penalty should in {[e.value for e in Penalty]}, but got {penalty}"
        if penalty == "elasticnet":
            assert (
                l1_ratio > 0 and l1_ratio < 1
            ), f"l1_ratio should >0 and <1 if use ElasticNet (L1+L2)"

        self.max_iter = max_iter
        self.eta0 = eta0
        self.fit_intercept = fit_intercept
        self.penalty = Penalty(penalty)
        self.alpha = alpha
        self.patience = patience
        self.bsize = batch_size
        self.early_stop = early_stop
        if self.penalty == Penalty.L1:
            self.l1_ratio = 1
        elif self.penalty == Penalty.L2:
            self.l1_ratio = 0
        elif self.penalty == Penalty.EN:
            self.l1_ratio = l1_ratio
        else:
            self.l1_ratio = None

        self._w = None
        self._b = 0.0

    def _sign(self, x):
        """
        The sign function in perceptron.
        if x <= 0, f(x)=-1 else if x > 0, f(x)=1
        """
        x = jnp.where(x <= 0, -1, 1)
        return x

    def _hinge_function(self, x, y, w, b):
        """
        loss = \frac{1}{n} \sum_{i=1}^{n}max\{-y_i(wx_i+b), 0\}.
        """
        y_hat = jnp.matmul(x, w) + b
        y_hat = y_hat.reshape(y_hat.shape[0], 1)
        return jnp.mean(jnp.maximum(jnp.multiply(-y_hat, y), 0))

    def _update_parameters(self, x_i, y_i, w, b):
        # if y_i * [(w * x_i) + b] <= 0, the point is misclassified used for updating w and b.
        decision = y_i * (jnp.dot(w, x_i) + b) <= 0

        dw = decision * (self.eta0 * y_i * x_i)
        if self.fit_intercept:
            db = decision * self.eta0 * y_i
        else:
            db = jnp.array([0.0])

        # add regularization terms
        if self.penalty != Penalty.NONE:
            l1_reg = jnp.sign(w)
            l2_reg = w
            reg = self.l1_ratio * l1_reg + (1 - self.l1_ratio) * l2_reg
            dw += decision * self.alpha * reg

        dwb = jnp.concatenate((dw, db), axis=0)

        return dwb

    def fit(self, x, y):
        """Fit Perceptron classifier.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Training data.

        y : ndarray of shape (n_samples,)
            Target values. Only -1 or 1.

        Returns
        -------
        self : object
            Returns an instance of self.
        """

        assert len(x.shape) == 2, f"expect x to be 2 dimension array, got {x.shape}"

        n_samples, n_features = x.shape
        if self.bsize == -1 or self.bsize > n_samples:
            self.bsize = n_samples

        w = jnp.zeros(n_features)
        b = jnp.array([0.0])

        not_early_stop = True

        best_loss = jnp.inf
        best_w = w
        best_b = b
        best_iter = -1

        for iter in range(self.max_iter):
            total_batch = math.ceil(float(n_samples) / self.bsize)
            for idx in range(total_batch):
                begin = idx * self.bsize
                end = min((idx + 1) * self.bsize, n_samples)
                x_slice = x[begin:end]
                y_slice = y[begin:end]

                update_w_b = jax.vmap(
                    self._update_parameters, in_axes=(0, 0, None, None), out_axes=0
                )(x_slice, y_slice, w, b)
                mean_w_b = jnp.mean(jnp.array(update_w_b), axis=0)

                w += not_early_stop * mean_w_b[:-1]

                if self.fit_intercept:
                    b += not_early_stop * mean_w_b[-1]

            if self.early_stop:
                sumloss = self._hinge_function(x, y, w, b)

                best_loss_flag = sumloss < best_loss
                best_loss = jnp.where(best_loss_flag, sumloss, best_loss)
                best_w = jnp.where(best_loss_flag, w, best_w)
                best_b = jnp.where(best_loss_flag, b, best_b)
                best_iter = jnp.where(best_loss_flag, iter, best_iter)

                not_early_stop *= iter <= best_iter + self.patience

        if self.early_stop:
            self._w = best_w
            self._b = best_b
        else:
            self._w = w
            self._b = b

        return self

    def predict(self, x):
        """Perceptron estimates.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Input data for prediction.

        Returns
        -------
        ndarray of shape (n_samples,)
            Returns the result of the sample for each class in the model.
        """
        assert self._w is not None, f"the model should be trained before prediction."

        pred_ = jnp.matmul(x, self._w) + self._b
        return self._sign(pred_)
