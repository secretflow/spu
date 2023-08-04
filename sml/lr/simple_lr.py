import numpy as np
import jax.numpy as jnp
from enum import Enum
from ..utils.fxp_approx import sigmoid, SigType


class Penalty(Enum):
    NONE = 'None'
    L1 = 'l1'  # not supported
    L2 = 'l2'
    Elastic = 'elasticnet' # not supported

class MultiClass(Enum):
    Binary = 'binary'
    Ovr = 'ovr' # not supported yet
    Multy = 'multinomial' # not supported yet


class Solver(Enum):
    SGD = 'sgd'

class LogisticRegression:
    """
    Logistic Regression (aka logit, MaxEnt) classifier.

    IMPORTANT: Something different between `LogisticRegression` in sklearn:
        1. sigmoid will be computed with approximation
        2. you must define multi_class because we can not inspect y to decision the problem type
        3. for now, only 0-1 binary classification is supported; so if your label is {-1,1}, you must change it first!

    Parameters
    ----------
    penalty: Specify the norm of the penalty:
        {'l1', 'l2', 'elasticnet', 'None'}, default='l2' (current only support l2)

    solver: Algorithm to use in the optimization problem, default='sgd'.

    multi_class: specify whether and which multi-classification form
        {'binary', 'ovr', 'multinomial'}, default='binary' (current only support binary)
            - binary: binary problem
            - ovr: for each label, will fit a binary problem
            - multinomial: the loss minimised is the multinomial loss that fit across the entire probability distribution

    class_weight: not support yet, for multi-class tasks, default=None

    sig_type: the approximation method for sigmoid function, default='sr'
        for all choices, refer to `SigType`

    l2_norm: the strength of L2 norm, must be a positive float, default=0.01

    epochs, learning_rate, batch_size: hyper-parameters for sgd solver
        epochs: default=20
        learning_rate: default=0.1
        batch_size: default=512

    """
    def __init__(
        self,
        penalty: str='l2',
        solver: str='sgd',
        multi_class: str='binary',
        class_weight=None,
        sig_type: str='sr',
        l2_norm: float=0.01,
        epochs: int=20,
        learning_rate: float=0.1,
        batch_size: int=512,
    ):
        # parameter check.
        assert epochs > 0, f"epochs should >0"
        assert learning_rate > 0, f"learning_rate should >0"
        assert batch_size > 0, f"batch_size should >0"
        assert penalty == 'l2', "only support L2 penalty for now"
        assert solver == 'sgd', "only support sgd solver for now"
        if penalty == Penalty.L2:
            assert l2_norm > 0, f"l2_norm should >0 if use L2 penalty"
        assert penalty in [
            e.value for e in Penalty
        ], f"penalty should in {[e.value for e in Penalty]}, but got {penalty}"
        assert sig_type in [
            e.value for e in SigType
        ], f"sig_type should in {[e.value for e in SigType]}, but got {sig_type}"
        assert class_weight == None, f"not support class_weight for now"
        assert multi_class == 'binary', f"only support binary problem for now"

        self._epochs = epochs
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._l2_norm = l2_norm
        self._penalty = Penalty(penalty)
        self._sig_type = SigType(sig_type)
        self._class_weight = class_weight
        self._multi_class = MultiClass(multi_class)

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
            pred = sigmoid(pred, self._sig_type)

            err = pred - y_slice
            grad = jnp.matmul(jnp.transpose(x_slice), err)

            if self._penalty == Penalty.L2:
                w_with_zero_bias = jnp.resize(w, (num_feat, 1))
                w_with_zero_bias = jnp.concatenate(
                    (w_with_zero_bias, jnp.zeros((1, 1))),
                    axis=0,
                )
                grad = grad + w_with_zero_bias * self._l2_norm
            elif self._penalty == Penalty.L1:
                raise NotImplementedError
            elif self._penalty == Penalty.Elastic:
                raise NotImplementedError
            else:
                # None penalty
                raise NotImplementedError

            step = (self._learning_rate * grad) / batch_size

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
        batch_size = min(self._batch_size, num_sample)
        total_batch = int(num_sample / batch_size)
        weights = jnp.zeros((num_feat + 1, 1))

        # not support class_weight for now
        if isinstance(self._class_weight, dict):
            raise NotImplementedError
        elif self._class_weight == 'balanced':
            raise NotImplementedError

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
        """
        pred = self.decision_function(x)

        if self._multi_class == MultiClass.Binary:
            prob = sigmoid(pred, self._sig_type)
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

        if self._multi_class == MultiClass.Binary:
            # for binary task, only check whether logit > 0 (prob > 0.5)
            label = jnp.select([pred>0], [1], 0)
        else:
            raise NotImplementedError

        return label

    def decision_function(self, x):
        if self._multi_class == MultiClass.Binary:
            num_feat = x.shape[1]
            w = self._weights
            assert w.shape[0] == num_feat + 1, f"w shape is mismatch to x={x.shape}"
            assert len(w.shape) == 1 or w.shape[1] == 1, "w should be list or 1D array"
            w.reshape((w.shape[0], 1))

            bias = w[-1, 0]
            w = jnp.resize(w, (num_feat, 1))
            pred = jnp.matmul(x, w) + bias
            return pred
        elif self._multi_class == MultiClass.Ovr:
            raise NotImplementedError
        else:
            # Multy model here
            raise NotImplementedError