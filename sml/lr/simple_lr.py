import numpy as np
import jax.numpy as jnp
from enum import Enum

def t1_sig(x, limit: bool = True):
    '''
    taylor series referenced from:
    https://mortendahl.github.io/2017/04/17/private-deep-learning-with-mpc/
    '''
    T0 = 1.0 / 2
    T1 = 1.0 / 4
    ret = T0 + x * T1
    if limit:
        return jnp.select([ret < 0, ret > 1], [0, 1], ret)
    else:
        return ret


def t3_sig(x, limit: bool = True):
    '''
    taylor series referenced from:
    https://mortendahl.github.io/2017/04/17/private-deep-learning-with-mpc/
    '''
    T3 = -1.0 / 48
    ret = t1_sig(x, False) + jnp.power(x, 3) * T3
    if limit:
        return jnp.select([x < -2, x > 2], [0, 1], ret)
    else:
        return ret


def t5_sig(x, limit: bool = True):
    '''
    taylor series referenced from:
    https://mortendahl.github.io/2017/04/17/private-deep-learning-with-mpc/
    '''
    T5 = 1.0 / 480
    ret = t3_sig(x, False) + jnp.power(x, 5) * T5
    if limit:
        return jnp.select([ret < 0, ret > 1], [0, 1], ret)
    else:
        return ret


def seg3_sig(x):
    '''
    f(x) = 0.5 + 0.125x if -4 <= x <= 4
           1            if       x > 4
           0            if  -4 > x
    '''
    return jnp.select([x < -4, x > 4], [0, 1], 0.5 + x * 0.125)


def df_sig(x):
    '''
    https://dergipark.org.tr/en/download/article-file/54559
    Dataflow implementation of sigmoid function:
    F(x) = 0.5 * ( x / ( 1 + |x| ) ) + 0.5
    df_sig has higher precision than sr_sig if x in [-2, 2]
    '''
    return 0.5 * (x / (1 + jnp.abs(x))) + 0.5


def sr_sig(x):
    '''
    https://en.wikipedia.org/wiki/Sigmoid_function#Examples
    Square Root approximation functions:
    F(x) = 0.5 * ( x / ( 1 + x^2 )^0.5 ) + 0.5
    sr_sig almost perfect fit to sigmoid if x out of range [-3,3]
    '''
    return 0.5 * (x / jnp.sqrt(1 + jnp.square(x))) + 0.5


def ls7_sig(x):
    '''Polynomial fitting'''
    return (
        5.00052959e-01
        + 2.35176260e-01 * x
        - 3.97212202e-05 * jnp.power(x, 2)
        - 1.23407424e-02 * jnp.power(x, 3)
        + 4.04588962e-06 * jnp.power(x, 4)
        + 3.94330487e-04 * jnp.power(x, 5)
        - 9.74060972e-08 * jnp.power(x, 6)
        - 4.74674505e-06 * jnp.power(x, 7)
    )


def mix_sig(x):
    '''
    mix ls7 & sr sig, use ls7 if |x| < 4 , else use sr.
    has higher precision in all input range.
    NOTICE: this method is very expensive, only use for hessian matrix.
    '''
    ls7 = ls7_sig(x)
    sr = sr_sig(x)
    return jnp.select([x < -4, x > 4], [sr, sr], ls7)


def real_sig(x):
    return 1 / (1 + jnp.exp(-x))

def sigmoid(x, sig_type):
    if sig_type is SigType.REAL:
        return real_sig(x)
    elif sig_type is SigType.T1:
        return t1_sig(x)
    elif sig_type is SigType.T3:
        return t3_sig(x)
    elif sig_type is SigType.T5:
        return t5_sig(x)
    elif sig_type is SigType.DF:
        return df_sig(x)
    elif sig_type is SigType.SR:
        return sr_sig(x)
    elif sig_type is SigType.MIX:
        return mix_sig(x)

class SigType(Enum):
    REAL = 'real'
    T1 = 't1'
    T3 = 't3'
    T5 = 't5'
    DF = 'df'
    SR = 'sr'
    # DO NOT use this except in hessian case.
    MIX = 'mix'

class Penalty(Enum):
    NONE = 'None'
    L1 = 'l1'  # not supported
    L2 = 'l2'
    Elastic = 'elasticnet' # not supported

class MultiClass(Enum):
    Ovr = 'ovr' # binary problem
    Multy = 'multinomial' # multi_class problem not supported


class SGDClassifier:
    def __init__(
        self,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        penalty: str,
        sig_type: str,
        l2_norm: float,
        class_weight: None,
        multi_class: str,
    ):
        # parameter check.
        assert epochs > 0, f"epochs should >0"
        assert learning_rate > 0, f"learning_rate should >0"
        assert batch_size > 0, f"batch_size should >0"
        assert penalty == 'l2', "only support L2 penalty for now"
        if penalty == Penalty.L2:
            assert l2_norm > 0, f"l2_norm should >0 if use L2 penalty"
        assert penalty in [
            e.value for e in Penalty
        ], f"penalty should in {[e.value for e in Penalty]}, but got {penalty}"
        assert sig_type in [
            e.value for e in SigType
        ], f"sig_type should in {[e.value for e in SigType]}, but got {sig_type}"
        assert class_weight == None, f"not support class_weight for now"
        assert multi_class == 'ovr', f"only support binary problem for now"

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
                pass
            elif self._penalty == Penalty.Elastic:
                pass

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

        # not support class_weight for now
        if isinstance(self._class_weight, dict):
            pass
        elif self._class_weight == 'balanced':
            pass

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
        if self._multi_class == MultiClass.Ovr:
            num_feat = x.shape[1]
            w = self._weights
            assert w.shape[0] == num_feat + 1, f"w shape is mismatch to x={x.shape}"
            assert len(w.shape) == 1 or w.shape[1] == 1, "w should be list or 1D array"
            w.reshape((w.shape[0], 1))

            bias = w[-1, 0]
            w = jnp.resize(w, (num_feat, 1))
            pred = jnp.matmul(x, w) + bias
            pred = sigmoid(pred, self._sig_type)
            return pred
        elif self._multi_class == MultiClass.Multy:
            # not support multi_class problem for now
            pass
