from enum import Enum

import numpy as np

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

    penalty : {'l2','l1','elasticnet','None'}, default=None
        The penalty (aka regularization term) to be used.

    alpha : float, default=0.0001
        Constant that multiplies the regularization term if regularization is
        used.

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
    """

    def __init__(
        self,
        penalty=None,
        alpha=0.0001,
        l1_ratio=0.15,
        fit_intercept=True,
        max_iter=1000,
        eta0=1.0,
    ):
        # parameter check.
        assert max_iter > 0, f"max_iter should >0"
        assert eta0 > 0, f"eta0 should >0"
        assert alpha > 0, f"alpha should >0"
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
        """The sign function in perceptron.
        if x <= 0, f(x)=-1 else if x > 0, f(x)=1
        """
        x = jnp.sign(x)
        x = jnp.where(x == 0, -1, x)
        return x

    def _update_parameters(self, x_i, y_i, w, b):
        # if y_i * (w * x_i) + b <= 0, the point is misclassified used for updating w and b.
        decision = y_i * (jnp.dot(w, x_i) + b) <= 0

        dw = decision * (self.eta0 * y_i * x_i)
        if self.fit_intercept:
            db = decision * self.eta0 * y_i
        else:
            db = 0

        # add regularization terms
        if self.penalty != Penalty.NONE:
            l1_reg = jnp.sign(w)
            l2_reg = w
            reg = self.l1_ratio * l1_reg + 0.5 * (1 - self.l1_ratio) * l2_reg
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
            Target values.

        Returns
        -------
        self : object
            Returns an instance of self.
        """

        assert len(x.shape) == 2, f"expect x to be 2 dimension array, got {x.shape}"

        n_samples, n_features = x.shape

        w = jnp.zeros(n_features)
        b = jnp.array([0.0])

        for iter in range(self.max_iter):
            update_w_b = jax.vmap(
                self._update_parameters, in_axes=(0, 0, None, None), out_axes=0
            )(x, y, w, b)
            sum_w_b = jnp.sum(jnp.array(update_w_b), axis=0)
            w += sum_w_b[:-1]
            b += sum_w_b[-1]

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

        pred_ = jnp.matmul(x, self._w) + self._b
        return self._sign(pred_)
