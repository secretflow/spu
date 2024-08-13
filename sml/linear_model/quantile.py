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

import numbers
import warnings
from warnings import warn

import jax
import jax.numpy as jnp
import pandas as pd
from jax import grad

# from _linprog import _linprog_simplex
from sml.linear_model.utils.linprog import _linprog_simplex

# from scipy.optimize import linprog


def _num_samples(x):
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


def _check_sample_weight(
    sample_weight, X, dtype=None, copy=False, only_non_negative=False
):
    '''
    description: 验证样本权重.
    return {*}
    '''
    # jax默认只支持float32，
    # 如果需要启用 float64 类型，可以设置 jax_enable_x64 配置选项或 JAX_ENABLE_X64 环境变量。
    n_samples = _num_samples(X)
    if dtype is not None and dtype not in [jnp.float32, jnp.float64]:
        dtype = jnp.float32

    if sample_weight is None:
        sample_weight = jnp.ones(n_samples, dtype=dtype)
    elif isinstance(sample_weight, numbers.Number):
        sample_weight = jnp.full(n_samples, sample_weight, dtype=dtype)
    else:
        sample_weight = jnp.asarray(sample_weight, dtype=dtype)
        if sample_weight.ndim != 1:
            raise ValueError("Sample weights must be 1D array or scalar")

        if sample_weight.shape[0] != n_samples:
            raise ValueError(
                "sample_weight.shape == {}, expected {}!".format(
                    sample_weight.shape, (n_samples,)
                )
            )

    if only_non_negative and not jnp.all(sample_weight >= 0):
        raise ValueError("`sample_weight` cannot contain negative weights")

    if copy:
        sample_weight = jnp.copy(sample_weight)

    return sample_weight


def _safe_indexing(X, indices, *, axis=0):
    if indices is None:
        return X

    if axis not in (0, 1):
        raise ValueError(
            "'axis' should be either 0 (to index rows) or 1 (to index "
            " column). Got {} instead.".format(axis)
        )

    if axis == 0 and isinstance(indices, str):
        raise ValueError("String indexing is not supported with 'axis=0'")

    if axis == 1 and isinstance(X, list):
        raise ValueError("axis=1 is not supported for lists")

    if axis == 1 and hasattr(X, "shape") and len(X.shape) != 2:
        raise ValueError(
            "'X' should be a 2D JAXNumPy array,  "
            "dataframe when indexing the columns (i.e. 'axis=1'). "
            "Got {} instead with {} dimension(s).".format(type(X), len(X.shape))
        )

    if axis == 1 and isinstance(indices, str) and not isinstance(X, pd.DataFrame):
        raise ValueError(
            "Specifying the columns using strings is only supported for dataframes."
        )

    if isinstance(X, pd.DataFrame):
        return pandas_indexing(X, indices, axis=axis)
    elif isinstance(X, jnp.ndarray):
        return numpy_indexing(X, indices, axis=axis)
    elif isinstance(X, list):
        return list_indexing(X, indices, axis=axis)
    else:
        raise ValueError("Unsupported input type for X: {}".format(type(X)))


def pandas_indexing(X, indices, axis=0):
    if axis == 0:
        return X.iloc[indices]
    elif axis == 1:
        return X[indices]


def numpy_indexing(X, indices, axis=0):
    if axis == 0:
        return X[indices]
    elif axis == 1:
        return X[:, indices]


def list_indexing(X, indices, axis=0):
    if axis == 0:
        return [X[idx] for idx in indices]
    else:
        raise ValueError("axis=1 is not supported for lists")


class QuantileRegressor:

    def __init__(
        self, quantile=0.5, alpha=1.0, fit_intercept=True, lr=0.01, max_iter=1000
    ):
        self.quantile = quantile
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.lr = lr
        self.max_iter = max_iter

        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y, sample_weight=None):
        n_samples, n_features = X.shape
        n_params = n_features

        sample_weight = jnp.ones((n_samples,))

        if self.fit_intercept:
            n_params += 1

        alpha = jnp.sum(sample_weight) * self.alpha

        c = jnp.concatenate(
            [
                jnp.full(2 * n_params, fill_value=alpha),
                sample_weight * self.quantile,
                sample_weight * (1 - self.quantile),
            ]
        )

        if self.fit_intercept:
            c = c.at[0].set(0)
            c = c.at[n_params].set(0)

        eye = jnp.eye(n_samples)
        if self.fit_intercept:
            ones = jnp.ones((n_samples, 1))
            A = jnp.concatenate([ones, X, -ones, -X, eye, -eye], axis=1)
        else:
            A = jnp.concatenate([X, -X, eye, -eye], axis=1)

        b = y

        n, m = A.shape
        av = jnp.arange(n) + m

        result = _linprog_simplex(c, A, b, maxiter=self.max_iter, tol=1e-3)

        solution = result[0]

        params = solution[:n_params] - solution[n_params : 2 * n_params]
        self.n_iter_ = result[2]

        if self.fit_intercept:
            self.coef_ = params[1:]
            self.intercept_ = params[0]
        else:
            self.coef_ = params
            self.intercept_ = 0.0
        return self

    def predict(self, X):
        if self.fit_intercept:
            X = jnp.column_stack((jnp.ones(X.shape[0]), X))

            return jnp.dot(X, jnp.hstack([self.intercept_, self.coef_]))
        else:
            return jnp.dot(X, self.coef_)
