'''
Author: Li Zhihang
Date: 2024-07-03 11:29:34
LastEditTime: 2024-07-27 19:24:52
FilePath: /klaus/spu/sml/quantile/quantile.py
Description: 报错status=1，原因是b矩阵有负值，需要将其转为非负值
'''
import jax.numpy as jnp
from jax import grad
import jax

import numbers
import pandas as pd
from warnings import warn  
import warnings

# from scipy.optimize import linprog

# from _linprog import _linprog_simplex
from sml.quantile.utils.linprog import _linprog_simplex

def _num_samples(x):
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

def _check_sample_weight(sample_weight, X, dtype=None, copy=False, only_non_negative=False):
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

    def __init__(self, quantile=0.5, alpha=1.0, fit_intercept=True, lr=0.01, max_iter=1000, n_samples=100):
        self.quantile = quantile
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.lr = lr
        self.max_iter = max_iter
        self.n_samples = n_samples
        
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X, y, sample_weight=None):
        n_samples, n_features = X.shape
        n_params = n_features
        
        # sample_weight = _check_sample_weight(sample_weight, X)
        sample_weight = jnp.ones((self.n_samples,))
        
        if self.fit_intercept:
            n_params += 1
        
        alpha = jnp.sum(sample_weight) * self.alpha
        
        # indices = jnp.nonzero(sample_weight)[0]
        # n_indices = len(indices)
        # if n_indices < len(sample_weight):
        #     sample_weight = sample_weight[indices]
        #     X = _safe_indexing(X, indices)
        #     y = _safe_indexing(y, indices)
        
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
        
        # eye = jnp.eye(n_indices)
        eye = jnp.eye(self.n_samples)
        if self.fit_intercept:
            # ones = jnp.ones((n_indices,1))
            ones = jnp.ones((self.n_samples,1))
            A = jnp.concatenate([ones, X, -ones, -X, eye, -eye], axis=1)
        else:
            A = jnp.concatenate([X, -X, eye, -eye], axis=1)
        
        b = y
        
        n,m = A.shape
        av = jnp.arange(n) + m

        result = _linprog_simplex(c, A, b, maxiter=self.max_iter,tol=1e-3)
        
        # result = linprog(c,A_eq=A,b_eq=b,method='simplex')
        # print("result",result)
        # print("Optimal solution:", result['x'])
        # print("Optimal solution:", result[0])
        # print("Optimal value:", result[0]@c)
        solution = result[0]
        # solution = result['x']
        # 取消了1: "Iteration limit reached."因为这个方法就是达到迭代次数停止的
        # if not result[1]:
        # # if not result['success']:
        #     failure = {
        #         1: "Iteration limit reached.",
        #         2: "Optimization failed. Unable to find a feasible"
        #            " starting point.",
        #         3: "Optimization failed. The problem appears to be unbounded.",
        #         4: "Optimization failed. Singular matrix encountered."
        #     }
        #     warnings.warn(
        #         "Linear programming for QuantileRegressor did not succeed.\n"
        #         f"Status is {result[1]}: "
        #         # + failure.setdefault(result[1], "unknown reason")
        #         + "\n"
        #         + "Result message of linprog:\n"
        #         # + result[2],
        #         # ConvergenceWarning,
        #     )
            
        params = solution[:n_params] - solution[n_params : 2 * n_params]
        # print("params:",params)
        self.n_iter_ = result[2]
        # self.n_iter_ = result['nit']
        
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
            # print("X:", X)
            # print("intercept_:", self.intercept_)
            # print("coef_:", self.coef_)
            return jnp.dot(X, jnp.hstack([self.intercept_, self.coef_]))
        else:
            return jnp.dot(X, self.coef_)
        # return jnp.dot(X, jnp.hstack([self.intercept_, self.coef_]))

from jax import random
from sklearn.linear_model import QuantileRegressor as SklearnQuantileRegressor
@jax.jit
def compare_quantile_regressors(X, y, quantile=0.2, alpha=0.1, lr=0.01, max_iter=1000):
    # 训练和预测自定义模型
    custom_model = QuantileRegressor(quantile=quantile, alpha=alpha, fit_intercept=True, lr=lr, max_iter=max_iter)
    custom_model.fit(X, y)
    custom_y_pred = custom_model.predict(X)

    print("Custom Model:")
    print("Mean of y <= Custom Predictions:", jnp.mean(y <= custom_y_pred))
    print("Custom Coefficients:", custom_model.coef_)
    print("Custom Intercept:", custom_model.intercept_)

if __name__ == "__main__":
    key = random.PRNGKey(42)
    key, subkey = random.split(key)
    X = random.normal(subkey, (100, 2))
    y = 5 * X[:, 0] + 2 * X[:, 1] + random.normal(key, (100,)) * 0.1
    
    compare_quantile_regressors(X,y)





# # 设置随机种子
# key = random.PRNGKey(42)
# # 生成 X 数据
# key, subkey = random.split(key)
# X = random.normal(subkey, (100, 2)) 
# # 生成 y 数据
# y = 5 * X[:, 0] + 2 * X[:, 1] + random.normal(key, (100,)) * 0.1  # 高相关性，带有小噪声

# # print

# custom_model = QuantileRegressor(quantile=0.2, alpha=0.1, fit_intercept=True, lr=0.01, max_iter=1000)
# custom_model.fit(X, y)
# custom_y_pred = custom_model.predict(X)

# print(jnp.mean(y <= custom_model.predict(X)))
# print("Custom Coefficients:", custom_model.coef_)
# print("Custom Intercept:", custom_model.intercept_)


# from sklearn.linear_model import QuantileRegressor as SklearnQuantileRegressor

# sklearn_model = SklearnQuantileRegressor(quantile=0.2, alpha=0.1, fit_intercept=True, solver='highs')
# sklearn_model.fit(X, y)
# sklearn_y_pred = sklearn_model.predict(X)

# print(jnp.mean(y <= sklearn_model.predict(X)))
# print("Sklearn Coefficients:", sklearn_model.coef_)
# print("Sklearn Intercept:", sklearn_model.intercept_)




# # Print first 10 predictions
# print("Sklearn Predictions:", sklearn_y_pred[:10])
# print("Custom Predictions:", custom_y_pred[:10])

