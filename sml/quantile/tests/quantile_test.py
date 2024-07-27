'''
Author: Li Zhihang
Date: 2024-07-07 20:27:16
LastEditTime: 2024-07-27 11:36:49
FilePath: /klaus/spu/sml/quantile/tests/quantile_test.py
Description: 
'''
import unittest

import jax.numpy as jnp
import numpy as np
from sklearn.linear_model import QuantileRegressor as SklearnQuantileRegressor

import spu.spu_pb2 as spu_pb2  # type: ignore
import spu.utils.simulation as spsim
from sml.quantile.quantile import QuantileRegressor as SmlQuantileRegressor


class UnitTests(unittest.TestCase):
    def test_forest(self):
        def proc_wrapper(
            quantile=0.5,
            alpha=1.0,
            fit_intercept=True,
            lr=0.01,
            max_iter=1000,
        ):
            quantile_custom = SmlQuantileRegressor(
                quantile,
                alpha,
                fit_intercept,
                lr,
                max_iter,
            )
            
            def proc(X, y):
                quantile_custom_fit = quantile_custom.fit(X, y)
                # acc = jnp.mean(y <= quantile_custom_fit.predict(X))
                result = quantile_custom_fit.predict(X)
                return result

            return proc
        
        n_samples, n_features = 100, 2
        # def generate_data():
        #     """
        #     Generate random data for testing.

        #     Returns:
        #     -------
        #     X : array-like, shape (n_samples, n_features)
        #         Feature data.
        #     y : array-like, shape (n_samples,)
        #         Target data.
        #     coef : array-like, shape (n_features + 1,)
        #         True coefficients, including the intercept term and feature weights.

        #     """
        #     np.random.seed(42)
        #     X = np.random.rand(n_samples, n_features)
        #     coef = np.random.rand(n_features + 1)  # +1 for the intercept term
        #     y = X @ coef[1:] + coef[0]
        #     sample_weight = np.random.rand(n_samples)
        #     return X, y, coef, sample_weight
        
        def generate_data():
            from jax import random
            # 设置随机种子
            key = random.PRNGKey(42)
            # 生成 X 数据
            key, subkey = random.split(key)
            X = random.normal(subkey, (100, 2)) 
            # 生成 y 数据
            y = 5 * X[:, 0] + 2 * X[:, 1] + random.normal(key, (100,)) * 0.1  # 高相关性，带有小噪声
            return X, y
        
        # bandwidth and latency only work for docker mode
        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64
        )
        
        # X, y, coef, sample_weight = generate_data()
        X, y = generate_data()
        
        # compare with sklearn
        quantile_sklearn = SklearnQuantileRegressor(quantile=0.7, alpha=0.1, fit_intercept=True, solver='highs')
        quantile_sklearn_fit = quantile_sklearn.fit(X, y)
        acc_sklearn = jnp.mean(y <= quantile_sklearn_fit.predict(X))
        print(f"Accuracy in SKlearn: {acc_sklearn:.2f}")
        
        # run
        proc = proc_wrapper(quantile=0.7, alpha=0.1, fit_intercept=True, lr=0.01, max_iter=1000)
        result = spsim.sim_jax(sim, proc)(X, y)
        acc_custom = jnp.mean(y <= result)
        
        # print acc
        
        print(f"Accuracy in SPU: {acc_custom:.2f}")
        
if __name__ == "__main__":
    unittest.main()
        
