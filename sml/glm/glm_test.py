import unittest
from jax import random
import jax.numpy as jnp
import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as spsim
from glm import _GeneralizedLinearRegressor, PoissonRegressor, GammaRegressor, TweedieRegressor
import inspect

class TestGeneralizedLinearRegressor(unittest.TestCase):
    """
    测试广义线性回归模型的拟合和预测功能。

    """

    def setUp(self):
        """
        设置测试所需的随机数据。

        """
        # Generate some random data
        seed = random.PRNGKey(0)
        n_samples, n_features = 100, 5
        X = random.uniform(seed, shape=(n_samples, n_features))
        coef = random.uniform(seed, shape=(n_features + 1,))  # +1 是截距项
        y = X @ coef[1:] + coef[0]
        self.X = X
        self.y = y
        self.log_y = jnp.log(self.y)
        self.round_logy = jnp.round(self.log_y)

    def test_fit_predict(self):
        """
        测试不同优化算法的广义线性回归模型的拟合和预测功能。

        """
        # Set up the simulator
        sim = spsim.Simulator.simple(3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64)

        def proc_ncSolver():
            """
            使用Newton-Cholesky算法拟合广义线性回归模型，并返回模型的系数。

            Returns:
            -------
            array-like, shape (n_features + 1,)
                模型的系数，包括截距项和特征权重。

            """
            model = _GeneralizedLinearRegressor(solver="newton-cholesky")
            model.fit(self.X, self.y)
            return model.coef_

        def test_Poisson():
            """
            使用PoissonRegressor拟合广义线性回归模型，并返回模型的系数。

            Returns:
            -------
            array-like, shape (n_features + 1,)
                模型的系数，包括截距项和特征权重。

            """
            model = PoissonRegressor()
            model.fit(self.X, self.round_logy)
            return model.coef_

        def test_Gamma():
            """
            使用GammaRegressor拟合广义线性回归模型，并返回模型的系数。

            Returns:
            -------
            array-like, shape (n_features + 1,)
                模型的系数，包括截距项和特征权重。

            """
            model = GammaRegressor()
            model.fit(self.X, self.log_y)
            return model.coef_

        def test_Tweedie():
            """
            使用TweedieRegressor拟合广义线性回归模型，并返回模型的系数。

            Returns:
            -------
            array-like, shape (n_features + 1,)
                模型的系数，包括截距项和特征权重。

            """
            model = TweedieRegressor()
            model.fit(self.X, self.log_y)
            return model.coef_

        def proc_test(proc):
            """
            测试指定的拟合算法的结果是否正确。

            Parameters:
            ----------
            proc : function
                拟合算法函数。

            Returns:
            -------
            None

            """
            # Run the simulation and get the results
            sim_res = spsim.sim_jax(sim, proc)()
            res = proc()

            # Calculate the difference between simulation and actual results
            norm_diff = jnp.linalg.norm(sim_res - res)
            print(proc.__name__, "-norm_diff:", "%.5f" % norm_diff)

            # Assert that the difference is within the tolerance
            assert norm_diff < 3e-3

        proc_test(proc_ncSolver)
        # proc_test(proc_lbfgsSolver) # Uncomment this line if needed for lbfgs solver
        proc_test(test_Poisson)
        proc_test(test_Gamma)
        proc_test(test_Tweedie)

if __name__ == '__main__':
    # Run the unit tests
    unittest.main()
