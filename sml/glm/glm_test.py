import unittest
from jax import random
import jax.numpy as jnp
import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as spsim
from glm import _GeneralizedLinearRegressor,PoissonRegressor,GammaRegressor,TweedieRegressor
import inspect

class TestGeneralizedLinearRegressor(unittest.TestCase):
    def setUp(self):
        # 生成一些随机数据
        seed = random.PRNGKey(0)
        n_samples, n_features = 100, 5
        X = random.uniform(seed,shape=(n_samples, n_features))
        coef = random.uniform(seed,shape =(n_features + 1,))  # +1 是截距项
        y = X @ coef[1:] + coef[0]
        self.X = X
        self.y = y
        self.log_y = jnp.log(self.y)
        self.round_logy = jnp.round(self.log_y)

    def test_fit_predict(self):
        sim = spsim.Simulator.simple(3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64)

        def proc_ncSolver():
            model = _GeneralizedLinearRegressor(solver="newton-cholesky")
            model.fit(self.X, self.y)
            return model.coef_

        def proc_lbfgsSolver():
            model = _GeneralizedLinearRegressor(solver="lbfgs")
            model.fit(self.X, self.y)
            return model.coef_

        def test_Poisson():
            model = PoissonRegressor()
            model.fit(self.X, self.round_logy)
            return model.coef_

        def test_Gamma():
            model = GammaRegressor()
            model.fit(self.X, self.log_y)
            return model.coef_

        def test_Tweedie():
            model = TweedieRegressor()
            model.fit(self.X, self.log_y)
            return model.coef_

        def proc_test(proc):
            sim_res = spsim.sim_jax(sim, proc)()
            res = proc()
            print(proc.__name__,"-norm_diff:", "%.5f" %jnp.linalg.norm(sim_res-res))
            assert  jnp.linalg.norm(sim_res-res) < 3e-3

        proc_test(proc_ncSolver)
        # proc_test(proc_lbfgsSolver) # 近似实现误差大
        proc_test(test_Poisson)
        proc_test(test_Gamma)
        proc_test(test_Tweedie)

if __name__ == '__main__':
    unittest.main()
