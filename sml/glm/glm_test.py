import unittest
from jax import random
import jax.numpy as jnp
import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as spsim
from glm import _GeneralizedLinearRegressor, PoissonRegressor, GammaRegressor, TweedieRegressor
import inspect

class TestGeneralizedLinearRegressor(unittest.TestCase):
    """
    Test the fitting and prediction functionality of Generalized Linear Regression models.

    """

    def setUp(self):
        """
        Set up random data for testing.

        """
        # Generate some random data
        seed = random.PRNGKey(0)
        n_samples, n_features = 100, 5
        X = random.uniform(seed, shape=(n_samples, n_features))
        coef = random.uniform(seed, shape=(n_features + 1,))  # +1 for the intercept term
        y = X @ coef[1:] + coef[0]
        self.X = X
        self.y = y
        self.log_y = jnp.log(self.y)
        self.round_logy = jnp.round(self.log_y)

    def test_fit_predict(self):
        """
        Test the fitting and prediction functionality of different optimization algorithms for Generalized Linear Regression models.

        """
        # Set up the simulator
        sim = spsim.Simulator.simple(3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64)

        def proc_ncSolver():
            """
            Fit Generalized Linear Regression model using Newton-Cholesky algorithm and return the model coefficients.

            Returns:
            -------
            array-like, shape (n_features + 1,)
                Model coefficients, including the intercept term and feature weights.

            """
            model = _GeneralizedLinearRegressor(solver="newton-cholesky")
            model.fit(self.X, self.y)
            return model.coef_

        def test_Poisson():
            """
            Fit Generalized Linear Regression model using PoissonRegressor and return the model coefficients.

            Returns:
            -------
            array-like, shape (n_features + 1,)
                Model coefficients, including the intercept term and feature weights.

            """
            model = PoissonRegressor()
            model.fit(self.X, self.round_logy)
            return model.coef_

        def test_Gamma():
            """
            Fit Generalized Linear Regression model using GammaRegressor and return the model coefficients.

            Returns:
            -------
            array-like, shape (n_features + 1,)
                Model coefficients, including the intercept term and feature weights.

            """
            model = GammaRegressor()
            model.fit(self.X, self.log_y)
            return model.coef_

        def test_Tweedie():
            """
            Fit Generalized Linear Regression model using TweedieRegressor and return the model coefficients.

            Returns:
            -------
            array-like, shape (n_features + 1,)
                Model coefficients, including the intercept term and feature weights.

            """
            model = TweedieRegressor()
            model.fit(self.X, self.log_y)
            return model.coef_

        def proc_test(proc):
            """
            Test if the results of the specified fitting algorithm are correct.

            Parameters:
            ----------
            proc : function
                Fitting algorithm function.

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
