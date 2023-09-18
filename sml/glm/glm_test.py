import unittest
from jax import random
import jax.numpy as jnp
import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as spsim
from glm import (
    _GeneralizedLinearRegressor,
    PoissonRegressor,
    GammaRegressor,
    TweedieRegressor,
)
import numpy as np
import scipy.stats as stats
from sklearn.linear_model._glm import (
    _GeneralizedLinearRegressor as std__GeneralizedLinearRegressor,
)
from sklearn.linear_model._glm import PoissonRegressor as std_PoissonRegressor
from sklearn.linear_model._glm import GammaRegressor as std_GammaRegressor
from sklearn.linear_model._glm import TweedieRegressor as std_TweedieRegressor

verbose = 0
n_samples, n_features = 100, 5


def generate_data(noise=False):
    """
    Generate random data for testing.

    Parameters:
    ----------
    noise : bool, optional (default=False)
        Whether to add noise.

    Returns:
    -------
    X : array-like, shape (n_samples, n_features)
        Feature data.
    y : array-like, shape (n_samples,)
        Target data.
    coef : array-like, shape (n_features + 1,)
        True coefficients, including the intercept term and feature weights.

    """
    np.random.seed(42)
    X = np.random.rand(n_samples, n_features)
    coef = np.random.rand(n_features + 1)  # +1 for the intercept term
    y = X @ coef[1:] + coef[0]
    if noise:
        noise = np.random.normal(loc=0, scale=0.05, size=num_samples)
        y += noise
    sample_weight = np.random.rand(n_samples)
    return X, y, coef, sample_weight


X, y, coef, sample_weight = generate_data()
exp_y = jnp.exp(y)
round_exp_y = jnp.round(exp_y)
sim = spsim.Simulator.simple(3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM128)


def accuracy_test(model, std_model, y, coef, num=5):
    """
    Test the fitting, prediction, and scoring functionality of the generalized linear regression model.

    Parameters:
    ----------
    model : object
        Generalized linear regression model object.
    X : array-like, shape (n_samples, n_features)
        Feature data.
    y : array-like, shape (n_samples,)
        Target data.
    coef : array-like, shape (n_features + 1,)
        True coefficients, including the intercept term and feature weights.
    num : int, optional (default=5)
        Number of coefficients to display.

    Returns:
    -------
    None

    """
    model.fit(X, y, sample_weight)
    std_model.fit(X, y, sample_weight)
    norm_diff = jnp.linalg.norm(
        model.predict(X)[:num] - jnp.array(std_model.predict(X)[:num])
    )
    if verbose:
        print('True Coefficients:', coef[:num])
        print("Fitted Coefficients:", model.coef_[:num])
        print("std Fitted Coefficients:", std_model.coef_[:num])
        print("D^2 Score:", model.score(X[:num], y[:num]))
        print("X:", X[:num])
        print("Samples:", y[:num])
        print("Predictions:", model.predict(X[:num]))
        print("std Predictions:", std_model.predict(X[:num]))
        print("norm of predict between ours and std: %f" % norm_diff)
        print("_________________________________")
        print()
    assert norm_diff < 1e-2


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
    if verbose:
        print(proc.__name__, "-norm_diff:", "%.5f" % norm_diff)

    # Assert that the difference is within the tolerance
    assert norm_diff < 1e-4


def proc_ncSolver():
    """
    Fit Generalized Linear Regression model using Newton-Cholesky algorithm and return the model coefficients.

    Returns:
    -------
    array-like, shape (n_features + 1,)
        Model coefficients, including the intercept term and feature weights.

    """
    model = _GeneralizedLinearRegressor(solver="newton-cholesky")
    model.fit(X, y)
    return model.coef_


def proc_lbfgsSolver():
    """
    Fit Generalized Linear Regression model using Newton-Cholesky algorithm and return the model coefficients.

    Returns:
    -------
    array-like, shape (n_features + 1,)
        Model coefficients, including the intercept term and feature weights.

    """
    model = _GeneralizedLinearRegressor(solver="lbfgs")
    model.fit(X, y)
    return model.coef_


def proc_Poisson():
    """
    Fit Generalized Linear Regression model using PoissonRegressor and return the model coefficients.

    Returns:
    -------
    array-like, shape (n_features + 1,)
        Model coefficients, including the intercept term and feature weights.

    """
    model = PoissonRegressor()
    model.fit(X, round_exp_y)
    return model.coef_


def proc_Gamma():
    """
    Fit Generalized Linear Regression model using GammaRegressor and return the model coefficients.

    Returns:
    -------
    array-like, shape (n_features + 1,)
        Model coefficients, including the intercept term and feature weights.

    """
    model = GammaRegressor()
    model.fit(X, exp_y)
    return model.coef_


def proc_Tweedie():
    """
    Fit Generalized Linear Regression model using TweedieRegressor and return the model coefficients.

    Returns:
    -------
    array-like, shape (n_features + 1,)
        Model coefficients, including the intercept term and feature weights.

    """
    model = TweedieRegressor()
    model.fit(X, exp_y)
    return model.coef_


class TestGeneralizedLinearRegressor(unittest.TestCase):
    def test_ncSolver_accuracy(self):
        # Test the accuracy of the Generalized Linear Regression model using Newton-Cholesky solver
        model = _GeneralizedLinearRegressor()
        std_model = std__GeneralizedLinearRegressor(alpha=0)
        accuracy_test(model, std_model, exp_y, coef)
        print('test_ncSolver_accuracy: OK')

    def test_Poisson_accuracy(self):
        # Test the accuracy of the PoissonRegressor model
        model = PoissonRegressor()
        std_model = PoissonRegressor(alpha=0)
        accuracy_test(model, std_model, round_exp_y, coef)
        print('test_Poisson_accuracy: OK')

    def test_gamma_accuracy(self):
        # Test the accuracy of the GammaRegressor model
        model = GammaRegressor()
        std_model = std_GammaRegressor(alpha=0)
        accuracy_test(model, std_model, exp_y, coef)
        print('test_gamma_accuracy: OK')

    def test_Tweedie_accuracy(self, power=0):
        # Test the accuracy of the TweedieRegressor model
        model = TweedieRegressor(power=power)
        std_model = std_TweedieRegressor(alpha=0, power=power)
        accuracy_test(model, std_model, exp_y, coef)
        print('test_Tweedie_accuracy: OK')

    def test_ncSolver_encrypted(self):
        # Test if the results of the Newton-Cholesky solver are correct after encryption
        proc_test(proc_ncSolver)
        print('test_ncSolver_encrypted: OK')

    def test_lbfgsSolver_encrypted(self):
        # Test if the results of the LBFGS solver are correct after encryption
        proc_test(proc_lbfgsSolver)
        print('test_lbfgsSolver_encrypted: OK')

    def test_Poisson_encrypted(self):
        # Test if the results of the PoissonRegressor model are correct after encryption
        proc_test(proc_Poisson)
        print('test_Poisson_encrypted: OK')

    def test_gamma_encrypted(self):
        # Test if the results of the GammaRegressor model are correct after encryption
        proc_test(proc_Gamma)
        print('test_gamma_encrypted: OK')

    def test_Tweedie_encrypted(self, power=0):
        # Test if the results of the TweedieRegressor model are correct after encryption
        proc_test(proc_Tweedie)
        print('test_Tweedie_encrypted: OK')


if __name__ == '__main__':
    # Run the unit tests
    unittest.main()
