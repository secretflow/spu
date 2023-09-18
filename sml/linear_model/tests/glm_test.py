# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import jax.numpy as jnp
import numpy as np
from sml.linear_model.glm import (
    GammaRegressor,
    PoissonRegressor,
    TweedieRegressor,
    _GeneralizedLinearRegressor,
)
from sklearn.linear_model._glm import GammaRegressor as std_GammaRegressor
from sklearn.linear_model._glm import PoissonRegressor as std_PoissonRegressor
from sklearn.linear_model._glm import TweedieRegressor as std_TweedieRegressor
from sklearn.linear_model._glm import (
    _GeneralizedLinearRegressor as std__GeneralizedLinearRegressor,
)

import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as spsim

verbose = 0
n_samples, n_features = 100, 5


def generate_data():
    """
    Generate random data for testing.

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
    model = _GeneralizedLinearRegressor(solver="newton-cholesky", max_iter=5)
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
    model = PoissonRegressor(max_iter=5)
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
    model = GammaRegressor(max_iter=5)
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
    model = TweedieRegressor(max_iter=5)
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
        std_model = std_PoissonRegressor(alpha=0)
        accuracy_test(model, std_model, round_exp_y, coef)
        print('test_Poisson_accuracy: OK')

    def test_gamma_accuracy(self):
        # Test the accuracy of the GammaRegressor model
        model = GammaRegressor()
        std_model = std_GammaRegressor(alpha=0)
        accuracy_test(model, std_model, exp_y, coef)
        print('test_gamma_accuracy: OK')

    def test_Tweedie_accuracy(self, power=1.5):
        # Test the accuracy of the TweedieRegressor model
        model = TweedieRegressor(power=power)
        std_model = std_TweedieRegressor(alpha=0, power=power)
        accuracy_test(model, std_model, exp_y, coef)
        print('test_Tweedie_accuracy: OK')

    def test_ncSolver_encrypted(self):
        # Test if the results of the Newton-Cholesky solver are correct after encryption
        proc_test(proc_ncSolver)
        print('test_ncSolver_encrypted: OK')

    def test_Poisson_encrypted(self):
        # Test if the results of the PoissonRegressor model are correct after encryption
        proc_test(proc_Poisson)
        print('test_Poisson_encrypted: OK')

    def test_gamma_encrypted(self):
        # Test if the results of the GammaRegressor model are correct after encryption
        proc_test(proc_Gamma)
        print('test_gamma_encrypted: OK')

    def test_Tweedie_encrypted(self):
        # Test if the results of the TweedieRegressor model are correct after encryption
        proc_test(proc_Tweedie)
        print('test_Tweedie_encrypted: OK')


if __name__ == '__main__':
    # Run the unit tests
    unittest.main()
