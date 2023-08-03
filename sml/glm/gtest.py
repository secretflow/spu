from glm import *
import numpy as np
import scipy.stats as stats
import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as spsim
from sklearn.linear_model._glm import _GeneralizedLinearRegressor as std__GeneralizedLinearRegressor
from sklearn.linear_model._glm import PoissonRegressor  as std_PoissonRegressor
from sklearn.linear_model._glm import GammaRegressor as std_GammaRegressor
from sklearn.linear_model._glm import TweedieRegressor as std_TweedieRegressor
import unittest

n_samples, n_features = 100, 5
verbose = 0
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
expy = jnp.exp(y)
roundexpy = jnp.round(expy)

def test(model,std_model, y, coef, num=5):
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
        print(type(model).__name__)
        model.fit(X, y, sample_weight)
        std_model.fit(X,y,sample_weight)
        norm_diff = jnp.linalg.norm(model.predict(X)[:num]-jnp.array(std_model.predict(X)[:num]))
        if verbose:
            print('True Coefficients:', coef[:num])
            print("Fitted Coefficients:", model.coef_[:num])
            print("std Fitted Coefficients:", std_model.coef_[:num])
            print("D^2 Score:", model.score(X[:num], y[:num]))
            print("X:", X[:num])
            print("Samples:", y[:num])
            print("Predictions:", model.predict(X[:num]))
            print("std Predictions:", std_model.predict(X[:num]))
        print("norm of predict between ours and std: %f" %norm_diff)
        assert norm_diff < 1e-2
        print("_________________________________")
        print()

class GeneralizedLinearRegressorCorrectnessTest(unittest.TestCase):
    def test_glm(self,):
        """
        Test the functionality of the _GeneralizedLinearRegressor model.

        """
        from glm import _GeneralizedLinearRegressor
        model = _GeneralizedLinearRegressor()
        std_model = std__GeneralizedLinearRegressor(alpha=0)
        test(model, std_model, y, coef)


    def test_Poisson(self,):
        """
        Test the functionality of the PoissonRegressor model.

        """
        model = PoissonRegressor()
        std_model = PoissonRegressor(alpha=0)
        test(model, std_model, roundexpy, coef)


    def test_gamma(self,):
        """
        Test the functionality of the GammaRegressor model.

        """
        model = GammaRegressor()
        std_model = std_GammaRegressor(alpha=0)
        test(model, std_model, expy, coef)


    def test_Tweedie(self,power=0):
        """
        Test the functionality of the TweedieRegressor model.

        """
        model = TweedieRegressor(power=power)
        std_model = std_TweedieRegressor(alpha=0,power=power)
        test(model, std_model, expy, coef)



if __name__ == '__main__':
    # Run the tests
    # test_glm()
    # test_gamma()
    # test_Poisson()
    # test_Tweedie()
    unittest.main()
