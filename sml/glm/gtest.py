from glm import *
import numpy as np
import scipy.stats as stats
import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as spsim

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

def test(model, X, y, coef, sample_weight=None, num=5):
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
    print('True Coefficients:', coef[:num])
    print("Fitted Coefficients:", model.coef_[:num])
    print("D^2 Score:", model.score(X[:num], y[:num]))
    print("X:", X[:num])
    print("Samples:", y[:num])
    print("Predictions:", model.predict(X[:num]))

def test_glm():
    """
    Test the functionality of the _GeneralizedLinearRegressor model.

    """
    X, y, coef, sample_weight = generate_data()
    from glm import _GeneralizedLinearRegressor
    model = _GeneralizedLinearRegressor()
    test(model, X, y, coef, sample_weight)

def test_lbfgs():
    """
    Test the functionality of the _GeneralizedLinearRegressor model using the LBFGS optimization algorithm.

    """
    X, y, coef, sample_weight = generate_data()
    from glm import _GeneralizedLinearRegressor
    model = _GeneralizedLinearRegressor(solver='lbfgs')
    test(model, X, y, coef, sample_weight)

def test_Poisson():
    """
    Test the functionality of the PoissonRegressor model.

    """
    X, y, coef, sample_weight = generate_data()
    y = jnp.round(jnp.exp(y))
    model = PoissonRegressor()
    test(model, X, y, coef, sample_weight)

def test_gamma():
    """
    Test the functionality of the GammaRegressor model.

    """
    X, y, coef, sample_weight = generate_data()
    alpha = 10
    y = np.array([stats.gamma.rvs(a=alpha, scale=y_i/alpha) for y_i in y])
    y = jnp.exp(y)
    model = GammaRegressor()
    test(model, X, y, coef, sample_weight)

def test_Tweedie():
    """
    Test the functionality of the TweedieRegressor model.

    """
    X, y, coef, sample_weight = generate_data()
    y = jnp.round(jnp.exp(y))
    model = TweedieRegressor()
    test(model, X, y, coef, sample_weight)

def test_sim():
    """
    Test the fitting functionality of the _GeneralizedLinearRegressor model using the simulator.

    """
    sim = spsim.Simulator.simple(
            # 3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM128)
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64)
    X, y, coef, sample_weight = generate_data()

    def proc(X, y):
        from glm import _GeneralizedLinearRegressor
        model = _GeneralizedLinearRegressor(solver="newton-cholesky")
        model.fit(X, y)
        coef_fit = model.coef_
        return coef_fit

    result = spsim.sim_jax(sim, proc)(X, y)

if __name__ == '__main__':
    # Run the tests
    test_glm()
    test_gamma()
    test_lbfgs()
    test_Poisson()
    test_Tweedie()
    test_sim()
