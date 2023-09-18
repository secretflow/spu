import jax.numpy as jnp
import numpy as np
import sys
import os

# Add the parent directory to the system path to import custom modules
sys.path.append('../../')
import sml.utils.emulation as emulation
import spu.utils.distributed as ppd
from glm import (
    _GeneralizedLinearRegressor,
    PoissonRegressor,
    GammaRegressor,
    TweedieRegressor,
)

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


def emul_SGDClassifier(mode: emulation.Mode.MULTIPROCESS, num=10):
    """
    Execute the encrypted SGD classifier in a simulation environment and output the results.

    Parameters:
    -----------
    mode : emulation.Mode.MULTIPROCESS
        The running mode of the simulation environment, using multi-process mode.
    num : int, optional (default=5)
        The number of values to output.

    Returns:
    -------
    None
    """

    def proc_ncSolver(X, y):
        """
        Fit the generalized linear regression model using the Newton-Cholesky algorithm and calculate the D^2 evaluation metric and prediction results.

        Parameters:
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix 1.

        y : array-like, shape (n_samples,)
            Target values.

        Returns:
        -------
        float
            The result of the D^2 evaluation metric.
        array-like, shape (n_samples,)
            Model's prediction results.

        """
        model = _GeneralizedLinearRegressor(solver="newton-cholesky")
        model.fit(X, y)
        return model.score(X, y), model.predict(X)

    try:
        # Specify the file paths for cluster and dataset
        CLUSTER_ABY3_3PC = os.path.join('../../', emulation.CLUSTER_ABY3_3PC)
        # Create the emulator with specified mode and bandwidth/latency settings
        emulator = emulation.Emulator(CLUSTER_ABY3_3PC, mode, bandwidth=300, latency=20)
        emulator.up()

        # Run the proc_ncSolver function using both plaintext and encrypted data
        raw_score, raw_result = proc_ncSolver(ppd.get(X), ppd.get(y))
        score, result = emulator.run(proc_ncSolver)(X, y)

        # Print the results
        print("Plaintext D^2: %.2f" % raw_score)
        print("Plaintext Result (Top %s):" % num, raw_result[:num])
        print("Encrypted D^2: %.2f" % score)
        print("Encrypted Result (Top %s):" % num, result[:num])

    finally:
        emulator.down()


if __name__ == "__main__":
    # Run the emul_SGDClassifier function in MULTIPROCESS mode
    emul_SGDClassifier(emulation.Mode.MULTIPROCESS)
