import jax.numpy as jnp
import numpy as np
import sys
import os

# Add the parent directory to the system path to import custom modules
sys.path.append('../../')
import sml.utils.emulation as emulation
import spu.utils.distributed as ppd
from glm import _GeneralizedLinearRegressor, PoissonRegressor, GammaRegressor, TweedieRegressor


def emul_SGDClassifier(mode: emulation.Mode.MULTIPROCESS, num=5):
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

    def proc_ncSolver(x1, x2, y):
        """
        Fit the generalized linear regression model using the Newton-Cholesky algorithm and calculate the D^2 evaluation metric and prediction results.

        Parameters:
        ----------
        x1 : array-like, shape (n_samples, n_features1)
            Feature matrix 1.
        x2 : array-like, shape (n_samples, n_features2)
            Feature matrix 2.
        y : array-like, shape (n_samples,)
            Target values.

        Returns:
        -------
        float
            The result of the D^2 evaluation metric.
        array-like, shape (n_samples,)
            Model's prediction results.

        """
        X = jnp.concatenate((x1, x2), axis=1)
        model = _GeneralizedLinearRegressor(solver="newton-cholesky")
        model.fit(X, y)
        return model.score(X, y), model.predict(X)

    try:
        # Specify the file paths for cluster and dataset
        CLUSTER_ABY3_3PC = os.path.join('../../', emulation.CLUSTER_ABY3_3PC)
        DATASET_MOCK_REGRESSION_BASIC = os.path.join('../../', emulation.DATASET_MOCK_REGRESSION_BASIC)

        # Create the emulator with specified mode and bandwidth/latency settings
        emulator = emulation.Emulator(
            CLUSTER_ABY3_3PC, mode, bandwidth=300, latency=20
        )
        emulator.up()

        # Prepare the dataset using the emulator
        (x1, x2), y = emulator.prepare_dataset(DATASET_MOCK_REGRESSION_BASIC)

        # Run the proc_ncSolver function using both plaintext and encrypted data
        raw_score, raw_result = proc_ncSolver(ppd.get(x1), ppd.get(x2), ppd.get(y))
        score, result = emulator.run(proc_ncSolver)(x1, x2, y)

        # Print the results
        print("Plaintext D^2: %.2f" % raw_score)
        print("Plaintext Result (Top %s):" % num, jnp.round(raw_result[:num]))
        print("Encrypted D^2: %.2f" % score)
        print("Encrypted Result (Top %s):" % num, jnp.round(result[:num]))

    finally:
        emulator.down()


if __name__ == "__main__":
    # Run the emul_SGDClassifier function in MULTIPROCESS mode
    emul_SGDClassifier(emulation.Mode.MULTIPROCESS)
