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
import numpy as np
from sml.linear_model.glm import _GeneralizedLinearRegressor

import sml.utils.emulation as emulation
import spu.utils.distributed as ppd

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
        X, y, coef, sample_weight = generate_data()
        # Create the emulator with specified mode and bandwidth/latency settings
        emulator = emulation.Emulator(
            emulation.CLUSTER_ABY3_3PC, mode, bandwidth=300, latency=20
        )
        emulator.up()

        # Run the proc_ncSolver function using both plaintext and encrypted data
        raw_score, raw_result = proc_ncSolver(X, y)

        X, y = emulator.seal(X, y)
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
