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

import os
import sys

import jax.numpy as jnp
import jax.random as random
import numpy as np
from sklearn.decomposition import PCA as SklearnPCA

# Add the library directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

import sml.utils.emulation as emulation
from sml.decomposition.pca import PCA


def test_pca(mode: emulation.Mode.MULTIPROCESS):
    def emul_powerPCA():
        print("start power method emulation.")

        def proc_transform(X):
            model = PCA(
                method='power_iteration',
                n_components=2,
                max_power_iter=200,
            )

            model.fit(X)
            X_transformed = model.transform(X)
            X_variances = model._variances
            X_reconstructed = model.inverse_transform(X_transformed)

            return X_transformed, X_variances, X_reconstructed

        # Create a simple dataset
        X = random.normal(random.PRNGKey(0), (10, 20))
        X_spu = emulator.seal(X)
        result = emulator.run(proc_transform)(X_spu)

        # The transformed data should have 2 dimensions
        assert result[0].shape[1] == 2
        # The mean of the transformed data should be approximately 0
        assert jnp.allclose(jnp.mean(result[0], axis=0), 0, atol=1e-3)

        # Compare with sklearn
        model = SklearnPCA(n_components=2)
        model.fit(X)
        X_transformed_sklearn = model.transform(X)
        X_variances = model.explained_variance_

        # Compare the transform results(omit sign)
        np.testing.assert_allclose(
            np.abs(X_transformed_sklearn), np.abs(result[0]), rtol=0.1, atol=0.1
        )

        # Compare the variance results
        np.testing.assert_allclose(X_variances, result[1], rtol=0.1, atol=0.1)

        X_reconstructed = model.inverse_transform(X_transformed_sklearn)

        np.testing.assert_allclose(X_reconstructed, result[2], atol=1e-3)

    def emul_jacobi_PCA():
        print("start jacobi method emulation.")

        def proc_transform(X):
            model = PCA(
                method='serial_jacobi_iteration',
                n_components=4,
                max_jacobi_iter=5,
            )

            model.fit(X)
            X_transformed = model.transform(X)
            X_variances = model._variances
            X_reconstructed = model.inverse_transform(X_transformed)

            return X_transformed, X_variances, X_reconstructed

        # Create a simple dataset
        X = random.normal(random.PRNGKey(0), (10, 20))
        X_spu = emulator.seal(X)
        result = emulator.run(proc_transform)(X_spu)

        # The mean of the transformed data should be approximately 0
        assert jnp.allclose(jnp.mean(result[0], axis=0), 0, atol=1e-3)

        # Compare with sklearn
        model = SklearnPCA(n_components=4)
        model.fit(X)
        X_transformed_sklearn = model.transform(X)
        X_variances = model.explained_variance_

        # Compare the transform results(omit sign)
        np.testing.assert_allclose(
            np.abs(X_transformed_sklearn), np.abs(result[0]), rtol=0.1, atol=0.1
        )

        # Compare the variance results
        np.testing.assert_allclose(X_variances, result[1], rtol=0.1, atol=0.1)

        X_reconstructed = model.inverse_transform(X_transformed_sklearn)

        np.testing.assert_allclose(X_reconstructed, result[2], atol=0.1)

    try:
        # bandwidth and latency only work for docker mode
        emulator = emulation.Emulator(
            emulation.CLUSTER_ABY3_3PC, mode, bandwidth=300, latency=20
        )
        emulator.up()
        emul_powerPCA()
        emul_jacobi_PCA()
    finally:
        emulator.down()


if __name__ == "__main__":
    test_pca(emulation.Mode.MULTIPROCESS)
