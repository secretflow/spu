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
import numpy as np
from sklearn.decomposition import PCA as SklearnPCA

# Add the library directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

import sml.utils.emulation as emulation
from sml.decomposition.pca import PCA

np.random.seed(0)


def emul_rsvdPCA(mode: emulation.Mode.MULTIPROCESS):
    print("emul rsvdPCA.")

    def proc(X, random_matrix, n_components, n_oversamples, max_power_iter, scale):
        model = PCA(
            method='rsvd',
            n_components=n_components,
            n_oversamples=n_oversamples,
            random_matrix=random_matrix,
            max_power_iter=max_power_iter,
            scale=scale,
        )

        model.fit(X)
        X_transformed = model.transform(X)
        X_variances = model._variances
        X_reconstructed = model.inverse_transform(X_transformed)

        return X_transformed, X_variances, X_reconstructed

    try:
        # bandwidth and latency only work for docker mode
        conf_path = "sml/decomposition/emulations/3pc.json"
        emulator = emulation.Emulator(emulation.CLUSTER_FANTASTIC4_4PC, mode, bandwidth=300, latency=20)
        emulator.up()

        # Create a simple dataset
        X = np.random.normal(size=(50, 20))
        X_spu = emulator.seal(X)
        n_components = 1
        n_oversamples = 10
        max_power_iter = 100
        scale = (10000000, 10000)

        # Create random_matrix
        random_matrix = np.random.normal(
            size=(X.shape[1], n_components + n_oversamples)
        )
        random_matrix_spu = emulator.seal(random_matrix)

        result = emulator.run(proc, static_argnums=(2, 3, 4, 5))(
            X_spu, random_matrix_spu, n_components, n_oversamples, max_power_iter, scale
        )

        # The transformed data should have n_components dimensions
        assert result[0].shape[1] == n_components

        # The mean of the transformed data should be approximately 0
        assert jnp.allclose(jnp.mean(result[0], axis=0), 0, atol=1e-3)

        # Compare with sklearn
        model = SklearnPCA(
            n_components=n_components,
            svd_solver="randomized",
            power_iteration_normalizer="QR",
            random_state=0,
        )
        model.fit(X)
        X_transformed_sklearn = model.transform(X)
        X_variances = model.explained_variance_
        X_reconstructed = model.inverse_transform(X_transformed_sklearn)

        # Compare the transform results(omit sign)
        np.testing.assert_allclose(
            np.abs(X_transformed_sklearn), np.abs(result[0]), rtol=1, atol=0.1
        )

        # Compare the variance results
        np.testing.assert_allclose(X_variances, result[1], rtol=1, atol=0.1)

        assert np.allclose(X_reconstructed, result[2], atol=1e-1)

    finally:
        emulator.down()


if __name__ == "__main__":
    emul_rsvdPCA(emulation.Mode.MULTIPROCESS)
