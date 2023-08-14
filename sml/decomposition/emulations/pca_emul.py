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


def emul_powerPCA(mode: emulation.Mode.MULTIPROCESS):
    def proc(X):
        model = PCA(
            method='power_iteration',
            n_components=2,
        )

        model.fit(X)
        X_transformed = model.transform(X)
        X_variances = model._variances

        return X_transformed, X_variances

    def proc_reconstruct(X):
        model = PCA(
            method='power_iteration',
            n_components=2,
        )

        model.fit(X)
        X_reconstructed = model.inverse_transform(model.transform(X))

        return X_reconstructed

    try:
        # bandwidth and latency only work for docker mode
        emulator = emulation.Emulator(
            emulation.CLUSTER_ABY3_3PC, mode, bandwidth=300, latency=20
        )
        emulator.up()
        # Create a simple dataset
        X = random.normal(random.PRNGKey(0), (15, 100))
        result = emulator.run(proc)(X)
        print("X_transformed_jax: ", result[0])
        print("X_transformed_jax: ", result[1])
        # The transformed data should have 2 dimensions
        assert result[0].shape[1] == 2
        # The mean of the transformed data should be approximately 0
        assert jnp.allclose(jnp.mean(result[0], axis=0), 0, atol=1e-3)

        # Compare with sklearn
        model = SklearnPCA(n_components=2)
        model.fit(X)
        X_transformed = model.transform(X)
        X_variances = model.explained_variance_

        print("X_transformed_sklearn: ", X_transformed)
        print("X_variances_sklearn: ", X_variances)

        result = emulator.run(proc_reconstruct)(X)

        print("X_reconstructed_jax: ", result)

        # Compare with sklearn
        model = SklearnPCA(n_components=2)
        model.fit(X)
        X_reconstructed = model.inverse_transform(model.transform(X))

        print("X_reconstructed_sklearn: ", X_reconstructed)

        assert np.allclose(X_reconstructed, result, atol=1e-3)

    finally:
        emulator.down()


def emul_rsvdPCA(mode: emulation.Mode.MULTIPROCESS):
    def proc(X, random_matrix):
        model = PCA(
            method='rsvd',
            n_components=5,
            n_oversamples=n_oversamples,
            random_matrix=random_matrix,
            scale=[10000000, 10000],
        )

        model.fit(X)
        X_transformed = model.transform(X)
        X_variances = model._variances

        return X_transformed, X_variances

    def proc_reconstruct(X, random_matrix):
        model = PCA(
            method='rsvd',
            n_components=5,
            n_oversamples=n_oversamples,
            random_matrix=random_matrix,
            scale=[10000000, 10000],
        )

        model.fit(X)
        X_reconstructed = model.inverse_transform(model.transform(X))

        return X_reconstructed

    try:
        # bandwidth and latency only work for docker mode
        emulator = emulation.Emulator(
            emulation.CLUSTER_ABY3_3PC, mode, bandwidth=300, latency=20
        )
        emulator.up()
        # Create a simple dataset
        X = random.normal(random.PRNGKey(0), (1000, 20))
        X_spu = emulator.seal(X)
        n_components = 5
        n_oversamples = 10

        # Create random_matrix
        random_state = np.random.RandomState(0)
        random_matrix = random_state.normal(
            size=(X.shape[1], n_components + n_oversamples)
        )
        random_matrix_spu = emulator.seal(random_matrix)

        result = emulator.run(proc)(X_spu, random_matrix_spu)
        print("X_transformed_jax: ", result[0])
        print("X_transformed_jax: ", result[1])

        # The transformed data should have 2 dimensions
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
        X_transformed = model.transform(X)
        X_variances = model.explained_variance_

        print("X_transformed_sklearn: ", X_transformed)
        print("X_variances_sklearn: ", X_variances)

        result = emulator.run(proc_reconstruct)(X_spu, random_matrix_spu)

        print("X_reconstructed_jax: ", result)

        # Compare with sklearn
        model = SklearnPCA(n_components=n_components)
        model.fit(X)
        X_reconstructed = model.inverse_transform(model.transform(X))

        print("X_reconstructed_sklearn: ", X_reconstructed)

        assert np.allclose(X_reconstructed, result, atol=1e-1)

    finally:
        emulator.down()


if __name__ == "__main__":
    emul_powerPCA(emulation.Mode.MULTIPROCESS)
    emul_rsvdPCA(emulation.Mode.MULTIPROCESS)
