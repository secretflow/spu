import os
import sys

import jax.numpy as jnp
import jax.random as random
import numpy as np
from sklearn.decomposition import PCA as SklearnPCA

# Add the library directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

import sml.utils.emulation as emulation
from sml.fyy_pca.jacobi_pca import PCA


def emul_powerPCA(mode: emulation.Mode.MULTIPROCESS):
    print("start power method emulation.")

    def proc_transform(X):
        model = PCA(
            method='power_iteration',
            n_components=6,
            max_power_iter=200,
        )

        model.fit(X)
        X_transformed = model.transform(X)
        X_variances = model._variances
        X_reconstructed = model.inverse_transform(X_transformed)

        return X_transformed, X_variances, X_reconstructed

    try:
        # bandwidth and latency only work for docker mode
        emulator = emulation.Emulator(
            emulation.CLUSTER_ABY3_3PC, mode, bandwidth=300, latency=20
        )
        emulator.up()

        # Create a simple dataset
        X = random.normal(random.PRNGKey(0), (10, 20))
        X_spu = emulator.seal(X)
        result = emulator.run(proc_transform)(X_spu)

        # # The transformed data should have 2 dimensions
        # assert result[0].shape[1] == 2
        # The mean of the transformed data should be approximately 0
        assert jnp.allclose(jnp.mean(result[0], axis=0), 0, atol=1e-3)

        # Compare with sklearn
        model = SklearnPCA(n_components=6)
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

    finally:
        emulator.down()


def emul_jacobi_PCA(mode: emulation.Mode.MULTIPROCESS):
    print("start jacobi method emulation.")

    def proc_transform(X, rotate_matrix):
        model = PCA(
            method='serial_jacobi_iteration',
            n_components=6,
            rotate_matrix=rotate_matrix,
            max_jacobi_iter=5,
        )

        model.fit(X)
        X_transformed = model.transform(X)
        X_variances = model._variances
        X_reconstructed = model.inverse_transform(X_transformed)

        return X_transformed, X_variances, X_reconstructed

    try:
        # bandwidth and latency only work for docker mode
        emulator = emulation.Emulator(
            emulation.CLUSTER_ABY3_3PC, mode, bandwidth=300, latency=20
        )
        emulator.up()

        # Create a simple dataset
        X = random.normal(random.PRNGKey(0), (10, 20))

        # Create rotate_matrix
        rotate_matrix = jnp.eye(X.shape[1])

        X_spu = emulator.seal(X)
        rotate_matrix_spu = emulator.seal(rotate_matrix)
        result = emulator.run(proc_transform)(X_spu, rotate_matrix_spu)

        # The mean of the transformed data should be approximately 0
        assert jnp.allclose(jnp.mean(result[0], axis=0), 0, atol=1e-3)

        # Compare with sklearn
        model = SklearnPCA(n_components=6)
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

    finally:
        emulator.down()


if __name__ == "__main__":
    # emul_powerPCA(emulation.Mode.MULTIPROCESS)
    emul_jacobi_PCA(emulation.Mode.MULTIPROCESS)
