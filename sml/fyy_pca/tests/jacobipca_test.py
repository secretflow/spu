import os
import sys
import unittest

import jax.numpy as jnp
import numpy as np
from jax import random
from sklearn.decomposition import PCA as SklearnPCA

import spu.libspu as libspu
import spu.utils.simulation as spsim

# Add the sml directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from sml.fyy_pca.jacobi_pca import PCA

np.random.seed(0)


class UnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(" ========= start test of pca package ========= \n")

        # 1. init sim
        config64 = libspu.RuntimeConfig(
            protocol=libspu.ProtocolKind.ABY3,
            field=libspu.FieldType.FM64,
            # enable_pphlo_profile=True,
        )
        cls.sim64 = spsim.Simulator(3, config64)
        config128 = libspu.RuntimeConfig(
            protocol=libspu.ProtocolKind.ABY3,
            field=libspu.FieldType.FM128,
            fxp_fraction_bits=30,
            # enable_pphlo_profile = True
        )
        cls.sim128 = spsim.Simulator(3, config128)

    def test_power(self):
        print("start test power method.")

        # Test fit_transform
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

        # Run the simulation
        result = spsim.sim_jax(self.sim64, proc_transform)(X)

        # The transformed data should have 2 dimensions
        self.assertEqual(result[0].shape[1], 2)

        # The mean of the transformed data should be approximately 0
        self.assertTrue(jnp.allclose(jnp.mean(result[0], axis=0), 0, atol=1e-3))

        X_np = np.array(X)

        # Run fit_transform using sklearn
        sklearn_pca = SklearnPCA(n_components=2)
        X_transformed_sklearn = sklearn_pca.fit_transform(X_np)

        # Compare the transform results(omit sign)
        np.testing.assert_allclose(
            np.abs(X_transformed_sklearn), np.abs(result[0]), rtol=0.1, atol=0.1
        )

        # Compare the variance results
        np.testing.assert_allclose(
            sklearn_pca.explained_variance_, result[1], rtol=0.1, atol=0.1
        )

        # Run inverse_transform using sklearn
        X_reconstructed_sklearn = sklearn_pca.inverse_transform(X_transformed_sklearn)

        # Compare the results
        np.testing.assert_allclose(
            X_reconstructed_sklearn, result[2], atol=0.01, rtol=0.01
        )

        abs_diff = np.abs(np.abs(X_transformed_sklearn) - np.abs(result[0]))
        rel_error = abs_diff / (np.abs(X_transformed_sklearn) + 1e-10)

        print("relative error:\n", rel_error)
        print("avg absolute error:\n", np.mean(abs_diff))
        print("avg relative error:\n", np.mean(rel_error))

    def test_jacobi(self):
        print("start test serial_jacobi method.")

        def proc_transform(X, rotate_matrix):
            model = PCA(
                method='serial_jacobi_iteration',
                n_components=4,
                rotate_matrix=rotate_matrix,
                max_jacobi_iter=5,
            )

            model.fit(X)
            X_transformed = model.transform(X)
            X_variances = model._variances
            X_reconstructed = model.inverse_transform(X_transformed)
            X_components = model._components

            return X_transformed, X_variances, X_reconstructed, X_components

        # Create a simple dataset
        X = random.normal(random.PRNGKey(0), (10, 20))

        # Create rotate_matrix
        rotate_matrix = jnp.eye(X.shape[1])

        # Run the simulation
        result = spsim.sim_jax(self.sim64, proc_transform)(X, rotate_matrix)

        # The transformed data should have 2 dimensions
        self.assertEqual(result[0].shape[1], 4)

        # The mean of the transformed data should be approximately 0
        self.assertTrue(jnp.allclose(jnp.mean(result[0], axis=0), 0, atol=1e-3))

        X_np = np.array(X)

        # Run fit_transform using sklearn
        sklearn_pca = SklearnPCA(n_components=4)
        X_transformed_sklearn = sklearn_pca.fit_transform(X_np)

        # Compare the transform results(omit sign)
        np.testing.assert_allclose(
            np.abs(X_transformed_sklearn), np.abs(result[0]), rtol=0.1, atol=0.1
        )

        # Compare the variance results
        np.testing.assert_allclose(
            sklearn_pca.explained_variance_, result[1], rtol=0.1, atol=0.1
        )

        # Run inverse_transform using sklearn
        X_reconstructed_sklearn = sklearn_pca.inverse_transform(X_transformed_sklearn)

        # Compare the results
        np.testing.assert_allclose(
            X_reconstructed_sklearn, result[2], atol=0.1, rtol=0.1
        )

        abs_diff = np.abs(np.abs(X_transformed_sklearn) - np.abs(result[0]))
        rel_error = abs_diff / (np.abs(X_transformed_sklearn) + 1e-10)

        print("relative error:\n", rel_error)
        print("avg absolute error:\n", np.mean(abs_diff))
        print("avg relative error:\n", np.mean(rel_error))

        # eigval , eigvec = np.linalg.eig(sklearn_pca.get_covariance())
        # sorted_indices = np.argsort(eigval)[::-1]
        # top_k_indices = sorted_indices[:4]
        # X_components_np = (eigvec.T[top_k_indices]).T

        # abs_diff = np.abs(np.abs(X_components_np) - np.abs(result[3]))
        # rel_error = abs_diff / (np.abs(X_components_np) + 1e-10)

        # print(X_components_np)
        # print(result[3])

        # print("relative error:\n",rel_error)
        # print("max relative error:\n",np.max(rel_error))
        # print("avg absolute error:\n",np.mean(abs_diff))
        # print("avg relative error:\n",np.mean(rel_error))


if __name__ == "__main__":
    unittest.main()
