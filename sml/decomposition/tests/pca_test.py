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
import unittest

import jax.numpy as jnp
import numpy as np
from jax import random
from sklearn.decomposition import PCA as SklearnPCA
import spu.spu_pb2 as spu_pb2  
import spu.utils.simulation as spsim

# Add the sml directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from sml.decomposition.pca import PCA


class UnitTests(unittest.TestCase):
    def test_power(self):
        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64
        )

        # Test fit_transform
        def proc_transform(X):
            model = PCA(
                method='power_iteration',
                n_components=2,
            )

            model.fit(X)
            X_transformed = model.transform(X)
            X_variances = model._variances

            return X_transformed, X_variances

        # Create a simple dataset
        X = random.normal(random.PRNGKey(0), (15, 100))

        # Run the simulation
        result = spsim.sim_jax(sim, proc_transform)(X)

        # The transformed data should have 2 dimensions
        self.assertEqual(result[0].shape[1], 2)

        # The mean of the transformed data should be approximately 0
        self.assertTrue(jnp.allclose(jnp.mean(result[0], axis=0), 0, atol=1e-3))

        X_np = np.array(X)

        # Run fit_transform using sklearn
        sklearn_pca = SklearnPCA(n_components=2)
        X_transformed_sklearn = sklearn_pca.fit_transform(X_np)

        # Compare the transform results
        print("X_transformed_sklearn: ", X_transformed_sklearn)
        print("X_transformed_jax", result[0])

        # Compare the variance results
        print(
            "X_transformed_sklearn.explained_variance_: ",
            sklearn_pca.explained_variance_,
        )
        print("X_transformed_jax.explained_variance_: ", result[1])

        # Test inverse_transform
        def proc_reconstruct(X):
            model = PCA(
                method='power_iteration',
                n_components=2,
            )

            model.fit(X)
            X_reconstructed = model.inverse_transform(model.transform(X))

            return X_reconstructed

        # Run the simulation
        result = spsim.sim_jax(sim, proc_reconstruct)(X)

        # Run inverse_transform using sklearn
        X_reconstructed_sklearn = sklearn_pca.inverse_transform(X_transformed_sklearn)

        # Compare the results
        self.assertTrue(np.allclose(X_reconstructed_sklearn, result, atol=1e-3))

    def test_rsvd(self):
        config = spu_pb2.RuntimeConfig(
            protocol=spu_pb2.ProtocolKind.ABY3,
            field=spu_pb2.FieldType.FM128,
            fxp_fraction_bits=30,
        )
        sim = spsim.Simulator(3, config)

        # Test fit_transform
        def proc_transform(X, random_matrix):
            model = PCA(
                method='rsvd',
                n_components=5,
                random_matrix=random_matrix,
            )

            model.fit(X)
            X_transformed = model.transform(X)
            X_variances = model._variances

            return X_transformed, X_variances

        # Create a simple dataset
        X = random.normal(random.PRNGKey(0), (1000, 10))

        # Create random_matrix
        random_state = np.random.RandomState(0)
        random_matrix = random_state.normal(
            size=(X.shape[1], 5)
        )
        
        # Run the simulation
        result = spsim.sim_jax(sim, proc_transform)(X, random_matrix)

        # The transformed data should have 2 dimensions
        self.assertEqual(result[0].shape[1], 5)

        # The mean of the transformed data should be approximately 0
        self.assertTrue(jnp.allclose(jnp.mean(result[0], axis=0), 0, atol=1e-3))

        X_np = np.array(X)

        # Run fit_transform using sklearn
        sklearn_pca = SklearnPCA(n_components=5)
        sklearn_pca.fit(X_np)
        X_transformed_sklearn = sklearn_pca.transform(X_np)

        # Compare the transform results
        print("X_transformed_sklearn: ", X_transformed_sklearn)
        print("X_transformed_jax", result[0])

        # Compare the variance results
        print(
            "X_transformed_sklearn.explained_variance_: ",
            sklearn_pca.explained_variance_,
        )
        print("X_transformed_jax.explained_variance_: ", result[1])

        # Test inverse_transform
        def proc_reconstruct(X, random_matrix):
            model = PCA(
                method='rsvd',
                n_components=5,
                random_matrix=random_matrix,
            )

            model.fit(X)
            X_reconstructed = model.inverse_transform(model.transform(X))

            return X_reconstructed

        # Run the simulation
        result = spsim.sim_jax(sim, proc_reconstruct)(X, random_matrix)

        # Run inverse_transform using sklearn
        X_reconstructed_sklearn = sklearn_pca.inverse_transform(X_transformed_sklearn)

        # Compare the results
        self.assertTrue(np.allclose(X_reconstructed_sklearn, result, atol=1e-1))


if __name__ == "__main__":
    unittest.main()
