# Copyright 2025 Ant Group Co., Ltd.
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
import time
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.manifold import spectral_embedding
from sklearn.neighbors import kneighbors_graph

import spu.libspu as libspu
import spu.utils.simulation as spsim
from sml.manifold.se import SE


class UnitTests(unittest.TestCase):
    def test_se(self):
        sim = spsim.Simulator.simple(3, libspu.ProtocolKind.ABY3, libspu.FieldType.FM64)

        def se(sX, num_samples, num_features, k, num_components):
            embedding = SE(
                n_components=num_components,
                n_neighbors=k,
                n_samples=num_samples,
                n_features=num_features,
                max_iterations=5,
            )
            X_transformed = embedding.fit_transform(sX)
            return X_transformed

        # Set sample size and dimensions
        num_samples = (
            6  # Number of samples, se can meet larger num_samples, such as 150
        )
        num_features = (
            4  # Sample dimension, se can meet larger num_features, such as 12
        )
        k = 5  # Number of nearest neighbors
        num_components = 3  # Dimension after dimensionality reduction

        # Generate random input
        seed = int(time.time())
        key = jax.random.PRNGKey(seed)
        X = jax.random.uniform(
            key, shape=(num_samples, num_features), minval=0.0, maxval=1.0
        )

        ans = spsim.sim_jax(
            sim,
            se,
            static_argnums=(
                1,
                2,
                3,
                4,
            ),
        )(X, num_samples, num_features, k, num_components)

        # sklearn test
        affinity_matrix = kneighbors_graph(
            X, n_neighbors=k, mode="distance", include_self=False
        )

        # Make the matrix symmetric
        affinity_matrix = 0.5 * (affinity_matrix + affinity_matrix.T)

        embedding = spectral_embedding(
            affinity_matrix, n_components=num_components, random_state=None
        )

        # Since the final calculation result is calculated by the eigenvector, the accuracy cannot reach 1e-3
        np.testing.assert_allclose(jnp.abs(embedding), jnp.abs(ans), rtol=0, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
