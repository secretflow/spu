# Copyright 2024 Ant Group Co., Ltd.
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
import time
import unittest

import jax
import jax.numpy as jnp
from sklearn.manifold import spectral_embedding
from sklearn.neighbors import kneighbors_graph

import spu.libspu as libspu
import spu.utils.simulation as spsim
from sml.manifold.kneighbors import mpc_kneighbors_graph
from sml.manifold.SE import normalization, se

# Add the sml directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

# from sml.neighbors.knn import KNNClassifer


class UnitTests(unittest.TestCase):
    def test_knn(self):
        sim = spsim.Simulator.simple(3, libspu.ProtocolKind.ABY3, libspu.FieldType.FM64)

        def SE(sX, num_samples, num_features, k, num_components):
            Knn = mpc_kneighbors_graph(sX, num_samples, num_features, k)
            Knn = 0.5 * (Knn + Knn.T)
            D, L = normalization(Knn)
            ans = se(L, num_samples, D, num_components)
            return ans

        # Set sample size and dimensions
        num_samples = 20  # Number of samples
        num_features = 4  # Sample dimension
        k = 15  # Number of nearest neighbors
        num_components = 3  # Dimension after dimensionality reduction

        # Generate random input
        seed = int(time.time())
        key = jax.random.PRNGKey(seed)
        X = jax.random.uniform(
            key, shape=(num_samples, num_features), minval=0.0, maxval=1.0
        )

        # 运行模拟器
        ans = spsim.sim_jax(
            sim,
            SE,
            static_argnums=(
                1,
                2,
                3,
                4,
            ),
        )(X, num_samples, num_features, k, num_components)

        print('ans: \n', ans)

        # sklearn test
        affinity_matrix = kneighbors_graph(
            X, n_neighbors=k, mode="distance", include_self=False
        )

        # Make the matrix symmetric
        affinity_matrix = 0.5 * (affinity_matrix + affinity_matrix.T)

        embedding = spectral_embedding(
            affinity_matrix, n_components=num_components, random_state=None
        )
        print('embedding: \n', embedding)

        # Calculate the maximum difference between the results of SE and sklearn test, i.e. accuracy
        max_abs_diff = jnp.max(jnp.abs(jnp.abs(embedding) - jnp.abs(ans)))
        print(max_abs_diff)


if __name__ == "__main__":
    unittest.main()
