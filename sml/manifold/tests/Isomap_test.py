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
import unittest

import jax
import jax.numpy as jnp
from sklearn.manifold import Isomap

import spu.libspu as libspu
import spu.utils.simulation as spsim
from sml.manifold.floyd import floyd_opt
from sml.manifold.kneighbors import mpc_kneighbors_graph
from sml.manifold.MDS import mds

# Add the sml directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

# from sml.neighbors.knn import KNNClassifer


class UnitTests(unittest.TestCase):
    def test_knn(self):
        sim = spsim.Simulator.simple(3, libspu.ProtocolKind.ABY3, libspu.FieldType.FM64)

        def mpc_isomap_floyd(
            sX,
            mpc_shortest_paths,
            num_samples,
            num_features,
            k,
            num_components,
        ):

            Knn = mpc_kneighbors_graph(sX, num_samples, num_features, k)

            Knn = (Knn == 0) * jnp.inf + Knn
            Knn = jnp.where(jnp.eye(Knn.shape[0]), 0, Knn)
            flag = Knn <= Knn.T
            Knn = flag * Knn + (1 - flag) * Knn.T

            mpc_shortest_paths = floyd_opt(Knn)

            B, ans, values, vectors = mds(
                mpc_shortest_paths, num_samples, num_components
            )
            return Knn, mpc_shortest_paths, B, ans, values, vectors

        # Set sample size and dimensions
        num_samples = 5  # Number of samples
        num_features = 4  # Sample dimension
        k = 3  # Number of nearest neighbors
        num_components = 2  # Dimension after dimensionality reduction

        # Generate random input
        # seed = int(time.time())
        seed = 5
        key = jax.random.PRNGKey(seed)
        X = jax.random.uniform(
            key, shape=(num_samples, num_features), minval=0.0, maxval=1.0
        )

        shortest_paths = jnp.zeros((num_samples, num_samples))

        # 运行模拟器
        Knn, mpc_shortest_paths, B, ans, values, vectors = spsim.sim_jax(
            sim, mpc_isomap_floyd, static_argnums=(2, 3, 4, 5)
        )(
            X,
            shortest_paths,
            num_samples,
            num_features,
            k,
            num_components,
        )

        # sklearn test
        embedding = Isomap(n_components=num_components, n_neighbors=k)
        X_transformed = embedding.fit_transform(X)
        print('X_transformed: \n', X_transformed)

        # Calculate the maximum difference between the results of SE and sklearn test, i.e. accuracy
        max_abs_diff = jnp.max(jnp.abs(jnp.abs(X_transformed) - jnp.abs(ans)))
        print(max_abs_diff)


if __name__ == "__main__":
    unittest.main()
