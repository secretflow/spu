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
import numpy as np
from scipy.sparse.csgraph import shortest_path

import spu.libspu as libspu
import spu.utils.simulation as spsim
from sml.manifold.dijkstra import mpc_dijkstra
from sml.manifold.floyd import floyd_opt

# Add the sml directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

# from sml.neighbors.knn import KNNClassifer


class UnitTests(unittest.TestCase):
    def test_knn(self):
        sim = spsim.Simulator.simple(3, libspu.ProtocolKind.ABY3, libspu.FieldType.FM64)

        def dijkstra_all_pairs(
            Knn,
            mpc_dist_inf,
            num_samples,
        ):

            def compute_distances_for_sample(i, Knn, num_samples, mpc_dist_inf):
                return mpc_dijkstra(Knn, num_samples, i, mpc_dist_inf)

            compute_distances = jax.vmap(
                compute_distances_for_sample, in_axes=(0, None, None, None)
            )

            indices = jnp.arange(num_samples)  # 样本索引
            mpc_shortest_paths = compute_distances(
                indices, Knn, num_samples, mpc_dist_inf
            )
            return mpc_shortest_paths

        num_samples = 6
        dist_inf = jnp.full(num_samples, np.inf)

        X = np.random.rand(num_samples, num_samples)
        X = (X + X.T) / 2
        X[X == 0] = np.inf
        np.fill_diagonal(X, 0)

        dijkstra_ans = spsim.sim_jax(sim, dijkstra_all_pairs, static_argnums=(2,))(
            X, dist_inf, num_samples
        )

        print('dijkstra_ans: \n', dijkstra_ans)

        floyd_ans = spsim.sim_jax(sim, floyd_opt)(X)
        print('floyd_ans: \n', floyd_ans)

        # sklearn test
        sklearn_ans = shortest_path(X, method="D", directed=False)
        print('sklearn_ans: \n', sklearn_ans)


if __name__ == "__main__":
    unittest.main()
