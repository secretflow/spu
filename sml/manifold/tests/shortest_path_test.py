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
# sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))


class UnitTests(unittest.TestCase):
    def test_shortest_path(self):
        sim = spsim.Simulator.simple(3, libspu.ProtocolKind.ABY3, libspu.FieldType.FM64)

        def dijkstra_all_pairs(D):
            num_samples=D.shape[0]

            def compute_distances_for_sample(i, D):
                return mpc_dijkstra(D, i)

            compute_distances = jax.vmap(
                compute_distances_for_sample, in_axes=(0, None)
            )

            indices = jnp.arange(num_samples)  # 样本索引
            mpc_shortest_paths = compute_distances(indices, D)
            return mpc_shortest_paths

        num_samples = 20

        X = np.random.rand(num_samples, num_samples)
        X = (X + X.T) / 2
        X[X == 0] = np.inf
        np.fill_diagonal(X, 0)

        dijkstra_ans = spsim.sim_jax(sim, dijkstra_all_pairs)(X)

        floyd_ans = spsim.sim_jax(sim, floyd_opt)(X)

        # sklearn test
        sklearn_ans = shortest_path(X, method="D", directed=False)

        np.testing.assert_allclose(dijkstra_ans, sklearn_ans, rtol=0, atol=1e-3)
        np.testing.assert_allclose(floyd_ans, sklearn_ans, rtol=0, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
