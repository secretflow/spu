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
from sklearn.neighbors import kneighbors_graph

import spu.libspu as libspu
import spu.utils.simulation as spsim
from sml.manifold.kneighbors import mpc_kneighbors_graph


class UnitTests(unittest.TestCase):
    def test_knn(self):
        sim = spsim.Simulator.simple(3, libspu.ProtocolKind.ABY3, libspu.FieldType.FM64)

        # Set sample size and dimensions
        num_samples = 20  # Number of samples
        num_features = 4  # Sample dimension
        k = 4  # Number of nearest neighbors

        # Generate random input
        seed = int(time.time())
        key = jax.random.PRNGKey(seed)
        X = jax.random.uniform(
            key, shape=(num_samples, num_features), minval=0.0, maxval=1.0
        )

        knn = spsim.sim_jax(
            sim,
            mpc_kneighbors_graph,
            static_argnums=(1,),
        )(X, k)

        # sklearn test
        affinity_matrix = kneighbors_graph(
            X, n_neighbors=k, mode="distance", include_self=False
        )

        np.testing.assert_allclose(
            jnp.abs(knn), jnp.abs(affinity_matrix.toarray()), rtol=0, atol=1e-3
        )


if __name__ == "__main__":
    unittest.main()
