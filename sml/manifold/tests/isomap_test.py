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
from sklearn.manifold import Isomap

import spu.libspu as libspu
import spu.utils.simulation as spsim
from sml.manifold.isomap import ISOMAP

# Add the sml directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

# from sml.neighbors.knn import KNNClassifer


class UnitTests(unittest.TestCase):
    def test_knn(self):
        sim = spsim.Simulator.simple(3, libspu.ProtocolKind.ABY3, libspu.FieldType.FM64)

        def isomap(
            sX,
            num_samples,
            num_features,
            k,
            num_components,
        ):
            embedding = ISOMAP(
                n_components=num_components,
                n_neighbors=k,
                n_samples=num_samples,
                n_features=num_features,
            )
            X_transformed = embedding.fit_transform(sX)
            return X_transformed

        # Set sample size and dimensions
        num_samples = 20  # Number of samples
        num_features = 4  # Sample dimension
        k = 15  # Number of nearest neighbors
        num_components = 3  # Dimension after dimensionality reduction

        # Generate random input
        # seed = int(time.time())
        seed = 5
        key = jax.random.PRNGKey(seed)
        X = jax.random.uniform(
            key, shape=(num_samples, num_features), minval=0.0, maxval=1.0
        )

        # 运行模拟器
        ans = spsim.sim_jax(sim, isomap, static_argnums=(1, 2, 3, 4))(
            X,
            num_samples,
            num_features,
            k,
            num_components,
        )

        # sklearn test
        embedding = Isomap(n_components=num_components, n_neighbors=k)
        X_transformed = embedding.fit_transform(X)
        print('X_transformed: \n', X_transformed)

        np.testing.assert_allclose(
            jnp.abs(X_transformed), jnp.abs(ans), rtol=0, atol=4e-2
        )


if __name__ == "__main__":
    unittest.main()
