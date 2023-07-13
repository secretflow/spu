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
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from sml.pca.simple_pca  import SimplePCA


class UnitTests(unittest.TestCase):
    def test_simple(self):
        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64
        )

        def proc(X):
            model = SimplePCA(
                method='full',
                n_components=2,
            )

            model.fit(X)
            X_transformed = model.transform(X)
            X_variances = model._variances

            return X_transformed, X_variances

        # Create a simple dataset
        X = random.normal(random.PRNGKey(0), (15, 5))

        # Run the simulation
        result = spsim.sim_jax(sim, proc)(X)

        # The transformed data should have 2 dimensions
        self.assertEqual(result[0].shape[1], 2)

        # The mean of the transformed data should be approximately 0
        self.assertTrue(jnp.allclose(jnp.mean(result[0], axis=0), 0, atol=1e-4))

        X_np = np.array(X)

        # Run PCA using sklearn
        sklearn_pca = SklearnPCA(n_components=2)
        X_transformed_sklearn = sklearn_pca.fit_transform(X_np)

        # Compare the results
        # Note: the signs of the components can be different between different PCA implementations,
        # so we need to compare the absolute values
        print("X_transformed_sklearn: ", X_transformed_sklearn)
        print("X_transformed_jax", result[0])

        # compare variance
        print("X_transformed_sklearn.explained_variance_: ", sklearn_pca.explained_variance_)
        print("X_transformed_jax.explained_variance_: ", result[1])



if __name__ == "__main__":
    unittest.main()
