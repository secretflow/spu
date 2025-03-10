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

import jax.numpy as jnp
import numpy as np
from scipy.linalg import eigh

import spu.libspu as libspu
import spu.utils.simulation as spsim
from sml.manifold.jacobi import Jacobi

# Add the sml directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))


class UnitTests(unittest.TestCase):
    def test_knn(self):
        sim = spsim.Simulator.simple(3, libspu.ProtocolKind.ABY3, libspu.FieldType.FM64)

        num_samples = 6

        X = np.random.rand(num_samples, num_samples)
        X = (X + X.T) / 2

        values, vectors = spsim.sim_jax(sim, Jacobi, static_argnums=(1,))(
            X, num_samples
        )

        print('values: \n', jnp.diag(values))

        print('vectors: \n', vectors)

        # scipy.linalg.eigh
        eigenvalues, eigenvectors = eigh(X)

        print("Eigenvalues: \n", eigenvalues)
        print("Eigenvectors: \n", eigenvectors)


if __name__ == "__main__":
    unittest.main()
