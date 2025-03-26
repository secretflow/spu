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
from scipy.linalg import eigh

import spu.libspu as libspu
import spu.utils.simulation as spsim
from sml.manifold.jacobi import Jacobi

# Add the sml directory to the path
# sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
class UnitTests(unittest.TestCase):
    def test_jacobi(self):
        sim = spsim.Simulator.simple(3, libspu.ProtocolKind.ABY3, libspu.FieldType.FM128)

        num_samples = 10

        X = np.random.rand(num_samples, num_samples)
        X = (X + X.T) / 2

        values, vectors = spsim.sim_jax(sim, Jacobi, static_argnums=(1,))(
            X, num_samples
        )
        
        # Matrix form verification: XV = VΛ
        vectors=vectors.T
        Λ = values
        XV = np.dot(X, vectors)
        VΛ = np.dot(vectors, Λ)
        np.testing.assert_allclose(
            VΛ, XV, rtol=0, atol=1e-3
        )

        # Verify the orthogonality of eigenvectors
        vT_v = np.dot(vectors.T, vectors)
        I=np.eye(num_samples)
        # Verify that vT_v is an identity matrix
        np.testing.assert_allclose(
            vT_v, I, rtol=0, atol=1e-3
        )

        # scipy.linalg.eigh
        eigenvalues, eigenvectors = eigh(X)
        
        # Verify eigenvalues
        eigenvalues=jax.lax.sort(eigenvalues)
        values=jnp.diag(values)
        values=jax.lax.sort(values)
        # As num_samples become larger, the accuracy decreases, and it cannot reach 1e-3
        np.testing.assert_allclose(
            jnp.abs(values), jnp.abs(eigenvalues), rtol=0, atol=1e-3
        )


if __name__ == "__main__":
    unittest.main()
