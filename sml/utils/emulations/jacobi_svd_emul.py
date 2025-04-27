# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

import jax.random as random
import numpy as np
from sklearn.decomposition import TruncatedSVD as SklearnSVD

# Add the library directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))

import sml.utils.emulation as emulation
from sml.utils.jacobi_svd import jacobi_svd


def emul_jacobi_svd(mode=emulation.Mode.MULTIPROCESS):
    print("Start Jacobi SVD emulation.")

    def proc_transform(A, max_iter=100, compute_uv=True):
        U, singular_values, V_T = jacobi_svd(
            A, max_iter=max_iter, compute_uv=compute_uv
        )
        return U, singular_values, V_T

    try:
        # bandwidth and latency only work for docker mode
        emulator = emulation.Emulator(
            emulation.CLUSTER_ABY3_3PC, mode, bandwidth=300, latency=20
        )
        emulator.up()

        # Create a random dataset
        A = random.normal(random.PRNGKey(0), (10, 10))
        A = (A + A.T) / 2
        A_np = np.array(A)
        A_spu = emulator.seal(A)
        results = emulator.run(proc_transform)(A_spu)

        # Compare with sklearn
        model = SklearnSVD(n_components=min(A_np.shape))
        model.fit(A_np)
        Sigma = model.singular_values_
        Matrix = model.components_

        # Sort Jacobi results[1] (singular values) in descending order
        sorted_indices = np.argsort(results[1])[
            ::-1
        ]  # Get indices for descending order
        sorted_singular_values = results[1][sorted_indices]
        sorted_V_T = results[2][
            sorted_indices, :
        ]  # Adjust V_T to match the sorted singular values

        # Compare the results
        np.testing.assert_allclose(Sigma, sorted_singular_values, rtol=0.01, atol=0.01)
        np.testing.assert_allclose(
            np.abs(Matrix), np.abs(sorted_V_T), rtol=0.1, atol=0.1
        )

    finally:
        emulator.down()


if __name__ == "__main__":
    emul_jacobi_svd(emulation.Mode.MULTIPROCESS)
