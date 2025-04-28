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

import jax.numpy as jnp
import numpy as np

# Add the sml directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

import sml.utils.emulation as emulation
from sml.utils.extmath import serial_jacobi_evd


def _generate_symmetric_matrix_with_bounded_eigenvalues(n, upper_bound=2):
    # Generate random feature values ​​not exceeding upper_bound
    eigenvalues = np.random.uniform(-upper_bound, upper_bound, n)

    # Create a diagonal matrix
    D = np.diag(eigenvalues)

    # Generate a random orthogonal matrix
    X = np.random.randn(n, n)
    Q, _ = np.linalg.qr(X)  # QR decomposition to obtain the orthogonal matrix

    # Constructing a symmetric matrix
    A = Q @ D @ Q.T

    return A


def emul_jacobievd(mode: emulation.Mode.MULTIPROCESS):
    print("start jacobi evd emulation.")
    np.random.seed(0)

    # ONLY test small matrix for usage purpose
    n = 32
    # mat = jnp.array(np.random.rand(n, n))
    # mat = (mat + mat.T) / 2
    mat = _generate_symmetric_matrix_with_bounded_eigenvalues(n)

    def _check_jacobievd_single(mat, max_jacobi_iter=5):
        print("start jacobi evd emulation test, with shape=", mat.shape)

        mat_spu = emulator.seal(mat)
        val, vec = emulator.run(serial_jacobi_evd, static_argnums=(1,))(
            mat_spu, max_jacobi_iter
        )
        sorted_indices = jnp.argsort(val)[::-1]
        eig_vec = vec.T[sorted_indices]
        eig_val = val[sorted_indices]

        val_np, vec_np = np.linalg.eig(mat)
        sorted_indices = jnp.argsort(val_np)[::-1]
        eig_vec_np = vec_np.T[sorted_indices]
        eig_val_np = val_np[sorted_indices]

        abs_diff = np.abs(np.abs(eig_vec_np) - np.abs(eig_vec))
        rel_error = abs_diff / (np.abs(eig_vec_np) + 1e-8)

        print("avg absolute error:\n", np.mean(abs_diff))
        print("avg relative error:\n", np.mean(rel_error))

        # check eigen values equal
        np.testing.assert_allclose(eig_val_np, eig_val, rtol=0.01, atol=0.01)

        # check eigen vectors (maybe with sign flip)
        np.testing.assert_allclose(
            np.abs(eig_vec_np), np.abs(eig_vec), rtol=0.01, atol=0.01
        )

        # check whether Ax - \lambda x = 0
        Ax = mat @ vec
        lambdax = vec * val
        np.testing.assert_allclose(np.abs(Ax), np.abs(lambdax), rtol=1e-2, atol=1e-2)

        # check orthogonality (eigenvector columns are pairwise orthogonal)
        ortho_check = eig_vec.T @ eig_vec
        np.testing.assert_allclose(
            ortho_check, np.eye(eig_vec.shape[1]), rtol=2e-2, atol=2e-2
        )

    try:
        conf_path = "sml/utils/emulations/3pc_128.json"
        emulator = emulation.Emulator(conf_path, mode, bandwidth=300, latency=20)
        emulator.up()

        _check_jacobievd_single(mat)

        print("evd emulation pass.")

    finally:
        emulator.down()


if __name__ == "__main__":
    emul_jacobievd(emulation.Mode.MULTIPROCESS)
