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

import unittest

import jax.numpy as jnp
import numpy as np

import spu.libspu as libspu
import spu.utils.simulation as spsim
from sml.utils.extmath import serial_jacobi_evd


def _generate_symmetric_matric(n):

    mat = jnp.array(np.random.rand(n, n))
    mat = (mat + mat.T) / 2

    return mat


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


def _generate_sample_pack(small=32, medium=50, large=100):
    # small_pack = _generate_symmetric_matric(small)
    small_pack = _generate_symmetric_matrix_with_bounded_eigenvalues(small)
    # medium_pack = _generate_symmetric_matrix_with_bounded_eigenvalues(medium)

    # large size (n>100) may fail due to accumulated errors
    # large_pack = _generate_symmetric_matrix_with_bounded_eigenvalues(large)

    data_pack = {
        "small": small_pack,
        # "medium": medium_pack,
        # "large": large_pack,
    }

    return data_pack


class ExtMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(" ========= start test of extmath package ========= \n")
        # 1. set seed
        np.random.seed(0)

        # 2. init simulator
        config64 = libspu.RuntimeConfig(
            protocol=libspu.ProtocolKind.ABY3,
            field=libspu.FieldType.FM64,
            fxp_fraction_bits=18,
            # enable_pphlo_profile=True,
        )
        config128 = libspu.RuntimeConfig(
            protocol=libspu.ProtocolKind.ABY3,
            field=libspu.FieldType.FM128,
            fxp_fraction_bits=30,
            # enable_pphlo_profile=True,
        )
        config64.enable_pphlo_profile = True
        config128.enable_pphlo_profile = True

        sim64 = spsim.Simulator(3, config64)
        sim128 = spsim.Simulator(3, config128)
        cls.sim_dict = {"FM64": sim64, "FM128": sim128}

        # 3. generate sample data
        cls.data_pack = _generate_sample_pack()

        print("all pre-work done!")

    @classmethod
    def tearDownClass(cls):
        print(" ========= test of extmath package end ========= \n")

    def _jacobi_evd_test_pack(
        self,
        max_jacobi_iter=5,
        is_plain=False,
        field="FM128",
    ):
        data_pack = self.data_pack

        for mat in data_pack.values():
            self._jacobi_evd_test_main(
                mat,
                max_jacobi_iter=max_jacobi_iter,
                is_plain=is_plain,
                field=field,
            )

    def _jacobi_evd_test_main(
        self,
        A,
        max_jacobi_iter=5,
        is_plain=False,
        field="FM128",
    ):

        print("test matrix shape: ", A.shape)
        print("field:", field)
        run_func = (
            serial_jacobi_evd
            if is_plain
            else spsim.sim_jax(
                self.sim_dict[field], serial_jacobi_evd, static_argnums=(1,)
            )
        )
        val, vec = run_func(A, max_jacobi_iter)
        sorted_indices = jnp.argsort(val)[::-1]
        eig_vec = vec.T[sorted_indices]
        eig_val = val[sorted_indices]

        val_np, vec_np = np.linalg.eig(A)
        sorted_indices = jnp.argsort(val_np)[::-1]
        eig_vec_np = vec_np.T[sorted_indices]
        eig_val_np = val_np[sorted_indices]

        abs_diff = np.abs(np.abs(eig_vec_np) - np.abs(eig_vec))
        rel_error = abs_diff / (np.abs(eig_vec_np) + 1e-8)

        print("avg absolute error:\n", np.mean(abs_diff))
        print("avg relative error:\n", np.mean(rel_error))

        top_k = 5
        rmse = np.sqrt(
            np.mean((np.abs(eig_vec_np[:top_k]) - np.abs(eig_vec[:top_k])) ** 2)
        )
        print("rmse of top-k eigenvectors:\n", rmse)

        # check eigenvalues equal
        np.testing.assert_allclose(eig_val_np, eig_val, rtol=1e-2, atol=1e-2)

        # check top k eigenvectors  (maybe with sign flip)
        np.testing.assert_allclose(
            np.abs(eig_vec_np[:top_k]), np.abs(eig_vec[:top_k]), rtol=1e-1, atol=1e-1
        )

        # check whether Ax - \lambda x = 0
        Ax = A @ vec
        lambdax = vec * val
        np.testing.assert_allclose(np.abs(Ax), np.abs(lambdax), rtol=1e-2, atol=1e-2)

        # check orthogonality (eigenvector columns are pairwise orthogonal)
        ortho_check = eig_vec.T @ eig_vec
        np.testing.assert_allclose(
            ortho_check, np.eye(eig_vec.shape[1]), rtol=2e-2, atol=2e-2
        )

    def test_jacobi_evd_plain(self):
        print(" ========= start test jacobi_evd plain =========\n")

        self._jacobi_evd_test_pack(is_plain=True)

        print(" ========= test evd plain end ========= \n")

    def test_jacobi_evd(self):
        print(" ========= start test jacobi_evd =========\n")

        self._jacobi_evd_test_pack(is_plain=False)

        print(" ========= test evd end ========= \n")


if __name__ == "__main__":
    unittest.main()
