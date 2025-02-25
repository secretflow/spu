import unittest

import jax.numpy as jnp
import numpy as np

import spu.libspu as libspu
import spu.utils.simulation as spsim
from sml.fyy_pca.jacobi_evd import (
    generate_ring_sequence,
    serial_jacobi_evd,
)


def _generate_symmetric_matric(n):

    mat = jnp.array(np.random.rand(n, n))
    mat = (mat + mat.T) / 2
    # mat = jnp.eye(n)

    return mat


def _generate_sample_pack(small=25, medium=50, large=100):
    small_pack = _generate_symmetric_matric(small)
    medium_pack = _generate_symmetric_matric(medium)
    large_pack = _generate_symmetric_matric(large)

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
        field="FM64",
        # eig_atol=1e-2,
        # eig_rtol=1e-2,
        # vec_atol=1e-2,
        # vec_rtol=1e-2,
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
        x,
        max_jacobi_iter=5,
        is_plain=False,
        field="FM64",
    ):

        print("test matrix shape: ", x.shape)
        print("field:", field)
        run_func = (
            serial_jacobi_evd
            if is_plain
            else spsim.sim_jax(
                self.sim_dict[field], serial_jacobi_evd, static_argnums=(2,)
            )
        )
        val, vec = run_func(x, jnp.eye(x.shape[0]), max_jacobi_iter)
        sorted_indices = jnp.argsort(val)[::-1]
        eig_vec = vec.T[sorted_indices]
        eig_val = val[sorted_indices]

        val_np, vec_np = np.linalg.eig(x)
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

    def test_jacobi_evd_plain(self):
        print(" ========= start test jacobi_evd plain =========\n")

        self._jacobi_evd_test_pack(is_plain=True)

        print(" ========= test svd end plain ========= \n")

    def test_jacobi_evd(self):
        print(" ========= start test jacobi_evd =========\n")

        self._jacobi_evd_test_pack(is_plain=False)

        print(" ========= test svd end ========= \n")


if __name__ == "__main__":
    unittest.main()
