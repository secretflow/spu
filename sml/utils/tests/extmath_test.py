# Copyright 2023 Ant Group Co., Ltd.
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

import spu.spu_pb2 as spu_pb2  # type: ignore
import spu.utils.simulation as spsim
from sml.utils.extmath import randomized_svd, svd


def _generate_matric(m, n):
    square_size = int(np.sqrt(m * n))
    mat1 = jnp.array(np.random.rand(m, n))
    mat2 = jnp.array(np.random.rand(n, m))
    square = jnp.array(np.random.rand(square_size, square_size))  # prevent 0 singular

    return [mat1, mat2, square]


def _generate_sample_pack(small=10, medium=100, large=1000):
    # for small size, use shape (small, small/2)
    small_pack = _generate_matric(small, int(small / 2))

    # for other size, use shape (other, other/10)
    medium_pack = _generate_matric(medium, int(medium / 10))
    large_pack = _generate_matric(large, int(large / 10))

    data_pack = {
        "small": small_pack,
        "medium": medium_pack,
        "large": large_pack,
    }

    return data_pack


class ExtMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(" ========= start test of extmath package ========= \n")
        # 1. set seed
        np.random.seed(0)

        # 2. init simulator
        config64 = spu_pb2.RuntimeConfig(
            protocol=spu_pb2.ProtocolKind.ABY3,
            field=spu_pb2.FieldType.FM64,
            fxp_fraction_bits=18,
        )
        config128 = spu_pb2.RuntimeConfig(
            protocol=spu_pb2.ProtocolKind.ABY3,
            field=spu_pb2.FieldType.FM128,
            fxp_fraction_bits=30,
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

    def _svd_test_pack(
        self,
        pack_name="small",
        max_power_iter=100,
        is_plain=False,
        field="FM128",  # FM64 seems will fail
        sin_atol=1e-2,
        sin_rtol=1e-2,
        vec_atol=1e-1,
        vec_rtol=1e-1,
        skip_square=False,
        scale=1,
    ):
        data_pack = self.data_pack[pack_name]
        if skip_square:
            data_pack = data_pack[:-1]

        for mat in data_pack:
            self._svd_test_main(
                mat / scale,
                max_power_iter=max_power_iter,
                is_plain=is_plain,
                field=field,
                sin_atol=sin_atol,
                sin_rtol=sin_rtol,
                vec_atol=vec_atol,
                vec_rtol=vec_rtol,
            )

    def _svd_test_main(
        self,
        x,
        max_power_iter=100,
        is_plain=False,
        field="FM128",  # FM64 seems will fail
        sin_atol=1e-2,
        sin_rtol=1e-2,
        vec_atol=1e-1,
        vec_rtol=1e-1,
    ):
        print("test matrix shape: ", x.shape)
        run_func = (
            svd
            if is_plain
            else spsim.sim_jax(self.sim_dict[field], svd, static_argnums=(1,))
        )

        jax_u, jax_s, jax_vt = jnp.linalg.svd(x, full_matrices=False)
        u, s, vt = run_func(x, max_power_iter)

        # 1. check svd shape matching(full_matrices=False)
        self.assertTupleEqual(jax_u.shape, u.shape)
        self.assertTupleEqual(jax_s.shape, s.shape)
        self.assertTupleEqual(jax_vt.shape, vt.shape)

        # 2. check singular values equal
        np.testing.assert_allclose(jax_s, s, rtol=sin_rtol, atol=sin_atol)

        # 3. check U/Vt (maybe with sign flip)
        np.testing.assert_allclose(
            np.dot(jax_u, jax_vt), np.dot(u, vt), rtol=vec_rtol, atol=vec_atol
        )

    def test_svd_plain(self):
        print(" ========= start test svd plain =========\n")

        # small & medium can pass the test
        self._svd_test_pack("small", is_plain=True)
        self._svd_test_pack("medium", is_plain=True)

        # TODO: for large size, only 2 rectangle tests can pass, square matrix will get huge error
        self._svd_test_pack(
            "large", is_plain=True, vec_atol=0.1, sin_atol=0.1, skip_square=True
        )

        print(" ========= test svd end plain ========= \n")

    def test_svd(self):
        print(" ========= start test svd =========\n")

        # small test can pass the test
        self._svd_test_pack("small", is_plain=False)

        # WARNING: the following two tests will cost much time, so we close them by default
        # medium test for square matrix will fail.
        # self._svd_test_pack(
        #     "medium",
        #     is_plain=False,
        #     sin_rtol=0.1,
        #     sin_atol=0.1,
        #     skip_square=True,
        # )

        # for large size, all tests will fail by default.
        # but for rectangle matrix, set scale can pass the test
        # self._svd_test_pack(
        #     "large",
        #     is_plain=False,
        #     vec_atol=0.1,
        #     sin_atol=0.1,
        #     skip_square=True,
        #     scale=100000,
        # )

        print(" ========= test svd end ========= \n")


if __name__ == "__main__":
    unittest.main()
