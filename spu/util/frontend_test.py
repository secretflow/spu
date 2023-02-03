# Copyright 2022 Ant Group Co., Ltd.
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
import tensorflow as tf

import spu.util.frontend as spu_fe
from spu import spu_pb2


class UnitTests(unittest.TestCase):
    def test_jax_compile(self):
        executable, output = spu_fe.compile(
            spu_fe.Kind.JAX,
            jnp.add,
            (np.array([1, 2]), np.array([2, 4])),
            {},
            ["in1", "in2"],
            [spu_pb2.VIS_PUBLIC, spu_pb2.VIS_PUBLIC],
            lambda out_flat: [f'test-out{idx}' for idx in range(len(out_flat))],
        )
        self.assertEqual(executable.name, "add")
        self.assertEqual(executable.input_names, ["in1", "in2"])
        self.assertEqual(executable.output_names, ["test-out0"])
        self.assertTrue(
            "  func.func @main(%arg0: tensor<2x!pphlo.pub<i32>>,"
            " %arg1: tensor<2x!pphlo.pub<i32>>) -> tensor<2x!pphlo.pub<i32>> {\n"
            "    %0 = \"pphlo.add\"(%arg0, %arg1) : (tensor<2x!pphlo.pub<i32>>,"
            " tensor<2x!pphlo.pub<i32>>) -> tensor<2x!pphlo.pub<i32>>\n"
            "    return %0 : tensor<2x!pphlo.pub<i32>>\n  }" in executable.code.decode()
        )
        self.assertEqual(output.shape, (2,))
        self.assertEqual(output.dtype, np.dtype("int32"))

    def test_tf_compile(self):
        executable, output = spu_fe.compile(
            spu_fe.Kind.Tensorflow,
            tf.add,
            (np.array([1, 2]), np.array([2, 4])),
            {},
            ["in1", "in2"],
            [spu_pb2.VIS_PUBLIC, spu_pb2.VIS_PUBLIC],
            lambda out_flat: [f'test-out{idx}' for idx in range(len(out_flat))],
        )
        self.assertEqual(executable.name, "add")
        self.assertEqual(executable.input_names, ["in1", "in2"])
        self.assertEqual(executable.output_names, ["test-out0"])
        self.assertTrue(
            "  func.func @main(%arg0: tensor<2x!pphlo.pub<i64>>,"
            " %arg1: tensor<2x!pphlo.pub<i64>>) -> tensor<2x!pphlo.pub<i64>> {\n"
            "    %0 = \"pphlo.add\"(%arg0, %arg1) : (tensor<2x!pphlo.pub<i64>>,"
            " tensor<2x!pphlo.pub<i64>>) -> tensor<2x!pphlo.pub<i64>>\n"
            "    return %0 : tensor<2x!pphlo.pub<i64>>\n  }" in executable.code.decode()
        )
        self.assertEqual(output.shape, (2,))
        self.assertEqual(output.dtype, np.dtype("int64"))


if __name__ == '__main__':
    unittest.main()
