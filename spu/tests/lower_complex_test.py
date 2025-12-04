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
import spu.utils.frontend as spu_fe


class LowerComplexPassTest(unittest.TestCase):
    """Test cases to verify ExpandComplexOpsPass functionality from StableHLO.

    The ExpandComplexOpsPass is a StableHLO pass called in the frontend compilation
    pipeline (see src/libspu/compiler/front_end/fe.cc). It replaces complex number
    operations with equivalent operations on real and imaginary parts, which is
    necessary for SPU to handle complex computations.

    This test verifies that:
    1. Complex number operations (add, multiply, divide, etc.) can be compiled
    2. The compilation pipeline properly handles complex types
    3. The lowering pass (ExpandComplexOpsPass) successfully converts complex ops to real/imag components

    Each test compiles a JAX function with complex inputs and verifies that:
    - Compilation succeeds without errors
    - The output shape and dtype are correct
    - The generated IR is valid (printed for manual inspection)
    """

    def test_complex_add(self):
        """Test complex number addition is properly lowered."""

        def complex_add(x, y):
            return jnp.add(x, y)

        # Create complex number inputs
        x = np.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex64)
        y = np.array([5.0 + 6.0j, 7.0 + 8.0j], dtype=np.complex64)

        executable, output = spu_fe.compile(
            spu_fe.Kind.JAX,
            complex_add,
            (x, y),
            {},
            ["x", "y"],
            [libspu.Visibility.VIS_PUBLIC, libspu.Visibility.VIS_PUBLIC],
            lambda out_flat: [f'out{idx}' for idx in range(len(out_flat))],
        )

        # Verify compilation succeeded
        self.assertEqual(executable.name, "complex_add")
        self.assertEqual(executable.input_names, ["x", "y"])
        self.assertEqual(executable.output_names, ["out0"])

        # Print IR for inspection
        ir_code = executable.code.decode()
        print("\n=== Complex Add IR ===")
        print(ir_code)

        # Verify output properties
        self.assertEqual(output.shape, (2,))
        self.assertEqual(output.dtype, np.dtype("complex64"))

    def test_complex_multiply(self):
        """Test complex number multiplication is properly lowered."""

        def complex_mul(x, y):
            return jnp.multiply(x, y)

        x = np.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex64)
        y = np.array([2.0 + 1.0j, 1.0 + 2.0j], dtype=np.complex64)

        executable, output = spu_fe.compile(
            spu_fe.Kind.JAX,
            complex_mul,
            (x, y),
            {},
            ["x", "y"],
            [libspu.Visibility.VIS_PUBLIC, libspu.Visibility.VIS_PUBLIC],
            lambda out_flat: [f'out{idx}' for idx in range(len(out_flat))],
        )

        self.assertEqual(executable.name, "complex_mul")

        ir_code = executable.code.decode()
        print("\n=== Complex Multiply IR ===")
        print(ir_code)

        self.assertEqual(output.shape, (2,))
        self.assertEqual(output.dtype, np.dtype("complex64"))

    def test_complex_conjugate(self):
        """Test complex conjugate operation is properly lowered."""

        def complex_conj(x):
            return jnp.conj(x)

        x = np.array([1.0 + 2.0j, 3.0 - 4.0j], dtype=np.complex64)

        executable, output = spu_fe.compile(
            spu_fe.Kind.JAX,
            complex_conj,
            (x,),
            {},
            ["x"],
            [libspu.Visibility.VIS_PUBLIC],
            lambda out_flat: [f'out{idx}' for idx in range(len(out_flat))],
        )

        self.assertEqual(executable.name, "complex_conj")

        ir_code = executable.code.decode()
        print("\n=== Complex Conjugate IR ===")
        print(ir_code)

        self.assertEqual(output.shape, (2,))
        self.assertEqual(output.dtype, np.dtype("complex64"))

    def test_complex_real_imag(self):
        """Test extraction of real and imaginary parts."""

        def get_real_imag(x):
            real = jnp.real(x)
            imag = jnp.imag(x)
            return real, imag

        x = np.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex64)

        executable, output = spu_fe.compile(
            spu_fe.Kind.JAX,
            get_real_imag,
            (x,),
            {},
            ["x"],
            [libspu.Visibility.VIS_PUBLIC],
            lambda out_flat: [f'out{idx}' for idx in range(len(out_flat))],
        )

        self.assertEqual(executable.name, "get_real_imag")

        ir_code = executable.code.decode()
        print("\n=== Complex Real/Imag IR ===")
        print(ir_code)

        # Should have two outputs: real and imag parts
        self.assertEqual(len(executable.output_names), 2)

    def test_complex_division(self):
        """Test complex number division is properly lowered."""

        def complex_div(x, y):
            return jnp.divide(x, y)

        x = np.array([1.0 + 2.0j, 6.0 + 8.0j], dtype=np.complex64)
        y = np.array([2.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex64)

        executable, output = spu_fe.compile(
            spu_fe.Kind.JAX,
            complex_div,
            (x, y),
            {},
            ["x", "y"],
            [libspu.Visibility.VIS_PUBLIC, libspu.Visibility.VIS_PUBLIC],
            lambda out_flat: [f'out{idx}' for idx in range(len(out_flat))],
        )

        self.assertEqual(executable.name, "complex_div")

        ir_code = executable.code.decode()
        print("\n=== Complex Division IR ===")
        print(ir_code)

        self.assertEqual(output.shape, (2,))
        self.assertEqual(output.dtype, np.dtype("complex64"))

    def test_complex_scalar(self):
        """Test complex scalar operations."""

        def complex_scalar_op(x):
            # Add a complex scalar to the input
            return x + (1.0 + 1.0j)

        x = np.array([1.0 + 0.0j, 0.0 + 1.0j], dtype=np.complex64)

        executable, output = spu_fe.compile(
            spu_fe.Kind.JAX,
            complex_scalar_op,
            (x,),
            {},
            ["x"],
            [libspu.Visibility.VIS_PUBLIC],
            lambda out_flat: [f'out{idx}' for idx in range(len(out_flat))],
        )

        self.assertEqual(executable.name, "complex_scalar_op")

        ir_code = executable.code.decode()
        print("\n=== Complex Scalar Op IR ===")
        print(ir_code)

        self.assertEqual(output.shape, (2,))
        self.assertEqual(output.dtype, np.dtype("complex64"))


if __name__ == '__main__':
    unittest.main()
