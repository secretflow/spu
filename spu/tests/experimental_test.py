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

import jax.numpy as jnp
import numpy as np
import pytest

import spu.libspu as libspu
import spu.utils.simulation as ppsim
from spu.experimental import epsilon, reveal


@pytest.fixture
def simulator():
    """Create a simulator instance for testing."""
    return ppsim.Simulator.simple(3, libspu.ProtocolKind.ABY3, libspu.FieldType.FM64)


@pytest.fixture
def compiler_options():
    """Create compiler options for testing."""
    copts = libspu.CompilerOptions()
    # Tweak compiler options
    copts.disable_div_sqrt_rewrite = True
    return copts


@pytest.fixture
def test_matrices():
    """Create test matrices with fixed seed for reproducible tests."""
    np.random.seed(42)  # Fixed seed for reproducible tests
    x = np.random.randn(3, 4)
    y = np.random.randn(4, 5)
    return x, y


def test_matmul_with_reveal_and_epsilon(simulator, compiler_options, test_matrices):
    """Test matrix multiplication with reveal and epsilon functions."""
    x, y = test_matrices

    # Define the function to test
    fn = lambda x, y: reveal(jnp.matmul(x, y)) + epsilon()

    # Create SPU function
    spu_fn = ppsim.sim_jax(simulator, fn, copts=compiler_options)

    # Execute both SPU and CPU versions
    spu_result = spu_fn(x, y)
    cpu_result = fn(x, y)

    # Verify that PPHLO was generated
    assert spu_fn.pphlo is not None, "PPHLO should be generated"

    # Verify shapes match
    assert (
        spu_result.shape == cpu_result.shape
    ), f"Shape mismatch: SPU {spu_result.shape} vs CPU {cpu_result.shape}"

    # Verify results are close (allowing for some numerical differences)
    np.testing.assert_allclose(spu_result, cpu_result, rtol=1e-3, atol=1e-5)

    # Print for debugging (pytest will capture this)
    print(f"SPU result shape: {spu_result.shape}")
    print(f"CPU result shape: {cpu_result.shape}")
    print(f"Results are numerically close: {np.allclose(spu_result, cpu_result)}")


def test_pphlo_generation(simulator, compiler_options, test_matrices):
    """Test that PPHLO is properly generated."""
    x, y = test_matrices

    fn = lambda x, y: reveal(jnp.matmul(x, y)) + epsilon()
    spu_fn = ppsim.sim_jax(simulator, fn, copts=compiler_options)

    # Execute to generate PPHLO
    _ = spu_fn(x, y)

    # Verify PPHLO contains expected operations
    pphlo_str = str(spu_fn.pphlo)
    assert (
        "matmul" in pphlo_str.lower() or "dot" in pphlo_str.lower()
    ), "PPHLO should contain matrix multiplication"
    assert len(pphlo_str) > 0, "PPHLO should not be empty"


if __name__ == "__main__":
    # Allow running as script for debugging
    pytest.main([__file__, "-v"])
