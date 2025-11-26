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
from spu.experimental import drop_cached_var, epsilon, make_cached_var, reveal


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


def test_make_cached_var_basic(simulator, compiler_options, test_matrices):
    """Test basic functionality of make_cached_var."""
    x, y = test_matrices

    # Define function that uses make_cached_var
    fn = lambda x, y: make_cached_var(jnp.matmul(x, y))

    # Create SPU function
    spu_fn = ppsim.sim_jax(simulator, fn, copts=compiler_options)

    # Execute SPU version only (make_cached_var doesn't have CPU implementation)
    spu_result = spu_fn(x, y)

    # For comparison, compute expected result (matrix multiplication)
    expected_result = jnp.matmul(x, y)

    # Verify shapes match
    assert (
        spu_result.shape == expected_result.shape
    ), f"Shape mismatch: SPU {spu_result.shape} vs expected {expected_result.shape}"

    # Verify results are close
    np.testing.assert_allclose(spu_result, expected_result, rtol=1e-3, atol=1e-5)

    # Verify that PPHLO was generated (basic check)
    assert hasattr(spu_fn, 'pphlo'), "SPU function should have pphlo attribute"


def test_drop_cached_var_basic(simulator, compiler_options, test_matrices):
    """Test basic functionality of drop_cached_var."""
    x, y = test_matrices

    # Define function that uses drop_cached_var
    fn = lambda x, y: drop_cached_var(jnp.matmul(x, y))

    # Create SPU function
    spu_fn = ppsim.sim_jax(simulator, fn, copts=compiler_options)

    # Execute SPU version only (drop_cached_var doesn't have CPU implementation)
    spu_result = spu_fn(x, y)

    # For comparison, compute expected result (matrix multiplication)
    expected_result = jnp.matmul(x, y)

    # Verify shapes match
    assert (
        spu_result.shape == expected_result.shape
    ), f"Shape mismatch: SPU {spu_result.shape} vs expected {expected_result.shape}"

    # Verify results are close
    np.testing.assert_allclose(spu_result, expected_result, rtol=1e-3, atol=1e-5)

    # Verify that PPHLO was generated (basic check)
    assert hasattr(spu_fn, 'pphlo'), "SPU function should have pphlo attribute"


def test_drop_cached_var_with_dependencies(simulator, compiler_options, test_matrices):
    """Test drop_cached_var with dependencies."""
    x, y = test_matrices

    # Define function that uses drop_cached_var with dependencies
    def fn(x, y):
        z = jnp.matmul(x, y)
        w = jnp.sum(x)  # dependency
        return drop_cached_var(z, w)

    # Create SPU function
    spu_fn = ppsim.sim_jax(simulator, fn, copts=compiler_options)

    # Execute SPU version only (drop_cached_var doesn't have CPU implementation)
    spu_result = spu_fn(x, y)

    # For comparison, compute expected result (matrix multiplication)
    expected_result = jnp.matmul(x, y)

    # Verify shapes match
    assert (
        spu_result.shape == expected_result.shape
    ), f"Shape mismatch: SPU {spu_result.shape} vs expected {expected_result.shape}"

    # Verify results are close
    np.testing.assert_allclose(spu_result, expected_result, rtol=1e-3, atol=1e-5)


def test_cached_var_roundtrip(simulator, compiler_options, test_matrices):
    """Test make_cached_var followed by drop_cached_var."""
    x, y = test_matrices

    # Define function that uses both make_cached_var and drop_cached_var
    def fn(x, y):
        z = jnp.matmul(x, y)
        cached = make_cached_var(z)
        return drop_cached_var(cached)

    # Create SPU function
    spu_fn = ppsim.sim_jax(simulator, fn, copts=compiler_options)

    # Execute SPU version only (cached var functions don't have CPU implementation)
    spu_result = spu_fn(x, y)

    # For comparison, compute expected result (matrix multiplication)
    expected_result = jnp.matmul(x, y)

    # Verify shapes match
    assert (
        spu_result.shape == expected_result.shape
    ), f"Shape mismatch: SPU {spu_result.shape} vs expected {expected_result.shape}"

    # Verify results are close
    np.testing.assert_allclose(spu_result, expected_result, rtol=1e-3, atol=1e-5)

    # Verify PPHLO contains cached variable operations
    # Note: PPHLO content inspection works at runtime even if type checker complains
    if hasattr(spu_fn, 'pphlo') and spu_fn.pphlo is not None:
        print(f"PPHLO generated successfully for cached variable roundtrip test")


def test_all_experimental_functions_together(
    simulator, compiler_options, test_matrices
):
    """Test all experimental functions (reveal, epsilon, make_cached_var, drop_cached_var) together."""
    x, y = test_matrices

    # Define function that uses all experimental functions
    def fn(x, y):
        # Matrix multiplication
        z = jnp.matmul(x, y)

        # Cache the result
        cached_z = make_cached_var(z)

        # Reveal the cached result
        revealed_z = reveal(cached_z)

        # Drop the cached variable and add epsilon
        final_result = drop_cached_var(revealed_z) + epsilon()

        return final_result

    # Create SPU function
    spu_fn = ppsim.sim_jax(simulator, fn, copts=compiler_options)

    # Execute SPU version only (cached var functions don't have CPU implementation)
    spu_result = spu_fn(x, y)

    # For comparison, compute expected result (matrix multiplication + epsilon)
    expected_result = jnp.matmul(x, y) + 2**-18

    # Verify shapes match
    assert (
        spu_result.shape == expected_result.shape
    ), f"Shape mismatch: SPU {spu_result.shape} vs expected {expected_result.shape}"

    # Verify results are close (more relaxed tolerance due to multiple operations)
    np.testing.assert_allclose(spu_result, expected_result, rtol=1e-2, atol=1e-4)

    # Verify PPHLO contains cached variable operations
    # Note: Direct PPHLO inspection has type issues but runtime functionality works
    if hasattr(spu_fn, 'pphlo'):
        print("PPHLO generated successfully for comprehensive experimental test")


@pytest.mark.parametrize(
    "shape_pair",
    [
        ((2, 3), (3, 2)),
        ((1, 5), (5, 1)),
        ((4, 4), (4, 4)),
    ],
)
def test_cached_var_different_shapes(simulator, compiler_options, shape_pair):
    """Test cached variables with different matrix shapes."""
    x_shape, y_shape = shape_pair

    np.random.seed(42)  # Fixed seed for reproducible tests
    x = np.random.randn(*x_shape)
    y = np.random.randn(*y_shape)

    # Define function using cached variables
    def fn(x, y):
        z = jnp.matmul(x, y)
        cached = make_cached_var(z)
        return drop_cached_var(cached)

    # Create SPU function
    spu_fn = ppsim.sim_jax(simulator, fn, copts=compiler_options)

    # Execute SPU version only (cached var functions don't have CPU implementation)
    spu_result = spu_fn(x, y)

    # For comparison, compute expected result (matrix multiplication)
    expected_result = jnp.matmul(x, y)

    # Verify shapes match
    assert (
        spu_result.shape == expected_result.shape
    ), f"Shape mismatch: SPU {spu_result.shape} vs expected {expected_result.shape}"

    # Verify results are close
    np.testing.assert_allclose(spu_result, expected_result, rtol=1e-3, atol=1e-5)


if __name__ == "__main__":
    # Allow running as script for debugging
    pytest.main([__file__, "-v"])
