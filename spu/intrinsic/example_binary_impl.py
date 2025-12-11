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

"""
Example binary intrinsic implementation using JAX FFI API.

This module demonstrates how to implement a custom SPU intrinsic with
TWO inputs and a DIFFERENT output shape. This is more complex than the
simple identity example in example_impl.py.

The example_binary operation:
- Takes two 2D inputs: in1 with shape [a, b] and in2 with shape [c, d]
- Returns a zero matrix with shape [a+c, b+d]
- This demonstrates how to handle operations where output shape differs from input

Key differences from simple example:
- Output shape must be computed from input shapes
- Need to handle shape computation in both Python and C++ sides
- vmap behavior is more complex (not simply shape-agnostic)
"""

__all__ = ["example_binary"]

import jax
import jax.numpy as jnp


# *********************************
# *  HELPER FUNCTIONS             *
# *********************************


def _compute_result_shape(shape1, shape2):
    """
    Compute the output shape for example_binary.

    The output shape is the element-wise sum of input shapes.
    For inputs [a, b] and [c, d], output is [a+c, b+d].

    Args:
        shape1: Shape tuple of first input.
        shape2: Shape tuple of second input.

    Returns:
        Tuple representing the output shape.
    """
    assert len(shape1) == 2 and len(shape2) == 2, "Inputs must be 2D"
    return (shape1[0] + shape2[0], shape1[1] + shape2[1])


# *********************************
# *  PUBLIC FACING INTERFACE      *
# *********************************


def example_binary(in1, in2):
    """
    Example binary intrinsic that demonstrates multi-input operations.

    This is a demonstration of how to create a custom SPU intrinsic with
    multiple inputs and a computed output shape. The actual implementation
    in C++ (pphlo_intrinsic_executor.cc) returns a zero matrix.

    Args:
        in1: First input array with shape [a, b].
        in2: Second input array with shape [c, d].

    Returns:
        A zero array with shape [a+c, b+d].

    Note:
        Both inputs must be 2D arrays with the same dtype.
    """
    return _example_binary_call(in1, in2)


# *********************************
# *  CORE IMPLEMENTATION          *
# *********************************


@jax.custom_jvp
@jax.custom_vjp
def _example_binary_call(in1, in2):
    """
    Internal implementation that calls the FFI.

    The decorators @jax.custom_jvp and @jax.custom_vjp allow us to define
    custom differentiation rules for this operation.
    """
    return _example_binary_impl(in1, in2)


def _example_binary_impl(in1, in2):
    """
    Low-level FFI call implementation.

    This function uses jax.ffi.ffi_call to generate an XLA custom_call op.
    The key difference from simple example is that we must compute the
    output shape from the input shapes.

    Args:
        in1: First input array.
        in2: Second input array.

    Returns:
        Result of the FFI call.

    Notes:
        - Output shape is computed as [in1.shape[0]+in2.shape[0], in1.shape[1]+in2.shape[1]]
        - `vmap_method="sequential"`: Since output shape depends on input shapes in a
          non-trivial way, we use sequential vmap which loops over the batch dimension.
          This is safer than broadcast_all for operations with shape dependencies.
    """
    # Compute output shape from inputs
    result_shape = _compute_result_shape(in1.shape, in2.shape)

    return jax.ffi.ffi_call(
        "example_binary",  # Target name matching C++ handler
        jax.ShapeDtypeStruct(result_shape, in1.dtype),  # Output spec
        has_side_effect=True,
        # Use sequential for operations where output shape depends on input shapes
        # in ways that don't simply add a batch dimension
        vmap_method="sequential",
    )(in1, in2)


# **********************************
# *  SUPPORT FOR FORWARD AUTODIFF  *
# **********************************

# JVP for this operation is not straightforward because:
# 1. The output doesn't directly depend on input values (always zeros)
# 2. The gradient would be zero regardless of input tangents
#
# For demonstration, we implement a simple rule that returns zeros.


@_example_binary_call.defjvp
def _example_binary_jvp(primals, tangents):
    """
    Forward-mode autodiff (JVP) rule for example_binary.

    Since example_binary always returns zeros (independent of input values),
    the tangent output is also zeros.

    Args:
        primals: Tuple of primal input values (in1, in2).
        tangents: Tuple of tangent values (in1_dot, in2_dot).

    Returns:
        Tuple of (primal_output, tangent_output).
    """
    in1, in2 = primals
    # Compute primal output
    output = _example_binary_call(in1, in2)
    # Tangent is zeros with same shape as output (since output doesn't depend on input values)
    output_dot = jnp.zeros_like(output)
    return output, output_dot


# **********************************
# *  SUPPORT FOR REVERSE AUTODIFF  *
# **********************************

# VJP for this operation:
# Since output = zeros (constant, doesn't depend on inputs),
# the gradients with respect to inputs are also zeros.


def _example_binary_fwd(in1, in2):
    """
    Forward pass for VJP.

    Returns the primal output and residuals for backward pass.
    We save the input shapes to generate correctly-shaped zero gradients.

    Args:
        in1: First input array.
        in2: Second input array.

    Returns:
        Tuple of (output, residuals).
    """
    output = _example_binary_call(in1, in2)
    # Save input shapes and dtypes for backward pass
    residuals = (in1.shape, in2.shape, in1.dtype)
    return output, residuals


def _example_binary_bwd(residuals, grad_output):
    """
    Backward pass for VJP.

    Since example_binary output doesn't depend on input values
    (always returns zeros), gradients with respect to inputs are zeros.

    Args:
        residuals: Tuple of (in1_shape, in2_shape, dtype) from forward pass.
        grad_output: Gradient of loss with respect to output.

    Returns:
        Tuple of gradients with respect to (in1, in2).
    """
    in1_shape, in2_shape, dtype = residuals
    # Output doesn't depend on input values, so gradients are zeros
    grad_in1 = jnp.zeros(in1_shape, dtype=dtype)
    grad_in2 = jnp.zeros(in2_shape, dtype=dtype)
    return (grad_in1, grad_in2)


# Register the VJP rule
_example_binary_call.defvjp(_example_binary_fwd, _example_binary_bwd)


# ************************************
# *  NOTES ON VMAP (BATCHING)        *
# ************************************

# For this operation, we use `vmap_method="sequential"` because:
#
# 1. The output shape depends on input shapes in a non-trivial way
#    (output_shape = in1_shape + in2_shape element-wise)
#
# 2. When vmapped, the relationship becomes complex:
#    - Input shapes: [batch, a, b] and [batch, c, d]
#    - Each batch element should produce shape [a+c, b+d]
#    - Final output should be [batch, a+c, b+d]
#
# 3. "broadcast_all" wouldn't work correctly here because it assumes
#    the operation is shape-agnostic, which this is not.
#
# 4. "sequential" processes each batch element independently, which
#    guarantees correct behavior at the cost of some performance.
#
# For shape-agnostic operations (like identity), use "broadcast_all".
# For operations with complex shape dependencies, use "sequential".
