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
Example intrinsic implementation using JAX FFI API.

This module demonstrates how to implement a custom SPU intrinsic using the
modern JAX FFI (Foreign Function Interface) API. The FFI API replaces the
deprecated `jaxlib.hlo_helpers.custom_call` with `jax.ffi.ffi_call`.

Key concepts:
- `jax.ffi.ffi_call`: Creates an FFI call that generates XLA custom_call ops
- `jax.custom_jvp`: Defines custom forward-mode autodiff (JVP) rules
- `jax.custom_vjp`: Defines custom reverse-mode autodiff (VJP) rules
- `vmap_method`: Controls how the op behaves under `jax.vmap`

For more complex intrinsics that need different behavior on CPU vs SPU,
see `spu/experimental/epsilon_impl.py` which uses `jax.ffi.ffi_lowering`.
"""

__all__ = ["example"]

import jax


# *********************************
# *  PUBLIC FACING INTERFACE      *
# *********************************


def example(input):
    """
    Example intrinsic that acts as an identity function.

    This is a demonstration of how to create a custom SPU intrinsic.
    In practice, the C++ backend (pphlo_intrinsic_executor.cc) would
    implement the actual logic for this operation.

    Args:
        input: A JAX array of any shape and dtype.

    Returns:
        The input array unchanged (identity operation).
    """
    return _example_call(input)


# *********************************
# *  CORE IMPLEMENTATION          *
# *********************************

# We use @jax.custom_jvp and @jax.custom_vjp decorators to define custom
# autodiff rules. The order matters: custom_jvp must be the outer decorator
# for both JVP and VJP to work correctly.


@jax.custom_jvp
@jax.custom_vjp
def _example_call(input):
    """
    Internal implementation that calls the FFI.

    The decorators @jax.custom_jvp and @jax.custom_vjp allow us to define
    custom differentiation rules for this operation.
    """
    return _example_impl(input)


def _example_impl(input):
    """
    Low-level FFI call implementation.

    This function uses jax.ffi.ffi_call to generate an XLA custom_call op
    with the target name "example". The SPU runtime will intercept this
    custom_call and dispatch it to the appropriate handler in
    pphlo_intrinsic_executor.cc.

    Args:
        input: A JAX array.

    Returns:
        Result of the FFI call (identity for this example).

    Notes:
        - `result_shape_dtypes`: Specifies the output shape and dtype.
          For identity ops, this matches the input.
        - `has_side_effect=True`: Prevents the compiler from optimizing
          away this call even if the result is unused.
        - `vmap_method="broadcast_all"`: When vmapped, batch dimensions
          are passed through to the FFI call. This works because our
          C++ implementation is shape-agnostic (just returns input).
    """
    return jax.ffi.ffi_call(
        "example",  # Target name matching C++ handler
        jax.ShapeDtypeStruct(input.shape, input.dtype),  # Output spec
        has_side_effect=True,
        vmap_method="broadcast_all",
    )(input)


# **********************************
# *  SUPPORT FOR FORWARD AUTODIFF  *
# **********************************

# JVP (Jacobian-Vector Product) defines forward-mode autodiff.
# For a function f(x), JVP computes: (f(x), df/dx @ tangent)
#
# For an identity function: f(x) = x, so df/dx = I (identity matrix)
# Therefore: JVP(x, tangent) = (x, tangent)


@_example_call.defjvp
def _example_jvp(primals, tangents):
    """
    Forward-mode autodiff (JVP) rule for the example op.

    For identity-like operations, the tangent simply passes through
    unchanged, just like the primal value.

    Args:
        primals: Tuple of primal (original) input values.
        tangents: Tuple of tangent (derivative direction) values.

    Returns:
        Tuple of (primal_output, tangent_output).
    """
    (input,) = primals
    (input_dot,) = tangents
    # Primal computation
    output = _example_call(input)
    # Tangent computation: for identity, tangent passes through
    output_dot = input_dot
    return output, output_dot


# **********************************
# *  SUPPORT FOR REVERSE AUTODIFF  *
# **********************************

# VJP (Vector-Jacobian Product) defines reverse-mode autodiff.
# This is what `jax.grad` uses internally.
#
# VJP requires two functions:
# 1. fwd: Forward pass that returns (output, residuals_for_backward)
# 2. bwd: Backward pass that uses residuals to compute gradients
#
# For an identity function: f(x) = x, so df/dx = I
# The gradient just passes through: grad_x = grad_output


def _example_fwd(input):
    """
    Forward pass for VJP.

    Returns the primal output and any residuals needed for the backward pass.
    For identity ops, no residuals are needed.

    Args:
        input: The input array.

    Returns:
        Tuple of (output, residuals). Residuals is None for identity ops.
    """
    return _example_call(input), None


def _example_bwd(residuals, grad_output):
    """
    Backward pass for VJP.

    Computes gradients with respect to inputs given the gradient of the output.
    For identity ops, the gradient simply passes through unchanged.

    Args:
        residuals: Values saved from the forward pass (None for identity ops).
        grad_output: Gradient of the loss with respect to the output.

    Returns:
        Tuple of gradients with respect to each input.
    """
    del residuals  # Unused for identity ops
    # For identity: grad_input = grad_output
    return (grad_output,)


# Register the VJP rule
_example_call.defvjp(_example_fwd, _example_bwd)


# ************************************
# *  NOTES ON VMAP (BATCHING)        *
# ************************************

# Batching (vmap) support is handled automatically by the `vmap_method`
# parameter in `jax.ffi.ffi_call`. With `vmap_method="broadcast_all"`:
#
# - When the function is vmapped, inputs with batch dimensions are passed
#   directly to the FFI call with the batch dimension intact.
# - The output is expected to have the same batch dimension as the input.
#
# This is appropriate for identity-like operations where the C++ handler
# is shape-agnostic. For operations that need special batching logic,
# you may need to use `jax.custom_batching.custom_vmap`.
#
# Available vmap_method options:
# - "broadcast_all": Pass batch dims through (for shape-agnostic ops)
# - "sequential": Loop over batch dimension (slower but always works)
# - "expand_dims": Add size-1 batch dims to unbatched inputs
