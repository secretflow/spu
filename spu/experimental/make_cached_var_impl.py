# Copyright 2024 Ant Group Co., Ltd.
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

"""make_cached_var intrinsic: marks a tensor for Beaver triple caching in MPC."""

__all__ = ["make_cached_var"]

import jax


def make_cached_var(input):
    """Mark tensor for Beaver triple caching. Identity op with caching side effect."""
    return _make_cached_var_call(input)


# Wrap with custom_jvp (outer) and custom_vjp (inner) for both AD modes
@jax.custom_jvp
@jax.custom_vjp
def _make_cached_var_call(input):
    return _make_cached_var_impl(input)


@_make_cached_var_call.defjvp
def _make_cached_var_jvp(primals, tangents):
    (input,) = primals
    (input_dot,) = tangents
    # Linear: tangent passes through
    return _make_cached_var_call(input), input_dot


def _make_cached_var_impl(input):
    return jax.ffi.ffi_call(
        "spu.make_cached_var",
        jax.ShapeDtypeStruct(input.shape, input.dtype),
        has_side_effect=True,
        vmap_method="broadcast_all",
    )(input)


def _make_cached_var_fwd(input):
    return _make_cached_var_call(input), None  # No residuals needed


def _make_cached_var_bwd(res, g):
    return (g,)  # Identity: gradient passes through


_make_cached_var_call.defvjp(_make_cached_var_fwd, _make_cached_var_bwd)
