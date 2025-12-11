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

"""drop_cached_var intrinsic: releases cached Beaver triples for a variable."""

__all__ = ["drop_cached_var"]

import jax
import jax.numpy as jnp


def drop_cached_var(input, *dependencies):
    """Release cached Beaver triples. Dependencies ensure execution order."""
    return _drop_cached_var_call(input, *dependencies)


# Use custom_jvp (outer) + custom_vjp (inner) to support both AD modes
@jax.custom_jvp
@jax.custom_vjp
def _drop_cached_var_call(input, *dependencies):
    """Wrapped call with custom autodiff rules."""
    return _drop_cached_var_impl(input, *dependencies)


@_drop_cached_var_call.defjvp
def _drop_cached_var_jvp(primals, tangents):
    """Forward-mode AD: only input tangent matters, deps are for ordering."""
    input, *deps = primals
    input_dot, *_ = tangents
    return _drop_cached_var_call(input, *deps), input_dot


def _drop_cached_var_impl(input, *dependencies):
    """FFI call to SPU runtime."""
    return jax.ffi.ffi_call(
        "spu.drop_cached_var",
        jax.ShapeDtypeStruct(input.shape, input.dtype),  # output spec
        has_side_effect=True,
        vmap_method="broadcast_all",  # batch dims pass through
    )(input, *dependencies)


def _drop_cached_var_fwd(input, *dependencies):
    """VJP forward: save dependencies for backward shape info."""
    return _drop_cached_var_call(input, *dependencies), dependencies


def _drop_cached_var_bwd(dependencies, g):
    """VJP backward: gradient to input, zeros to dependencies."""
    return (g,) + tuple(jnp.zeros_like(d) for d in dependencies)


_drop_cached_var_call.defvjp(_drop_cached_var_fwd, _drop_cached_var_bwd)
