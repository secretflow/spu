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

__all__ = ["make_cached_var"]

from functools import partial

from jax import ShapeDtypeStruct
from jax.core import ShapedArray
from jax.extend import core

# from jax.abstract_arrays import ShapedArray
from jax.interpreters import ad, batching, xla
from jax.extend.mlir import ir

# from jax.lib import xla_client
from jax.ffi import ffi_call
import jax.numpy as jnp


# Public facing interface
def make_cached_var(input):
    return _make_cached_var_prim.bind(input)


# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _make_cached_var_abstract(input):
    return ShapedArray(input.shape, input.dtype)


# We also need a lowering rule to provide an MLIR "lowering" of out primitive.
def _make_cached_var_lowering(ctx, input):
    # The inputs and outputs all have the same shape and memory layout
    # so let's predefine this specification
    input_type = ir.RankedTensorType(input.type)
    
    # Use JAX's built-in utilities to handle dtype conversion
    shape_dtype = ShapeDtypeStruct(input_type.shape, input_type.element_type)

    # Use the new ffi_call API
    call_result = ffi_call(
        "spu.make_cached_var",
        # Output types
        result_shape_dtypes=[shape_dtype],
        has_side_effect=True,
        # Vmap method for batching support
        vmap_method="broadcast_all",
    )(input)

    return call_result


# *********************************************
# *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
# *********************************************
_make_cached_var_prim = core.Primitive("make_cached_var")
_make_cached_var_prim.def_impl(partial(xla.apply_primitive, _make_cached_var_prim))
_make_cached_var_prim.def_abstract_eval(_make_cached_var_abstract)

# Register the lowering rule
from jax.interpreters.mlir import register_lowering
register_lowering(_make_cached_var_prim, _make_cached_var_lowering)


def _make_cached_var_transpose(ct, input):
    return [ct]


# Connect the JVP and batching rules
ad.primitive_jvps[_make_cached_var_prim] = partial(ad.linear_jvp, _make_cached_var_prim)
ad.primitive_transposes[_make_cached_var_prim] = _make_cached_var_transpose


def _make_cached_var_batch(args, axes):
    res = _make_cached_var_prim(*args)
    return res, axes[0]


batching.primitive_batchers[_make_cached_var_prim] = _make_cached_var_batch
