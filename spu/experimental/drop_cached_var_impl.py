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

__all__ = ["drop_cached_var"]

from functools import partial

from jax import core

# from jax.abstract_arrays import ShapedArray
from jax.interpreters import ad, batching, mlir, xla

# from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call


# Public facing interface
def drop_cached_var(input, *dependencies):
    # Add necessary preprocessing code
    return _drop_cached_var_prim.bind(input, *dependencies)


# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************


# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _drop_cached_var_abstract(input, *dependencies):
    return core.ShapedArray(input.shape, input.dtype)


# We also need a lowering rule to provide an MLIR "lowering" of out primitive.
def _drop_cached_var_lowering(ctx, input, *dependencies):
    # The inputs and outputs all have the same shape and memory layout
    # so let's predefine this specification
    dtype = mlir.ir.RankedTensorType(input.type)

    return custom_call(
        "spu.drop_cached_var",
        # Output types
        result_types=[dtype],
        # The inputs:
        operands=[input, *dependencies],
        has_side_effect=True,
    ).results


# *********************************************
# *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
# *********************************************
_drop_cached_var_prim = core.Primitive("drop_cached_var")
# Change this to True if there are more than 1 output
_drop_cached_var_prim.multiple_results = False
_drop_cached_var_prim.def_impl(partial(xla.apply_primitive, _drop_cached_var_prim))
_drop_cached_var_prim.def_abstract_eval(_drop_cached_var_abstract)

mlir.register_lowering(_drop_cached_var_prim, _drop_cached_var_lowering)


def _drop_cached_var_transpose(ct, input, *dependencies):
    return [ct] * (len(dependencies) + 1)


# Connect the JVP and batching rules
ad.primitive_jvps[_drop_cached_var_prim] = partial(ad.linear_jvp, _drop_cached_var_prim)
ad.primitive_transposes[_drop_cached_var_prim] = _drop_cached_var_transpose


def _drop_cached_var_batch(args, axes):
    res = _drop_cached_var_prim(*args)
    return res, axes[0]


batching.primitive_batchers[_drop_cached_var_prim] = _drop_cached_var_batch
