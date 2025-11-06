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

__all__ = ["example_binary"]

from functools import partial

import numpy as np
from jax import core, dtypes
from jax.core import ShapedArray
from jax.interpreters import ad, batching, mlir, xla

# from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call


# Public facing interface
def example_binary(in1, in2):
    # Add necessary preprocessing code
    return _example_binary_prim.bind(in1, in2)


# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************


def _compute_result_shape(shape1, shape2):
    return np.shape([shape1[0] + shape2[0], shape1[1] + shape2[1]])


# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _example_binary_abstract(in1, in2):
    in1_shape = in1.shape
    in2_shape = in2.shape
    in1_dtype = dtypes.canonicalize_dtype(in1.dtype)

    assert dtypes.canonicalize_dtype(in2.dtype) == in1_dtype

    result_shape = _compute_result_shape(in1_shape, in2_shape)

    # Concat shape
    return ShapedArray(result_shape, in1_dtype)


# We also need a lowering rule to provide an MLIR "lowering" of out primitive.
def _example_binary_lowering(ctx, in1, in2):
    # The inputs and outputs all have the same shape and memory layout
    # so let's predefine this specification
    in1_dtype = mlir.ir.RankedTensorType(in1.type)
    in2_dtype = mlir.ir.RankedTensorType(in2.type)

    result_shape = _compute_result_shape(in1_dtype.shape, in2_dtype.shape)
    result_type = mlir.ir.RankedTensorType.get(result_shape, in1_dtype.element_type)

    call = custom_call(
        "example_binary",
        # Output types
        result_types=[result_type],
        # The inputs:
        operands=[in1, in2],
    )

    return call.results


# **********************************
# *  SUPPORT FOR FORWARD AUTODIFF  *
# **********************************


def _example_binary_jvp(args, tangents):
    raise NotImplementedError()


# ************************************
# *  SUPPORT FOR BATCHING WITH VMAP  *
# ************************************


# Our op already supports arbitrary dimensions so the batching rule is quite
# simple. The jax.lax.linalg module includes some example of more complicated
# batching rules if you need such a thing.
def _example_binary_batch(args, axes):
    raise NotImplementedError()


# *********************************************
# *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
# *********************************************
_example_binary_prim = core.Primitive("example_binary")
# Change this to True if there are more than 1 output
_example_binary_prim.multiple_results = False
_example_binary_prim.def_impl(partial(xla.apply_primitive, _example_binary_prim))
_example_binary_prim.def_abstract_eval(_example_binary_abstract)

mlir.register_lowering(_example_binary_prim, _example_binary_lowering)

# Connect the JVP and batching rules
ad.primitive_jvps[_example_binary_prim] = _example_binary_jvp
batching.primitive_batchers[_example_binary_prim] = _example_binary_batch
