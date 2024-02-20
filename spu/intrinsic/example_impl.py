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

__all__ = ["example"]

from functools import partial

from jax import core, dtypes
from jax.core import ShapedArray
from jax.interpreters import ad, batching, mlir, xla

# from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call


# Public facing interface
def example(input):
    return _example_prim.bind(input)


# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************


# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _example_abstract(input):
    shape = input.shape
    dtype = dtypes.canonicalize_dtype(input.dtype)
    return ShapedArray(shape, dtype)


# We also need a lowering rule to provide an MLIR "lowering" of out primitive.
def _example_lowering(ctx, input):
    # The inputs and outputs all have the same shape and memory layout
    # so let's predefine this specification
    dtype = mlir.ir.RankedTensorType(input.type)

    call = custom_call(
        "example",
        # Output types
        result_types=[dtype],
        # The inputs:
        operands=[input],
    )

    return call.results


# **********************************
# *  SUPPORT FOR FORWARD AUTODIFF  *
# **********************************


def _example_jvp(args, tangents):
    raise NotImplementedError()


# ************************************
# *  SUPPORT FOR BATCHING WITH VMAP  *
# ************************************


# Our op already supports arbitrary dimensions so the batching rule is quite
# simple. The jax.lax.linalg module includes some example of more complicated
# batching rules if you need such a thing.
def _example_batch(args, axes):
    assert axes[0] == axes[1]
    return example(*args), axes


# *********************************************
# *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
# *********************************************
_example_prim = core.Primitive("example")
_example_prim.multiple_results = False
_example_prim.def_impl(partial(xla.apply_primitive, _example_prim))
_example_prim.def_abstract_eval(_example_abstract)

mlir.register_lowering(_example_prim, _example_lowering)

# Connect the JVP and batching rules
ad.primitive_jvps[_example_prim] = _example_jvp
batching.primitive_batchers[_example_prim] = _example_batch
