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

__all__ = ["reveal"]

from functools import partial

from jax import core
from jax.interpreters import ad, batching, mlir, xla
from jaxlib.hlo_helpers import custom_call
import numpy as np


# Public facing interface
def reveal(input: np.ndarray) -> np.ndarray:
    return _reveal_prim.bind(input)


# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _reveal_abstract(input):
    return core.ShapedArray(input.shape, input.dtype)


# We also need a lowering rule to provide an MLIR "lowering" of out primitive.
def _reveal_lowering(ctx, input):
    # Check current platform
    platform = (
        ctx.module_context.platforms[0]
        if hasattr(ctx, 'module_context') and hasattr(ctx.module_context, 'platforms')
        else "cpu"
    )

    if platform == "interpreter":
        # For SPU, use custom_call
        dtype = mlir.ir.RankedTensorType(input.type)

        call = custom_call(
            "spu.reveal",
            # Output types
            result_types=[dtype],
            # The inputs:
            operands=[input],
            has_side_effect=True,
        )

        return call.results
    else:
        return [input]


# *********************************************
# *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
# *********************************************
_reveal_prim = core.Primitive("reveal")
_reveal_prim.multiple_results = False
_reveal_prim.def_impl(partial(xla.apply_primitive, _reveal_prim))
_reveal_prim.def_abstract_eval(_reveal_abstract)

# Register MLIR lowering
mlir.register_lowering(_reveal_prim, _reveal_lowering)


def _make_reveal_transpose(ct, input):
    return [ct]


# Connect the JVP and batching rules
ad.primitive_jvps[_reveal_prim] = partial(ad.linear_jvp, _reveal_prim)
ad.primitive_transposes[_reveal_prim] = _make_reveal_transpose


def _reveal_batch(batched_args, batch_dims):
    (x,) = batched_args
    (bd,) = batch_dims
    return reveal(x), bd


batching.primitive_batchers[_reveal_prim] = _reveal_batch
