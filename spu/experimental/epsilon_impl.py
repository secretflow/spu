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

__all__ = ["epsilon"]

from functools import partial

import jax.numpy as jnp
import numpy as np
from jax._src.core import ShapedArray
from jax.extend import core
from jax.interpreters import ad, batching, mlir, xla
from jaxlib.hlo_helpers import custom_call


# Public facing interface
def epsilon() -> np.ndarray:
    return _epsilon_prim.bind()


# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _epsilon_abstract():
    return ShapedArray((), jnp.float32)


# We also need a lowering rule to provide an MLIR "lowering" of out primitive.
def _epsilon_lowering(ctx, *args, **kwargs):
    # Check current platform
    platform = (
        ctx.module_context.platforms[0]
        if hasattr(ctx, 'module_context') and hasattr(ctx.module_context, 'platforms')
        else "cpu"
    )

    if platform == "interpreter":
        # Create proper MLIR type for scalar float32
        f32_type = mlir.ir.F32Type.get()
        dtype = mlir.ir.RankedTensorType.get([], f32_type)
        # For SPU, use custom_call
        call = custom_call(
            "spu.epsilon",
            # Output types
            result_types=[dtype],
            # The inputs:
            operands=[],
            has_side_effect=True,
        )

        return call.results
    else:
        # For now, return a simple constant implementation
        # This creates a scalar constant with value 2^-18
        import jax._src.interpreters.mlir as mlir_impl

        constant_val = mlir_impl.ir_constant(jnp.array(2**-18, dtype=jnp.float32))
        return [constant_val]


# *********************************************
# *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
# *********************************************
_epsilon_prim = core.Primitive("epsilon")
_epsilon_prim.multiple_results = False
_epsilon_prim.def_impl(partial(xla.apply_primitive, _epsilon_prim))
_epsilon_prim.def_abstract_eval(_epsilon_abstract)

# Register MLIR lowering
mlir.register_lowering(_epsilon_prim, _epsilon_lowering)


def _make_epsilon_transpose(ct):
    # Since epsilon has no inputs, transpose just returns empty list
    return []


# Connect the JVP and batching rules
ad.primitive_jvps[_epsilon_prim] = partial(ad.linear_jvp, _epsilon_prim)
ad.primitive_transposes[_epsilon_prim] = _make_epsilon_transpose


def _epsilon_batch(batched_args, batch_dims):
    # Since epsilon has no inputs, just return epsilon() with no batch dimension
    return epsilon(), None


batching.primitive_batchers[_epsilon_prim] = _epsilon_batch
