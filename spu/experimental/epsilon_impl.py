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

"""Epsilon intrinsic: returns 2^(-fxp_fraction_bits), the smallest fixed-point value."""

__all__ = ["epsilon"]

import jax
import jax.numpy as jnp
from jax.core import ShapedArray
from jax.extend import core
from jax.interpreters import ad, batching
from jax.interpreters.mlir import register_lowering


def epsilon():
    """Return SPU epsilon (2^-fxp_fraction_bits). On CPU, returns 2^-18."""
    return _epsilon_prim.bind()


def _epsilon_abstract():
    """Output shape/dtype: scalar float32."""
    return ShapedArray((), jnp.float32)


def _epsilon_lowering(ctx, *args, **kwargs):
    """MLIR lowering: SPU uses FFI custom_call, CPU returns constant."""
    # Detect platform from MLIR context
    platform = (
        ctx.module_context.platforms[0]
        if hasattr(ctx, 'module_context') and hasattr(ctx.module_context, 'platforms')
        else "cpu"
    )

    if platform == "interpreter":
        # SPU backend: generate FFI custom_call op
        return jax.ffi.ffi_lowering(
            "spu.epsilon",
            operand_layouts=[],
            result_layouts=[()],
            has_side_effect=True,
        )(ctx, *args, **kwargs)
    else:
        # CPU fallback: return compile-time constant
        import jax._src.interpreters.mlir as mlir_impl

        constant_val = mlir_impl.ir_constant(jnp.array(2**-18, dtype=jnp.float32))
        return [constant_val]


# Register primitive
_epsilon_prim = core.Primitive("epsilon")
_epsilon_prim.multiple_results = False
_epsilon_prim.def_abstract_eval(_epsilon_abstract)
_epsilon_prim.def_impl(lambda: jnp.array(2**-18, dtype=jnp.float32))  # CPU impl
register_lowering(_epsilon_prim, _epsilon_lowering)

# Autodiff: constant has zero gradient
ad.primitive_transposes[_epsilon_prim] = lambda ct: []
ad.primitive_jvps[_epsilon_prim] = lambda primals, tangents: (
    epsilon(),
    jnp.zeros((), dtype=jnp.float32),  # tangent is zero
)

# Batching: scalar output, no batch dimension
batching.primitive_batchers[_epsilon_prim] = lambda args, dims: (epsilon(), None)
