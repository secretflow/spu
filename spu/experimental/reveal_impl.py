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

"""reveal intrinsic: converts secret MPC values to public (all parties see plaintext)."""

__all__ = ["reveal"]

import jax
from jax.core import ShapedArray
from jax.extend import core
from jax.interpreters import ad, batching
from jax.interpreters.mlir import register_lowering


def reveal(input):
    """Reveal secret value to all parties. On CPU, this is identity."""
    return _reveal_prim.bind(input)


def _reveal_abstract(input):
    return ShapedArray(input.shape, input.dtype)


def _reveal_lowering(ctx, input):
    """SPU: FFI custom_call for reveal. CPU: identity."""
    platform = (
        ctx.module_context.platforms[0]
        if hasattr(ctx, 'module_context') and hasattr(ctx.module_context, 'platforms')
        else "cpu"
    )

    if platform == "interpreter":
        return jax.ffi.ffi_lowering(
            "spu.reveal",
            operand_layouts=None,
            result_layouts=None,
            has_side_effect=True,
        )(ctx, input)
    else:
        return [input]  # CPU: identity


_reveal_prim = core.Primitive("reveal")
_reveal_prim.multiple_results = False
_reveal_prim.def_abstract_eval(_reveal_abstract)
_reveal_prim.def_impl(lambda input: input)
register_lowering(_reveal_prim, _reveal_lowering)


# Autodiff: identity-like, gradients pass through
def _reveal_jvp(primals, tangents):
    (input,) = primals
    (tangent,) = tangents
    out = reveal(input)
    if type(tangent) is ad.Zero:
        return out, ad.Zero.from_primal_value(out)
    return out, reveal(tangent)


ad.primitive_jvps[_reveal_prim] = _reveal_jvp
ad.primitive_transposes[_reveal_prim] = lambda ct, input: [ct]

# Batching: apply reveal to batched input
batching.primitive_batchers[_reveal_prim] = lambda args, dims: (
    reveal(args[0]),
    dims[0],
)
