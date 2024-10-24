__all__ = ["sparse_dot_general"]

from functools import partial

from jax import core, dtypes
from jax.abstract_arrays import ShapedArray
from jax.interpreters import ad, batching, mlir, xla
import jaxlib.mlir.dialects.stablehlo as hlo
from jax._src.lax.lax import (
    _dot_general_shape_rule,
    _dot_general_dtype_rule,
    _dot_general_lower,
    canonicalize_precision,
)

# from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call
from jax._src import api_util
import numpy as np


# Public facing interface
def sparse_dot_general(
    lhs,
    rhs,
    mask,
    prune_pattern,
    *,
    dimension_numbers,
    precision=None,
    preferred_element_type=None,
):
    lhs_rank = len(lhs.shape)
    rhs_rank = len(rhs.shape)

    # This code is copied from lax.dot_general
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers

    cdims = (
        api_util._ensure_index_tuple(lhs_contract),
        api_util._ensure_index_tuple(rhs_contract),
    )
    bdims = (
        api_util._ensure_index_tuple(lhs_batch),
        api_util._ensure_index_tuple(rhs_batch),
    )

    # Enforce [b, m, k] dot [b, k, n] -> [b, m, n], where b is optional dimension
    assert (
        lhs_rank <= 3 and lhs_rank >= 2
    ), f"expected lhs rank is either 2 or 3, but got {lhs_rank}"
    assert (
        rhs_rank <= 3 and rhs_rank >= 2
    ), f"expected rhs rank is either 2 or 3, but got {rhs_rank}"
    assert rhs.shape == mask.shape, f"expected mask has the same shape as rhs"
    for bdim in bdims:
        if len(bdim) == 0:
            continue
        else:
            assert bdim == (0,), f"expect batch dim == (0,), but got {bdim}"

    assert cdims[0] == (
        lhs_rank - 1,
    ), f"expect lhs contracting dim == (ndim-1,), but got {cdims[0]}"

    if rhs_rank == 2:
        assert cdims[1] == (
            0,
        ), f"expect lhs contracting dim == (0,), but got {cdims[1]}"
    else:
        assert cdims[1] == (
            1,
        ), f"expect lhs contracting dim == (1,), but got {cdims[1]}"

    preferred_element_type = (
        None
        if preferred_element_type is None
        else dtypes.canonicalize_dtype(np.dtype(preferred_element_type))
    )
    return _sparse_dot_general_prim.bind(
        lhs,
        rhs,
        mask,
        prune_pattern,
        dimension_numbers=(cdims, bdims),
        precision=canonicalize_precision(precision),
        preferred_element_type=preferred_element_type,
    )


# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************


# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _sparse_dot_abstract(
    lhs,
    rhs,
    mask,
    prune_pattern,
    *,
    dimension_numbers,
    precision,
    preferred_element_type,
):
    return ShapedArray(
        _dot_general_shape_rule(
            lhs,
            rhs,
            dimension_numbers=dimension_numbers,
            precision=precision,
            preferred_element_type=preferred_element_type,
        ),
        _dot_general_dtype_rule(
            lhs,
            rhs,
            dimension_numbers=dimension_numbers,
            precision=precision,
            preferred_element_type=preferred_element_type,
        ),
    )


# We also need a lowering rule to provide an MLIR "lowering" of out primitive.
def _sparse_dot_lowering(
    ctx,
    lhs,
    rhs,
    mask,
    prune_pattern,
    *,
    dimension_numbers,
    precision,
    preferred_element_type,
):
    # The inputs and outputs all have the same shape and memory layout
    # so let's predefine this specification
    (aval_out,) = ctx.avals_out
    return custom_call(
        "sparse_dot_general",
        # Output types
        out_types=[mlir.aval_to_ir_type(aval_out)],
        # The inputs:
        operands=[lhs, rhs, mask, prune_pattern],
    )


def _sparse_dot_lowering_default(
    ctx,
    lhs,
    rhs,
    mask,
    prune_pattern,
    *,
    dimension_numbers,
    precision,
    preferred_element_type,
):
    # Drop mask and prune_pattern
    ctx.avals_in = ctx.avals_in[0:2]
    return _dot_general_lower(
        ctx,
        lhs,
        rhs,
        dimension_numbers=dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )


# *********************************************
# *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
# *********************************************
_sparse_dot_general_prim = core.Primitive("sparse_dot_general")
# Change this to True if there are more than 1 output
_sparse_dot_general_prim.multiple_results = False
_sparse_dot_general_prim.def_impl(
    partial(xla.apply_primitive, _sparse_dot_general_prim)
)
_sparse_dot_general_prim.def_abstract_eval(_sparse_dot_abstract)

mlir.register_lowering(
    _sparse_dot_general_prim, _sparse_dot_lowering, platform="interpreter"
)
mlir.register_lowering(_sparse_dot_general_prim, _sparse_dot_lowering_default)

# # Connect the JVP and batching rules
# ad.primitive_jvps[_sparse_dot_general_prim] = _sparse_dot_jvp
# batching.primitive_batchers[_sparse_dot_general_prim] = _sparse_dot_batch
