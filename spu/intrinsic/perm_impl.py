__all__ = ["perm"]

from functools import partial

from jax import core, dtypes
from jax.core import ShapedArray
from jax.interpreters import ad, batching, mlir, xla

# from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call


# Public facing interface
def perm(in1, in2):
    # Add necessary preprocessing code
    return _perm_prim.bind(in1, in2)


# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************


# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _perm_abstract(in1, in2):
    shape = in1.shape
    dtype = dtypes.canonicalize_dtype(in1.dtype)
    return ShapedArray(shape, dtype)


# We also need a lowering rule to provide an MLIR "lowering" of out primitive.
def _perm_lowering(ctx, in1, in2):
    # The inputs and outputs all have the same shape and memory layout
    # so let's predefine this specification
    dtype = mlir.ir.RankedTensorType(in1.type)
    shape = dtype.shape
    result_type = mlir.ir.RankedTensorType.get(shape, dtype.element_type)

    return custom_call(
        "perm",
        # Output types
        result_types=[result_type],
        # The inputs:
        operands=[in1, in2],
    ).results


# **********************************
# *  SUPPORT FOR FORWARD AUTODIFF  *
# **********************************


def _perm_jvp(args, tangents):
    raise NotImplementedError()


# ************************************
# *  SUPPORT FOR BATCHING WITH VMAP  *
# ************************************


# Our op already supports arbitrary dimensions so the batching rule is quite
# simple. The jax.lax.linalg module includes some example of more complicated
# batching rules if you need such a thing.
def _perm_batch(args, axes):
    raise NotImplementedError()


# *********************************************
# *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
# *********************************************
_perm_prim = core.Primitive("perm")
# Change this to True if there are more than 1 output
_perm_prim.multiple_results = False
_perm_prim.def_impl(partial(xla.apply_primitive, _perm_prim))
_perm_prim.def_abstract_eval(_perm_abstract)

mlir.register_lowering(_perm_prim, _perm_lowering)

# Connect the JVP and batching rules
ad.primitive_jvps[_perm_prim] = _perm_jvp
batching.primitive_batchers[_perm_prim] = _perm_batch
