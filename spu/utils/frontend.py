# Copyright 2022 Ant Group Co., Ltd.
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

import functools
from enum import Enum
from typing import Callable, Dict, List
import warnings

from cachetools import LRUCache, cached

from .. import api as spu_api
from .. import spu_pb2


def _jax_compilation_key(fn: Callable, static_argnums, args: List, kwargs: Dict):
    import jax

    flat_args, _ = jax.tree_util.tree_flatten((args, kwargs))
    types = [(a.dtype, a.shape) if hasattr(a, 'dtype') else type(a) for a in flat_args]
    # A lambda capture can be constant in compiled XLA, so add to hash
    if hasattr(fn, '__closure__'):
        return f'{fn}-{static_argnums}-{types}-{fn.__closure__}'
    else:
        return f'{fn}-{static_argnums}-{types}'


@cached(cache=LRUCache(maxsize=128), key=_jax_compilation_key)
def _jax_compilation(fn: Callable, static_argnums, args: List, kwargs: Dict):
    import jax

    cfn, output = jax.xla_computation(
        fn, return_shape=True, static_argnums=static_argnums, backend="interpreter"
    )(*args, **kwargs)
    return cfn.as_serialized_hlo_module_proto(), output


## Frontend patches


def _patch_fcn(obj, func_name, wrapped_func):
    if hasattr(obj, func_name):
        old_fcn = getattr(obj, func_name)
        setattr(obj, func_name, wrapped_func)
        return old_fcn
    else:
        warnings.warn(f'Failed to patch {func_name} in {obj}')


# This is a fix to float->int cast in lax.sort
def _patched_lax_float_to_int_for_sort(x):
    return x


lax_patches = {
    # lax sort has  float->int bitcast which is causing problems on MPC protocols using fixed-point
    # Replace this function with a no-op
    '_float_to_int_for_sort': _patched_lax_float_to_int_for_sort,
}


def _patch_jax():
    import jax._src.lax.lax as lax

    patch_history = {}
    for fcn_name, fcn in lax_patches.items():
        patch_history[fcn_name] = _patch_fcn(lax, fcn_name, fcn)

    return patch_history


def _restore_jax_patch(patch_history):
    import jax._src.lax.lax as lax

    for fcn_name, fcn in patch_history.items():
        _patch_fcn(lax, fcn_name, fcn)


##


class Kind(Enum):
    JAX = 1
    Tensorflow = 2
    Torch = 3


def compile(
    kind: Kind,
    fn: Callable,
    args: List,
    kwargs: Dict,
    input_names: List[str],
    input_vis: List,
    outputNameGen: Callable,
    static_argnums=(),
):
    if kind == Kind.JAX:
        import jax

        patches = _patch_jax()

        ir_text, output = _jax_compilation(fn, static_argnums, args, kwargs)

        _restore_jax_patch(patches)

        output_flat, _ = jax.tree_util.tree_flatten(output)
        output_names = outputNameGen(output_flat)

    elif kind == Kind.Tensorflow:
        import tensorflow as tf

        tf_fn = tf.function(fn, jit_compile=True, experimental_relax_shapes=True)
        ir_text = tf_fn.experimental_get_compiler_ir(*args, **kwargs)(
            stage="hlo_serialized",
        )

        cf = tf_fn.get_concrete_function(*args, **kwargs)

        # TODO(junfeng): support input captures.
        assert (
            len(cf.captured_inputs) == 0
        ), "captured_inputs in TensorFlow functions is unsupported."

        output = cf.structured_outputs
        output_names = outputNameGen(cf.outputs)
    elif kind == Kind.Torch:
        import jax
        import torch
        import torch_mlir
        from torch_mlir._mlir_libs._mlir.ir import Attribute, Context

        assert isinstance(
            fn, torch.nn.Module
        ), "currently only torch.nn.Module is supported"

        # convert numpy.ndarray to torch tensor as torch_mlir required
        arg_tensors = [torch.Tensor(arg) for arg in args]
        # get mlir module
        module = torch_mlir.compile(
            fn, arg_tensors, output_type=torch_mlir.OutputType.MHLO
        )
        # get mlir func op of torch.nn.Module.forward function
        func_op = module.body.operations[0]
        # rename func name from 'forward' to 'main'
        with Context():
            func_op.attributes["sym_name"] = Attribute.parse('"main"')

        # parse output_num from func op signature string
        func_sig = func_op.attributes["function_type"]
        output_num = len(str(func_sig).split("->")[1].split(","))
        # get mhlo
        ir_text = bytes(str(module), 'utf-8')
        # mock output
        output = [0] * output_num
        output_names = outputNameGen(output)
        output = tuple(output) if output_num > 1 else output[0]
        _, output = jax.tree_util.tree_flatten(output)
    else:
        raise NameError(f"Unknown frontend type {kind}")

    if kind in [Kind.JAX, Kind.Tensorflow]:
        ir_type = "hlo"
        name = fn.func.__name__ if isinstance(fn, functools.partial) else fn.__name__
    elif kind == Kind.Torch:
        ir_type = "mhlo"
        name = repr(fn)
    mlir = spu_api.compile(ir_text, ir_type, input_vis)
    executable = spu_pb2.ExecutableProto(
        name=name,
        input_names=input_names,
        output_names=output_names,
        code=mlir,
    )
    return executable, output
