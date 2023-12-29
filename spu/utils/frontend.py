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
from threading import Lock
from typing import Callable, Dict, Iterable, List
from numpy import ndarray

from cachetools import LRUCache, cached

from .. import api as spu_api
from .. import spu_pb2

_jax_lock = Lock()


def _jax_compilation_key(
    fn: Callable, static_argnums, static_argnames, args: List, kwargs: Dict
):
    import jax
    import numpy as np
    from jax._src.api_util import argnames_partial_except, argnums_partial_except

    try:
        from jax.extend.linear_util import wrap_init  # Moved in jax 0.4.16
    except ImportError:
        from jax.linear_util import wrap_init

    def _function_contents(func):
        try:
            closure = []
            if hasattr(func, '__closure__') and func.__closure__:
                for cell in func.__closure__:
                    if isinstance(cell.cell_contents, ndarray):
                        closure.append(cell.cell_contents.ctypes.data)
                    else:
                        closure.append(cell.cell_contents)
            return (
                func.__name__,
                func.__defaults__,
                tuple(closure),
                func.__code__.co_code,
                func.__code__.co_consts,
            )
        except AttributeError:
            # Not a standard func
            return func

    f = wrap_init(fn)
    f, dkwargs = argnames_partial_except(f, static_argnames, kwargs)
    f, dargs = argnums_partial_except(f, static_argnums, args, allow_invalid=True)

    flat_args, tree = jax.tree_util.tree_flatten((dargs, dkwargs))
    types = []
    for a in flat_args:
        if hasattr(a, 'dtype'):
            types.append((a.dtype, a.shape))
        else:
            np_array = np.asarray(a)
            types.append((np_array.dtype, np_array.shape))
    hash_str = f'{hash(f)}-{hash(_function_contents(fn))}-{types}-{hash(tree)}'

    return hash_str


def _argnames_partial_except(fn, static_argnames, kwargs):
    if static_argnames is None:
        return fn, kwargs

    assert isinstance(
        static_argnames, (str, Iterable)
    ), f'type of static_argnames is {type(static_argnames)} while str or Iterable is required here.'
    if isinstance(static_argnames, str):
        static_argnames = (static_argnames,)

    static_kwargs = {k: kwargs.pop(k) for k in static_argnames if k in kwargs}
    return functools.partial(fn, **static_kwargs), kwargs


# FIXME: Figure out a proper way to hash lambda functions
# @cached(cache=LRUCache(maxsize=128), key=_jax_compilation_key)
def _jax_compilation(
    fn: Callable, static_argnums, static_argnames, args: List, kwargs: Dict
):
    import jax
    from jax._src.xla_bridge import _backend_lock, _backends, register_backend_factory
    from jax._src.lib import xla_client, xla_extension_version

    # Register interpreter backend since we don't want any cpu/gpu/tpu specific optimization
    if xla_extension_version < 164:
        # interpreter is registerd by default before jaxlib 0.4.13
        pass
    else:
        has_interpreter_backend = False
        with _backend_lock:
            if 'interpreter' in _backends:
                has_interpreter_backend = True
        if not has_interpreter_backend:
            if xla_extension_version < 194:
                # make_interpreter_client has been removed after jaxlib 0.4.16
                register_backend_factory(
                    'interpreter', xla_client.make_interpreter_client, priority=-100
                )
            else:
                from jax.interpreters.xla import Backend as xla_back

                register_backend_factory('interpreter', xla_back, priority=-100)

    fn, kwargs = _argnames_partial_except(fn, static_argnames, kwargs)

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
        return None


# This is a fix to float->int cast in lax.sort
def _patched_lax_float_to_int_for_sort(x):
    return x


class FcnReplaceLibrary:
    possible_names = []
    replace_fcn = None

    def __init__(self, names, rf):
        self.possible_names = names
        self.replace_fcn = rf


lax_patches = {
    # lax sort has  float->int bitcast which is causing problems on MPC protocols using fixed-point
    # Replace this function with a no-op
    'sort': FcnReplaceLibrary(
        ['_float_to_int_for_sort', '_canonicalize_float_for_sort'],
        _patched_lax_float_to_int_for_sort,
    )
}


def _patch_jax():
    import jax._src.lax.lax as lax

    patch_history = {}
    for fcn_name, replace in lax_patches.items():
        success = False
        for name in replace.possible_names:
            old_fcn = _patch_fcn(lax, name, replace.replace_fcn)
            if old_fcn:
                patch_history[name] = old_fcn
                success = True
                break
        if not success:
            raise RuntimeError(
                f'Failed to patch {fcn_name}, current jax version is not compatible with SPU'
            )

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
    m_args: List,
    m_kwargs: Dict,
    input_names: List[str],
    input_vis: List,
    outputNameGen: Callable,
    static_argnums=(),
    static_argnames=None,
    copts=spu_pb2.CompilerOptions(),
):
    if kind == Kind.JAX:
        import jax

        with _jax_lock:
            patches = _patch_jax()

            ir_text, output = _jax_compilation(
                fn, static_argnums, static_argnames, m_args, m_kwargs
            )

            _restore_jax_patch(patches)

            output_flat, _ = jax.tree_util.tree_flatten(output)
            output_names = outputNameGen(output_flat)

    elif kind == Kind.Tensorflow:
        import tensorflow as tf

        tf_fn = tf.function(fn, jit_compile=True, experimental_relax_shapes=True)
        ir_text = tf_fn.experimental_get_compiler_ir(*m_args, **m_kwargs)(
            stage="hlo_serialized",
        )

        cf = tf_fn.get_concrete_function(*m_args, **m_kwargs)

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
        arg_tensors = [torch.Tensor(arg) for arg in m_args]
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

    source = spu_pb2.CompilationSource()
    source.ir_txt = ir_text
    source.input_visibility.extend(input_vis)
    if kind in [Kind.JAX, Kind.Tensorflow]:
        source.ir_type = spu_pb2.SourceIRType.XLA
        name = fn.func.__name__ if isinstance(fn, functools.partial) else fn.__name__
    elif kind == Kind.Torch:
        source.ir_type = spu_pb2.SourceIRType.MLIR_HLO
        name = repr(fn)
    mlir = spu_api.compile(source, copts)
    executable = spu_pb2.ExecutableProto(
        name=name,
        input_names=input_names,
        output_names=output_names,
        code=mlir,
    )
    return executable, output
