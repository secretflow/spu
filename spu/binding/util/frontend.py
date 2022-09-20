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

from typing import List, Callable, Dict
from cachetools import LRUCache, cached

from enum import Enum
import functools

import spu.binding.api as spuapi
import spu.spu_pb2 as spu_pb2


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


class Kind(Enum):
    JAX = 1
    Tensorflow = 2


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

        xla, output = _jax_compilation(fn, static_argnums, args, kwargs)
        output_flat, _ = jax.tree_util.tree_flatten(output)
        output_names = outputNameGen(output_flat)
    elif kind == Kind.Tensorflow:
        import tensorflow as tf

        tf_fn = tf.function(fn, jit_compile=True, experimental_relax_shapes=True)
        xla = tf_fn.experimental_get_compiler_ir(*args, **kwargs)(
            stage="hlo_serialized",
        )

        cf = tf_fn.get_concrete_function(*args, **kwargs)

        # TODO(junfeng): support input captures.
        assert (
            len(cf.captured_inputs) == 0
        ), "captured_inputs in TensorFlow functions is unsupported."

        output = cf.structured_outputs
        output_names = outputNameGen(cf.outputs)
    else:
        raise NameError(f"Unknown frontend type {kind}")

    mlir = spuapi.compile(xla, input_vis)

    name = fn.func.__name__ if isinstance(fn, functools.partial) else fn.__name__
    executable = spu_pb2.ExecutableProto(
        name=name,
        input_names=input_names,
        output_names=output_names,
        code=mlir,
    )
    return executable, output
