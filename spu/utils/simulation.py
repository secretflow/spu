# Copyright 2021 Ant Group Co., Ltd.
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


import threading
from typing import Callable

import jax

try:
    import jax.extend.linear_util as jax_lu
except ImportError:
    import jax.linear_util as jax_lu  # fallback

import jax.numpy as jnp
import numpy as np
from jax._src import api_util as japi_util

from .. import api as spu_api
from .. import libspu  # type: ignore
from .. import spu_pb2
from . import frontend as spu_fe


# https://stackoverflow.com/questions/2829329/catch-a-threads-exception-in-the-caller-thread-in-python
class PropagatingThread(threading.Thread):
    def run(self):
        self.exc = None
        try:
            self.ret = self._target(*self._args, **self._kwargs)
        except BaseException as e:
            self.exc = e

    def join(self):
        super(PropagatingThread, self).join()
        if self.exc:
            raise self.exc
        return self.ret


class Simulator(object):
    def __init__(self, wsize: int, rt_config: spu_pb2.RuntimeConfig):
        self.wsize = wsize
        self.rt_config = rt_config
        self.io = spu_api.Io(wsize, rt_config)

    @classmethod
    def simple(cls, wsize: int, prot: spu_pb2.ProtocolKind, field: spu_pb2.FieldType):
        """helper method to create an SPU Simulator

        Args:
            wsize (int): the world size.

            prot (spu_pb2.ProtocolKind): protocol.

            field (spu_pb2.FieldType): field type.

        Returns:
            A SPU Simulator
        """
        config = spu_pb2.RuntimeConfig(protocol=prot, field=field)

        if prot == spu_pb2.ProtocolKind.CHEETAH:
            # config.cheetah_2pc_config.enable_mul_lsb_error = True
            # config.cheetah_2pc_config.ot_kind = spu_pb2.CheetahOtKind.YACL_Softspoken
            pass
        # config.enable_hal_profile = True
        # config.enable_pphlo_profile = True
        # config.enable_pphlo_trace = True
        # config.enable_action_trace = True
        # config.enable_type_checker = True
        return cls(wsize, config)

    def __call__(self, executable, *flat_args):
        flat_args = [np.array(jnp.array(x)) for x in flat_args]
        params = [
            self.io.make_shares(x, spu_pb2.Visibility.VIS_SECRET) for x in flat_args
        ]

        lctx_desc = libspu.link.Desc()
        for rank in range(self.wsize):
            lctx_desc.add_party(f"id_{rank}", f"thread_{rank}")

        def wrapper(rank):
            lctx = libspu.link.create_mem(lctx_desc, rank)
            rank_config = spu_pb2.RuntimeConfig()
            rank_config.CopyFrom(self.rt_config)
            if rank != 0:
                # rank_config.enable_pphlo_trace = False
                rank_config.enable_action_trace = False
                rank_config.enable_hal_profile = False
                rank_config.enable_pphlo_profile = False
            rt = spu_api.Runtime(lctx, rank_config)

            # do infeed.
            for idx, param in enumerate(params):
                rt.set_var(executable.input_names[idx], param[rank])

            # run
            rt.run(executable)

            # do outfeed
            return [rt.get_var(name) for name in executable.output_names]

        jobs = [
            PropagatingThread(target=wrapper, args=(rank,))
            for rank in range(self.wsize)
        ]

        [job.start() for job in jobs]
        parties = [job.join() for job in jobs]

        outputs = zip(*parties)
        return [self.io.reconstruct(out) for out in outputs]


def sim_jax(
    sim: Simulator,
    fun: Callable,
    static_argnums=(),
    copts=spu_pb2.CompilerOptions(),
):
    """
    Decorates a jax numpy fn that simulated on SPU.

        >>> sim = Simulator.simple(3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64)
        >>> spu_fn = sim_jax(sim, jnp.add)

    Then we can call spu_fn like normal jnp fn.

        >>> x = np.array([[1, 2], [3, 4]])
        >>> y = np.array([[5, 6], [7, 8]])
        >>> z = spu_fn(x, y)

    The function will be evaluated in an spu simulator.
    """

    def wrapper(*args, **kwargs):
        _, dyn_args = japi_util.argnums_partial_except(
            jax_lu.wrap_init(fun), static_argnums, args, allow_invalid=False
        )
        args_flat, _ = jax.tree_util.tree_flatten((dyn_args, kwargs))

        in_names = [f'in{idx}' for idx in range(len(args_flat))]

        def outputNameGen(out_flat):
            return [f'out{idx}' for idx in range(len(out_flat))]

        executable, output = spu_fe.compile(
            spu_fe.Kind.JAX,
            fun,
            args,
            kwargs,
            in_names,
            [spu_pb2.Visibility.VIS_SECRET] * len(args_flat),
            outputNameGen,
            static_argnums=static_argnums,
            copts=copts,
        )

        wrapper.pphlo = executable.code.decode("utf-8")

        out_flat = sim(executable, *args_flat)

        _, output_tree = jax.tree_util.tree_flatten(output)

        return jax.tree_util.tree_unflatten(output_tree, out_flat)

    return wrapper
