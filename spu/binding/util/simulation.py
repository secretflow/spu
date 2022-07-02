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
import jax.numpy as jnp
from jax import linear_util as lu
from jax._src import api_util as japi_util
import numpy as np

import spu.binding._lib.link as link
import spu.binding.api as ppapi
import spu.spu_pb2 as spu_pb2


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
        self.io = ppapi.Io(wsize, rt_config)

    @classmethod
    def simple(cls, wsize, prot, field):
        config = spu_pb2.RuntimeConfig(protocol=prot, field=field)
        # config.enable_hal_profile = True
        # config.enable_pphlo_profile = True
        # config.enable_pphlo_trace = True
        # config.enable_action_trace = True
        # config.enable_type_checker = True
        return cls(wsize, config)

    def __call__(self, text, num_returns, *flat_args):
        flat_args = [np.array(jnp.array(x)) for x in flat_args]
        params = [
            self.io.make_shares(x, spu_pb2.Visibility.VIS_SECRET) for x in flat_args
        ]

        lctx_desc = link.Desc()
        for rank in range(self.wsize):
            lctx_desc.add_party(f"id_{rank}", f"thread_{rank}")

        def wrapper(rank):
            lctx = link.create_mem(lctx_desc, rank)
            rank_config = spu_pb2.RuntimeConfig()
            rank_config.CopyFrom(self.rt_config)
            if rank != 0:
                # rank_config.enable_pphlo_trace = False
                rank_config.enable_action_trace = False
                rank_config.enable_hal_profile = False
                rank_config.enable_pphlo_profile = False
            rt = ppapi.Runtime(lctx, rank_config)

            # mock input, output names.
            input_names = [f'in{idx}' for idx in range(len(params))]
            output_names = [f'out{idx}' for idx in range(num_returns)]

            # make an spu executable.
            executable = spu_pb2.ExecutableProto(
                name='test',
                input_names=input_names,
                output_names=output_names,
                code=text.encode('utf-8'),
            )

            # do infeed.
            for idx, param in enumerate(params):
                rt.set_var(input_names[idx], param[rank])

            # run
            rt.run(executable)

            # do outfeed
            return [rt.get_var(name) for name in output_names]

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
    inline: bool = False,
):
    """
    Decorates a jax numpy fn that simulated on SPU.

        sim = Simulator.simple(3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64)
        spu_fn = sim_jax(sim, jnp.add)

    Then we can call spu_fn like normal jnp fn.

        x = np.array([[1, 2], [3, 4]])
        y = np.array([[5, 6], [7, 8]])
        z = spu_fn(x, y)

    The function will be evaluated in an spu simulator.
    """

    def wrapper(*args, **kwargs):
        xla, out_shape = jax.xla_computation(
            fun, backend="interpreter", return_shape=True, static_argnums=static_argnums
        )(*args, **kwargs)

        # copy from jax.xla_computation to make args aligned.
        f = lu.wrap_init(fun)
        f, dyn_args = japi_util.argnums_partial_except(
            f, static_argnums, args, allow_invalid=False
        )
        args_flat, _ = jax.tree_util.tree_flatten((dyn_args, kwargs))
        # end copy from jax.xla_computation

        # compile xla to pphlo
        xla_ir = spu_pb2.IrProto(
            ir_type=spu_pb2.IrType.IR_XLA_HLO,
            code=xla.as_serialized_hlo_module_proto(),
            meta=spu_pb2.XlaMeta(
                inputs=[spu_pb2.Visibility.VIS_SECRET] * len(args_flat)
            ),
        )
        pphlo_ir = ppapi.compile(xla_ir)

        wrapper.xla = xla.as_hlo_text()
        wrapper.pphlo = pphlo_ir.code.decode("utf-8")

        # simulate it.
        _, out_tree = jax.tree_util.tree_flatten(out_shape)
        out_flat = sim(wrapper.pphlo, out_tree.num_leaves, *args_flat)

        return jax.tree_util.tree_unflatten(out_tree, out_flat)

    return wrapper
