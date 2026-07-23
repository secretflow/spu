import jax.numpy as jnp
import numpy as np

import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as ppsim

if __name__ == "__main__":
    """
    You can modify the code below for debug purpose only.
    Please DONT commit it unless it will cause build break.
    """

    # sim = ppsim.Simulator.simple(2, spu_pb2.ProtocolKind.SEMI2K, spu_pb2.FieldType.FM64)
    # sim = ppsim.Simulator.simple(3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64)
    sim = ppsim.Simulator.simple(2, spu_pb2.ProtocolKind.SEMI2K, spu_pb2.FieldType.FM64)
    copts = spu_pb2.CompilerOptions()
    copts.disable_algebraicsimplifier = True


    HLO_IR = """HloModule jit_test, entry_computation_layout={(s32[4,4]{1,0}, s32[4,4,1]{2,1,0})->(s32[4,4]{0,1}, s32[4]{0})}, frontend_attributes={xla.sdy.meshes={}}

region_0.20 {
  Arg_0.21 = s32[] parameter(0)
  Arg_1.22 = s32[] parameter(1)
  ROOT minimum.23 = s32[] minimum(Arg_0.21, Arg_1.22)
}

ENTRY main.26 {
  Arg_0.1 = s32[4,4]{1,0} parameter(0)
  constant.17 = s32[] constant(2147483647)
  reshape.19 = s32[4,4,1]{2,1,0} parameter(1)
  transpose.18 = s32[4,4]{0,1} transpose(Arg_0.1), dimensions={1,0}
  reduce.24 = s32[4]{0} reduce(reshape.19, constant.17), dimensions={1,2}, to_apply=region_0.20
  ROOT tuple.25 = (s32[4,4]{0,1}, s32[4]{0}) tuple(transpose.18, reduce.24)
}"""

    input_args = [np.ones((4,4), dtype=np.int32), np.ones((4,4,1), dtype=np.int32)]
    result = ppsim.sim_hlo(sim, HLO_IR, 2, input_args, copts=copts)