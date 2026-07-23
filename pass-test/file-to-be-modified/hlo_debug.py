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
    sim = ppsim.Simulator.simple({partynum}, spu_pb2.ProtocolKind.{protocalchosen}, spu_pb2.FieldType.FM64)
    copts = spu_pb2.CompilerOptions()

    # copts.enable_pretty_print = True
    # copts.pretty_print_dump_dir = "/home1/leiyu.lyc/ppu/pass-test/tmp-exp/forest-while/unopt"

    HLO_IR = """{IR_to_be_tested}"""

    input_args = {input_args}
    result = ppsim.sim_hlo(sim, HLO_IR, {return_number}, input_args, copts=copts)