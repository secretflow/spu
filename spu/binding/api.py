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

from __future__ import annotations

from typing import List

import spu.spu_pb2 as spu_pb2

from . import _lib


class Runtime(object):
    """The SPU Virtual Machine Slice."""

    def __init__(self, link: _lib.link.Context, config: spu_pb2.RuntimeConfig):
        """Constructor of an SPU Runtime.

        Args:
            link (_lib.link.Context): Link context.
            config (RuntimeConfig): SPU Runtime Config.
        """
        self._vm = _lib.RuntimeWrapper(link, config.SerializeToString())

    def run(self, executable: spu_pb2.ExecutableProto) -> None:
        """Run an SPU executable.

        Args:
            executable (ExecutableProto): executable.

        """
        return self._vm.Run(executable.SerializeToString())

    def set_var(self, name: str, value: spu_pb2.ValueProto) -> None:
        """Set an SPU value.

        Args:
            name (str): Id of value.
            value (ValueProto): value data.

        """
        return self._vm.SetVar(name, value.SerializeToString())

    def get_var(self, name: str) -> spu_pb2.ValueProto:
        """Get an SPU value.

        Args:
            name (str): Id of value.

        Returns:
            ValueProto: Data data.
        """
        ret = spu_pb2.ValueProto()
        ret.ParseFromString(self._vm.GetVar(name))
        return ret

    def del_var(self, name: str) -> None:
        """Delete an SPU value.

        Args:
            name (str): Id of the value.
        """
        self._vm.DelVar(name)


class Io(object):
    """The SPU IO interface."""

    def __init__(self, world_size: int, config: spu_pb2.RuntimeConfig):
        """Constructor of an SPU Io.

        Args:
            world_size (int): # of participants of SPU Device.
            config (RuntimeConfig): SPU Runtime Config.
        """
        self._io = _lib.IoWrapper(world_size, config.SerializeToString())

    def make_shares(
        self, x: 'np.ndarray', vtype: spu_pb2.Visibility
    ) -> List[spu_pb2.ValueProto]:
        """Convert from NumPy array to list of SPU value(s).

        Args:
            x (np.ndarray): input.
            vtype (Visibility): visibility.

        Returns:
            [ValueProto]: output.
        """
        str_shares = self._io.MakeShares(x, vtype)
        rets = []
        for str_share in str_shares:
            value_share = spu_pb2.ValueProto()
            value_share.ParseFromString(str_share)
            rets.append(value_share)
        return rets

    def reconstruct(self, xs: List[spu_pb2.ValueProto]) -> 'np.ndarray':
        """Convert from list of SPU value(s) to NumPy array.

        Args:
            xs (ValueProto]): input.

        Returns:
            np.ndarray: output.
        """
        str_shares = [x.SerializeToString() for x in xs]
        return self._io.Reconstruct(str_shares)


def compile(src: spu_pb2.IrProto) -> spu_pb2.IrProto:
    """Compile from XLA HLO to SPU HLO."""
    from google.protobuf.json_format import MessageToJson

    if src.ir_type == spu_pb2.IrType.IR_XLA_HLO:
        mlir = _lib.compile(src.code, MessageToJson(src.meta), "")
        mlir_proto = spu_pb2.IrProto()
        mlir_proto.code = mlir
        mlir_proto.ir_type = spu_pb2.IrType.IR_MLIR_SPU
        return mlir_proto
    else:
        raise NameError("Unknown ir type")
