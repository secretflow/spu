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

import os
from typing import List

import spu.spu_pb2 as spu_pb2

from cachetools import LRUCache, cached

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

    def clear(self) -> None:
        """Delete all SPU values."""
        self._vm.Clear()


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
        self, x: 'np.ndarray', vtype: spu_pb2.Visibility, owner_rank: int = -1
    ) -> List[spu_pb2.ValueProto]:
        """Convert from NumPy array to list of SPU value(s).

        Args:
            x (np.ndarray): input.
            vtype (Visibility): visibility.
            owner_rank (int): the index of the trusted piece. if >= 0, colocation optimization may be applied.

        Returns:
            [ValueProto]: output.
        """
        str_shares = self._io.MakeShares(x, vtype, owner_rank)
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


@cached(cache=LRUCache(maxsize=128))
def _spu_compilation(ir_text: str, ir_type: str, json_meta: str):
    pp_dir = os.getenv('SPU_IR_DUMP_DIR')
    return _lib.compile(ir_text, ir_type, json_meta, pp_dir or "")


def compile(ir_text: str, ir_type: str, vis: List[spu_pb2.Visibility]) -> str:
    """Compile from textual HLO/MHLO IR to SPU bytecode.

    Args:
        ir_text (str): textual HLO/MHLO IR protobuf binary format.
        ir_type (str): "hlo" or "mhlo".
        vtype (Visibility): Visbilities .

    Returns:
        [ValueProto]: output.
    """
    from google.protobuf.json_format import MessageToJson

    # todo: rename XlaMeta to IrMeta?
    return _spu_compilation(
        ir_text, ir_type, MessageToJson(spu_pb2.XlaMeta(inputs=vis))
    )
