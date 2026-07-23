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

from cachetools import LRUCache, cached

from . import libspu  # type: ignore
from . import spu_pb2


class Runtime(object):
    """The SPU Virtual Machine Slice."""

    def __init__(self, link: libspu.link.Context, config: spu_pb2.RuntimeConfig):
        """Constructor of an SPU Runtime.

        Args:
            link (libspu.link.Context): Link context.
            config (spu_pb2.RuntimeConfig): SPU Runtime Config.
        """
        self._vm = libspu.RuntimeWrapper(link, config.SerializeToString())

    def run(self, executable: spu_pb2.ExecutableProto) -> None:
        """Run an SPU executable.

        Args:
            executable (spu_pb2.ExecutableProto): executable.

        """
        return self._vm.Run(executable.SerializeToString())

    def set_var(self, name: str, value: libspu.Share) -> None:
        """Set an SPU value.

        Args:
            name (str): Id of value.
            value (libspu.Share): value data.

        """
        return self._vm.SetVar(name, value)

    def get_var(self, name: str) -> libspu.Share:
        """Get an SPU value.

        Args:
            name (str): Id of value.

        Returns:
            libspu.Share: Data data.
        """
        return self._vm.GetVar(name)

    def get_var_chunk_count(self, name: str) -> int:
        """Get an SPU value.

        Args:
            name (str): Id of value.

        Returns:
            int: chunks count in libspu.Share
        """
        return self._vm.GetVarChunksCount(name)

    def get_var_meta(self, name: str) -> spu_pb2.ValueMetaProto:
        """Get an SPU value without content.

        Args:
            name (str): Id of value.

        Returns:
            spu_pb2.ValueMeta: Data meta with out content.
        """
        ret = spu_pb2.ValueMetaProto()
        ret.ParseFromString(self._vm.GetVarMeta(name))
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
            config (spu_pb2.RuntimeConfig): SPU Runtime Config.
        """
        self._io = libspu.IoWrapper(world_size, config.SerializeToString())

    def get_share_chunk_count(
        self, x: 'np.ndarray', vtype: spu_pb2.Visibility, owner_rank: int = -1
    ) -> int:
        return self._io.GetShareChunkCount(x, vtype, owner_rank)

    def make_shares(
        self, x: 'np.ndarray', vtype: spu_pb2.Visibility, owner_rank: int = -1
    ) -> List[libspu.Share]:
        """Convert from NumPy array to list of SPU value(s).

        Args:
            x (np.ndarray): input.
            vtype (spu_pb2.Visibility): visibility.
            owner_rank (int): the index of the trusted piece. if >= 0, colocation optimization may be applied.

        Returns:
            [libspu.Share]: output.
        """
        return self._io.MakeShares(x, vtype, owner_rank)

    def reconstruct(self, shares: List[libspu.Share]) -> 'np.ndarray':
        """Convert from list of SPU value(s) to NumPy array.

        Args:
            xs ([libspu.Share]): input.

        Returns:
            np.ndarray: output.
        """
        return self._io.Reconstruct(shares)


@cached(cache=LRUCache(maxsize=128))
def _spu_compilation(source: str, options_str: str):
    return libspu.compile(source, options_str)


def compile(source: spu_pb2.CompilationSource, copts: spu_pb2.CompilerOptions) -> str:
    """Compile from textual HLO/MHLO IR to SPU bytecode.

    Args:
        source (spu_pb2.CompilationSource): input to compiler.
        copts (spu_pb2.CompilerOptions): compiler options.

    Returns:
        [spu_pb2.ValueProto]: output.
    """

    return _spu_compilation(source.SerializeToString(), copts.SerializeToString())


def check_cpu_feature():
    """Check CPU features required by SPU."""
    libspu._check_cpu_features()
