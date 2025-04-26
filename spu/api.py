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


class Runtime(object):
    """The SPU Virtual Machine Slice."""

    def __init__(self, link: libspu.link.Context, config: libspu.RuntimeConfig):
        """Constructor of an SPU Runtime.

        Args:
            link (libspu.link.Context): Link context.
            config (libspu.RuntimeConfig): SPU Runtime Config.
        """
        self._vm = libspu.RuntimeWrapper(link, config)

    def run(self, executable: libspu.ExecutableProto) -> None:
        """Run an SPU executable.

        Args:
            executable (libspu.ExecutableProto): executable.

        """
        return self._vm.Run(executable)

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

    def get_var_meta(self, name: str) -> libspu.ValueMetaProto:
        """Get an SPU value without content.

        Args:
            name (str): Id of value.

        Returns:
            libspu.ValueMeta: Data meta with out content.
        """
        return self._vm.GetVarMeta(name)

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

    def __init__(self, world_size: int, config: libspu.RuntimeConfig):
        """Constructor of an SPU Io.

        Args:
            world_size (int): # of participants of SPU Device.
            config (libspu.RuntimeConfig): SPU Runtime Config.
        """
        self._io = libspu.IoWrapper(world_size, config)

    def get_share_chunk_count(
        self, x: 'np.ndarray', vtype: libspu.Visibility, owner_rank: int = -1
    ) -> int:
        return self._io.GetShareChunkCount(x, vtype, owner_rank)

    def make_shares(
        self, x: 'np.ndarray', vtype: libspu.Visibility, owner_rank: int = -1
    ) -> List[libspu.Share]:
        """Convert from NumPy array to list of SPU value(s).

        Args:
            x (np.ndarray): input.
            vtype (libspu.Visibility): visibility.
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
def _spu_compilation(
    source: libspu.CompilationSource, options: libspu.CompilerOptions
) -> bytes:
    return libspu.compile(source, options)


def compile(source: libspu.CompilationSource, copts: libspu.CompilerOptions) -> bytes:
    """Compile from textual HLO/MHLO IR to SPU bytecode.

    Args:
        source (libspu.CompilationSource): input to compiler.
        copts (libspu.CompilerOptions): compiler options.

    Returns:
        [libspu.ValueProto]: output.
    """

    return _spu_compilation(source, copts)


def check_cpu_feature():
    """Check CPU features required by SPU."""
    libspu._check_cpu_features()
