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

    def run(self, executable: libspu.Executable) -> None:
        """Run an SPU executable.

        Args:
            executable (libspu.Executable): executable.

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

    def get_var_meta(self, name: str) -> libspu.ValueMeta:
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


def parse_data_type(v: str) -> libspu.DataType:
    match v:
        case "DT_INVALID":
            return libspu.DataType.DT_INVALID
        case "DT_I1":
            return libspu.DataType.DT_I1
        case "DT_I8":
            return libspu.DataType.DT_I8
        case "DT_U8":
            return libspu.DataType.DT_U8
        case "DT_I16":
            return libspu.DataType.DT_I16
        case "DT_U16":
            return libspu.DataType.DT_U16
        case "DT_I32":
            return libspu.DataType.DT_I32
        case "DT_U32":
            return libspu.DataType.DT_U32
        case "DT_I64":
            return libspu.DataType.DT_I64
        case "DT_U64":
            return libspu.DataType.DT_U64
        case "DT_F16":
            return libspu.DataType.DT_F16
        case "DT_F32":
            return libspu.DataType.DT_F32
        case "DT_F64":
            return libspu.DataType.DT_F64
        case _:
            raise ValueError(f"Invalid data type: {v}")


def parse_visibility(v: str) -> libspu.Visibility:
    match v:
        case "VIS_INVALID":
            return libspu.Visibility.VIS_INVALID
        case "VIS_SECRET":
            return libspu.Visibility.VIS_SECRET
        case "VIS_PUBLIC":
            return libspu.Visibility.VIS_PUBLIC
        case "VIS_PRIVATE":
            return libspu.Visibility.VIS_PRIVATE
        case _:
            raise ValueError(f"Invalid visibility: {v}")


def parse_field_type(v: str) -> libspu.FieldType:
    match v:
        case "FT_INVALID":
            return libspu.FieldType.FT_INVALID
        case "FM32":
            return libspu.FieldType.FM32
        case "FM64":
            return libspu.FieldType.FM64
        case "FM128":
            return libspu.FieldType.FM128
        case _:
            raise ValueError(f"Invalid field type: {v}")


def parse_protocol_kind(v: str) -> libspu.ProtocolKind:
    match v:
        case "PROT_INVALID":
            return libspu.ProtocolKind.PROT_INVALID
        case "REF2K":
            return libspu.ProtocolKind.REF2K
        case "SEMI2K":
            return libspu.ProtocolKind.SEMI2K
        case "ABY3":
            return libspu.ProtocolKind.ABY3
        case "CHEETAH":
            return libspu.ProtocolKind.CHEETAH
        case "SECURENN":
            return libspu.ProtocolKind.SECURENN
        case _:
            raise ValueError(f"Invalid protocol kind: {v}")


def parse_cheetah_ot_kind(v: str) -> libspu.CheetahOtKind:
    match v:
        case "YACL_Ferret":
            return libspu.CheetahOtKind.YACL_Ferret
        case "YACL_Softspoken":
            return libspu.CheetahOtKind.YACL_Softspoken
        case "EMP_Ferret":
            return libspu.CheetahOtKind.EMP_Ferret
        case _:
            raise ValueError(f"Invalid cheetah ot kind: {v}")


def parse_sort_method(v: str) -> libspu.RuntimeConfig.SortMethod:
    match v:
        case "":
            return libspu.RuntimeConfig.SortMethod.SORT_DEFAULT
        case "SORT_DEFAULT":
            return libspu.RuntimeConfig.SortMethod.SORT_DEFAULT
        case "SORT_RADIX":
            return libspu.RuntimeConfig.SortMethod.SORT_RADIX
        case "SORT_QUICK":
            return libspu.RuntimeConfig.SortMethod.SORT_QUICK
        case "SORT_NETWORK":
            return libspu.RuntimeConfig.SortMethod.SORT_NETWORK
        case _:
            raise ValueError(f"Invalid sort method: {v}")


def parse_exp_mode(v: str) -> libspu.RuntimeConfig.ExpMode:
    match v:
        case "EXP_DEFAULT":
            return libspu.RuntimeConfig.ExpMode.EXP_DEFAULT
        case "EXP_PADE":
            return libspu.RuntimeConfig.ExpMode.EXP_PADE
        case "EXP_TAYLOR":
            return libspu.RuntimeConfig.ExpMode.EXP_TAYLOR
        case "EXP_PRIME":
            return libspu.RuntimeConfig.ExpMode.EXP_PRIME
        case _:
            raise ValueError(f"Invalid exp mode: {v}")


def parse_log_mode(v: str) -> libspu.RuntimeConfig.LogMode:
    match v:
        case "LOG_DEFAULT":
            return libspu.RuntimeConfig.LogMode.LOG_DEFAULT
        case "LOG_PADE":
            return libspu.RuntimeConfig.LogMode.LOG_PADE
        case "LOG_NEWTON":
            return libspu.RuntimeConfig.LogMode.LOG_NEWTON
        case "LOG_MINMAX":
            return libspu.RuntimeConfig.LogMode.LOG_MINMAX
        case _:
            raise ValueError(f"Invalid log mode: {v}")


def parse_sigmoid_mode(v: str) -> libspu.RuntimeConfig.SigmoidMode:
    match v:
        case "SIGMOID_DEFAULT":
            return libspu.RuntimeConfig.SigmoidMode.SIGMOID_DEFAULT
        case "SIGMOID_MM1":
            return libspu.RuntimeConfig.SigmoidMode.SIGMOID_MM1
        case "SIGMOID_SEG3":
            return libspu.RuntimeConfig.SigmoidMode.SIGMOID_SEG3
        case "SIGMOID_REAL":
            return libspu.RuntimeConfig.SigmoidMode.SIGMOID_REAL
        case _:
            raise ValueError(f"Invalid sigmoid mode: {v}")


def parse_beaver_type(v: str) -> libspu.RuntimeConfig.BeaverType:
    match v:
        case "TRUSTED_FIRST_PARTY":
            return libspu.RuntimeConfig.BeaverType.TrustedFirstParty
        case "TrustedFirstParty":
            return libspu.RuntimeConfig.BeaverType.TrustedFirstParty
        case "TRUSTED_THIRD_PARTY":
            return libspu.RuntimeConfig.BeaverType.TrustedThirdParty
        case "TrustedThirdParty":
            return libspu.RuntimeConfig.BeaverType.TrustedThirdParty
        case "MULTI_PARTY":
            return libspu.RuntimeConfig.BeaverType.MultiParty
        case "MultiParty":
            return libspu.RuntimeConfig.BeaverType.MultiParty
        case _:
            raise ValueError(f"Invalid beaver type: {v}")
