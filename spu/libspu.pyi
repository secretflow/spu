# Copyright 2024 Ant Group Co., Ltd.
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

import enum
from dataclasses import dataclass
from typing import overload

class logging:
    '''
    Simulate the logging module using class
    '''

    class LogLevel(enum.IntEnum):
        DEBUG = 0
        INFO = 1
        WARN = 2
        ERROR = 3

    @dataclass
    class LogOptions:
        enable_console_logger: bool = True
        system_log_path: str = ""
        trace_log_path: str = ""
        log_level: logging.LogLevel = logging.LogLevel.INFO
        max_log_file_size: int = 500 * 1024 * 1024
        max_log_file_count: int = 10
        trace_content_length: int = 100

    @staticmethod
    def setup_logging(options: LogOptions) -> None: ...

class link:
    '''
    Simulate the link module using class
    '''

    @dataclass
    class CertInfo:
        certificate_path: str
        private_key_path: str

    @dataclass
    class VerifyOptions:
        verify_depth: str
        ca_file_path: str

    @dataclass
    class RetryOptions:
        max_retry: int
        retry_interval_ms: int
        retry_interval_incr_ms: int
        max_retry_interval_ms: int
        error_codes: set[int]
        http_codes: set[int]
        aggressive_retry: bool

    @dataclass
    class SSLOptions:
        cert: link.CertInfo
        verify: link.VerifyOptions

    @dataclass
    class ContextDesc:
        class Party:
            @property
            def id(self) -> str: ...
            @property
            def host(self) -> str: ...

        id: str
        connect_retry_times: int
        connect_retry_interval_ms: int
        recv_timeout_ms: int
        http_max_payload_size: int
        http_timeout_ms: int
        brpc_channel_protocol: str
        brpc_channel_connection_type: str
        throttle_window_size: int
        enable_ssl: bool
        client_ssl_opts: link.SSLOptions
        server_ssl_opts: link.SSLOptions
        link_type: str
        retry_opts: link.RetryOptions
        disable_msg_seq_id: bool
        @property
        def parties(self) -> list[Party]: ...
        def add_party(self, id: str, host: str) -> None: ...

    class Context:
        @property
        def rank(self) -> int: ...
        @property
        def world_size(self) -> int: ...
        def id(self) -> str: ...
        def spawn(self): ...
        def barrier(self): ...
        def all_gather(self, data: str) -> list[str]: ...
        def gather(self, data: str, root: int) -> list[str]: ...
        def broadcast(self, data: str, root: int) -> str: ...
        def stop_link(self) -> None: ...
        def scatter(self, data: list[str], root: int) -> str: ...
        def send(self, dst_rank: int, data: str) -> None: ...
        def send_async(self, dst_rank: int, data: str) -> None: ...
        def recv(self, src_rank: int) -> bytes: ...
        def next_rank(self, strides: int = 1) -> int: ...

    @staticmethod
    def create_brpc(
        desc: ContextDesc, self_rank: int, *, log_details: bool = False
    ) -> Context: ...
    @staticmethod
    def create_mem(desc: ContextDesc, self_rank: int) -> Context: ...

class DataType(enum.IntEnum):
    DT_INVALID = 0
    DT_I1 = 1
    DT_I8 = 2
    DT_U8 = 3
    DT_I16 = 4
    DT_U16 = 5
    DT_I32 = 6
    DT_U32 = 7
    DT_I64 = 8
    DT_U64 = 9
    DT_F16 = 10
    DT_F32 = 11
    DT_F64 = 12

class Visibility(enum.IntEnum):
    VIS_INVALID = 0
    VIS_SECRET = 1
    VIS_PUBLIC = 2
    VIS_PRIVATE = 3

class FieldType(enum.IntEnum):
    FT_INVALID = 0
    FM32 = 1
    FM64 = 2
    FM128 = 3

class ProtocolKind(enum.IntEnum):
    PROT_INVALID = 0
    REF2K = 1
    SEMI2K = 2
    ABY3 = 3
    CHEETAH = 4
    SECURENN = 5
    # SWIFT = 6

class ClientSSLConfig:
    def __init__(
        self,
        certificate: str = "",
        private_key: str = "",
        ca_file_path: str = "",
        verify_depth: str = "",
    ):
        self.certificate = certificate
        self.private_key = private_key
        self.ca_file_path = ca_file_path
        self.verify_depth = verify_depth

class TTPBeaverConfig:
    def __init__(
        self,
        server_host: str = "",
        adjust_rank: int = 0,
        asym_crypto_schema: str = "",
        server_public_key: str = "",
        transport_protocol: str = "",
        ssl_config: ClientSSLConfig | None = None,
    ):
        self.server_host = server_host
        self.adjust_rank = adjust_rank
        self.asym_crypto_schema = asym_crypto_schema
        self.server_public_key = server_public_key
        self.transport_protocol = transport_protocol
        self.ssl_config = ssl_config

class CheetahOtKind(enum.IntEnum):
    YACL_Ferret = 0
    YACL_Softspoken = 1
    EMP_Ferret = 2

class CheetahConfig:
    def __init__(
        self,
        disable_matmul_pack: bool,
        enable_mul_lsb_error: bool,
        ot_kind: CheetahOtKind = CheetahOtKind.YACL_Ferret,
    ):
        self.disable_matmul_pack = disable_matmul_pack
        self.enable_mul_lsb_error = enable_mul_lsb_error
        self.ot_kind = ot_kind

class RuntimeConfig:
    class SortMethod(enum.IntEnum):
        SORT_DEFAULT = 0
        SORT_RADIX = 1
        SORT_QUICK = 2
        SORT_NETWORK = 3

    class ExpMode(enum.IntEnum):
        EXP_DEFAULT = 0
        EXP_PADE = 1
        EXP_TAYLOR = 2
        EXP_PRIME = 3

    class LogMode(enum.IntEnum):
        LOG_DEFAULT = 0
        LOG_PADE = 1
        LOG_NEWTON = 2
        LOG_MINMAX = 3

    class SigmoidMode(enum.IntEnum):
        SIGMOID_DEFAULT = 0
        SIGMOID_MM1 = 1
        SIGMOID_SEG3 = 2
        SIGMOID_REAL = 3

    class BeaverType(enum.IntEnum):
        TrustedFirstParty = 0
        TrustedThirdParty = 1
        MultiParty = 2

    protocol: ProtocolKind
    field: FieldType
    fxp_fraction_bits: int
    max_concurrency: int
    enable_action_trace: bool
    enable_type_checker: bool
    enable_pphlo_trace: bool
    enable_runtime_snapshot: bool
    snapshot_dump_dir: str
    enable_pphlo_profile: bool
    enable_hal_profile: bool
    public_random_seed: int
    share_max_chunk_size: int
    sort_method: SortMethod
    quick_sort_threshold: int
    fxp_div_goldschmidt_iters: int
    fxp_exp_mode: ExpMode
    fxp_exp_iters: int
    fxp_log_mode: LogMode
    fxp_log_iters: int
    fxp_log_orders: int
    sigmoid_mode: SigmoidMode
    enable_lower_accuracy_rsqrt: bool
    sine_cosine_iters: int
    beaver_type: BeaverType
    ttp_beaver_config: TTPBeaverConfig
    cheetah_2pc_config: CheetahConfig
    trunc_allow_msb_error: bool
    experimental_disable_mmul_split: bool
    experimental_enable_inter_op_par: bool
    experimental_enable_intra_op_par: bool
    experimental_disable_vectorization: bool
    experimental_inter_op_concurrency: int
    experimental_enable_colocated_optimization: bool
    experimental_enable_exp_prime: bool
    experimental_exp_prime_offset: int
    experimental_exp_prime_disable_lower_bound: bool
    experimental_exp_prime_enable_upper_bound: bool

    # @staticmethod
    # def makeFromJson(json: str) -> 'RuntimeConfig': ...
    @overload
    def __init__(self): ...
    @overload
    def __init__(
        self,
        protocol: ProtocolKind = ProtocolKind.PROT_INVALID,
        field: FieldType = FieldType.FT_INVALID,
        fxp_fraction_bits: int = 0,
    ):
        self.protocol = protocol
        self.field = field
        self.fxp_fraction_bits = fxp_fraction_bits

    @overload
    def __init__(self, other: 'RuntimeConfig'): ...
    def ParseFromJsonString(self, data: str) -> bool: ...
    def ParseFromString(self, data: bytes) -> bool: ...
    def SerializeToString(self) -> bytes: ...
    def __str__(self) -> str: ...

class SourceIRType(enum.IntEnum):
    XLA = 0
    STABLEHLO = 1

class CompilationSource:
    def __init__(
        self,
        ir_type: SourceIRType = SourceIRType.XLA,
        ir_txt: bytes = b"",
        input_visibility: list[Visibility] = [],
    ):
        self.ir_type = ir_type
        self.ir_txt = ir_txt
        self.input_visibility = input_visibility

class XLAPrettyPrintKind(enum.IntEnum):
    TEXT = 0
    DOT = 1
    HTML = 2

class CompilerOptions:
    def __init__(
        self,
        enable_pretty_print: bool = False,
        pretty_print_dump_dir: str = "",
        xla_pp_kind: XLAPrettyPrintKind = XLAPrettyPrintKind.TEXT,
        disable_sqrt_plus_epsilon_rewrite=False,
        disable_div_sqrt_rewrite=False,
        disable_reduce_truncation_optimization=False,
        disable_maxpooling_optimization=False,
        disallow_mix_types_opts=False,
        disable_select_optimization=False,
        enable_optimize_denominator_with_broadcast=False,
        disable_deallocation_insertion=False,
        disable_partial_sort_optimization=False,
    ):
        self.enable_pretty_print = enable_pretty_print
        self.pretty_print_dump_dir = pretty_print_dump_dir
        self.xla_pp_kind = xla_pp_kind
        self.disable_sqrt_plus_epsilon_rewrite = disable_sqrt_plus_epsilon_rewrite
        self.disable_div_sqrt_rewrite = disable_div_sqrt_rewrite
        self.disable_reduce_truncation_optimization = (
            disable_reduce_truncation_optimization
        )
        self.disable_maxpooling_optimization = disable_maxpooling_optimization
        self.disallow_mix_types_opts = disallow_mix_types_opts
        self.disable_select_optimization = disable_select_optimization
        self.enable_optimize_denominator_with_broadcast = (
            enable_optimize_denominator_with_broadcast
        )
        self.disable_deallocation_insertion = disable_deallocation_insertion
        self.disable_partial_sort_optimization = disable_partial_sort_optimization

class Executable:
    def __init__(
        self,
        name: str = "",
        input_names: list[str] = [],
        output_names: list[str] = [],
        code: str | bytes = b"",
    ):
        self.name = name
        self.input_names = input_names
        self.output_names = output_names
        self.code = code

    def ParseFromString(self, data: bytes) -> bool: ...
    def SerializeToString(self) -> bytes: ...

class Share:
    meta: bytes
    share_chunks: list[bytes]

class Shape:
    dims: list[int]

@dataclass
class ValueMeta:
    data_type: DataType
    is_complex: bool
    visibility: Visibility
    shape: Shape
    storage_type: str
    def ParseFromString(self, data: bytes) -> bool: ...
    def SerializeToString(self) -> bytes: ...

@dataclass
class ValueChunk:
    total_bytes: int
    chunk_offset: int
    content: bytes
    def ParseFromString(self, data: bytes) -> bool: ...
    def SerializeToString(self) -> bytes: ...

class RuntimeWrapper:
    def __init__(self, link: link.Context, config: RuntimeConfig): ...
    def Run(self, executable: Executable): ...
    def SetVar(self, name: str, value: Share): ...
    def GetVar(self, name: str) -> Share: ...
    def GetVarChunksCount(self, name: str) -> int: ...
    def GetVarMeta(self, name: str) -> ValueMeta: ...
    def DelVar(self, name: str): ...
    def Clear(self): ...

class IoWrapper:
    def __init__(self, world_size: int, config: RuntimeConfig): ...
    def MakeShares(
        self, arr: bytes, visibility: int, owner_rank: int = -1
    ) -> list[Share]: ...
    def GetShareChunkCount(
        self, arr: bytes, visibility: int, owner_rank: int = -1
    ) -> int: ...
    def Reconstruct(self, vals: list[Share]) -> bytes: ...

def _check_cpu_features(): ...
def compile(source: CompilationSource, copts: CompilerOptions) -> bytes: ...
