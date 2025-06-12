# Copyright 2025 Ant Group Co., Ltd.
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
from typing import Callable

class ProgressData:
    def __init__(self): ...
    @property
    def total(self) -> int: ...
    @property
    def finished(self) -> int: ...
    @property
    def running(self) -> int: ...
    @property
    def percentage(self) -> int: ...
    @property
    def description(self) -> str: ...

class ProgressParams:
    def __init__(
        self,
        hook: Callable[[ProgressData], None] | None = None,
        interval_ms: int = 5 * 1000,
    ) -> None:
        self.hook = hook
        self.interval_ms = interval_ms

class PsiProtocol(enum.IntEnum):
    PROTOCOL_UNSPECIFIED = 0
    PROTOCOL_ECDH = 1
    PROTOCOL_KKRT = 2
    PROTOCOL_RR22 = 3
    PROTOCOL_ECDH_3PC = 4
    PROTOCOL_ECDH_NPC = 5
    PROTOCOL_KKRT_NPC = 6
    PROTOCOL_DP = 7

class EllipticCurveType(enum.IntEnum):
    CURVE_INVALID_TYPE = 0
    CURVE_25519 = 1
    CURVE_FOURQ = 2
    CURVE_SM2 = 3
    CURVE_SECP256K1 = 4
    CURVE_25519_ELLIGATOR2 = 5

class EcdhParams:
    def __init__(
        self,
        curve: EllipticCurveType = EllipticCurveType.CURVE_INVALID_TYPE,
        batch_size: int = 4096,
    ):
        self.curve = curve
        self.batch_size = batch_size

class Rr22Rarams:
    def __init__(self, low_comm_mode: bool = False):
        self.low_comm_mode = low_comm_mode

class DpParams:
    bob_sub_sampling: float = 0.9
    epsilon: float = 3.0

class PsiProtocolConfig:
    def __init__(
        self,
        protocol: PsiProtocol = PsiProtocol.PROTOCOL_UNSPECIFIED,
        receiver_rank: int = 0,
        bucket_size: int = 1 << 20,
        broadcast_result: bool = False,
        ecdh_params: EcdhParams = EcdhParams(),
        rr22_params: Rr22Rarams = Rr22Rarams(),
    ):
        self.protocol = protocol
        self.receiver_rank = receiver_rank
        self.bucket_size = bucket_size
        self.broadcast_result = broadcast_result
        self.ecdh_params = ecdh_params
        self.rr22_params = rr22_params

class SourceType(enum.IntEnum):
    SOURCE_TYPE_UNSPECIFIED = 0
    SOURCE_TYPE_FILE_CSV = 1

class InputParams:
    def __init__(
        self,
        type: SourceType = SourceType.SOURCE_TYPE_FILE_CSV,
        path: str = "",
        selected_keys: list[str] = [],
        keys_unique: bool = False,
    ):
        self.type = type
        self.path = path
        self.selected_keys = selected_keys
        self.keys_unique = keys_unique

class OutputParams:
    def __init__(
        self,
        type: SourceType = SourceType.SOURCE_TYPE_FILE_CSV,
        path: str = "",
        disable_alignment: bool = False,
        csv_null_rep: str = "NULL",
    ):
        self.type = type
        self.path = path
        self.disable_alignment = disable_alignment
        self.csv_null_rep = csv_null_rep

class CheckpointConfig:
    def __init__(self, enable: bool = False, path: str = ""):
        self.enable = enable
        self.path = path

class ResultJoinType(enum.IntEnum):
    JOIN_TYPE_UNSPECIFIED = 0
    JOIN_TYPE_INNER_JOIN = 1
    JOIN_TYPE_LEFT_JOIN = 2
    JOIN_TYPE_RIGHT_JOIN = 3
    JOIN_TYPE_FULL_JOIN = 4
    JOIN_TYPE_DIFFERENCE = 5

class ResultJoinConfig:
    def __init__(
        self,
        type: ResultJoinType = ResultJoinType.JOIN_TYPE_INNER_JOIN,
        left_side_rank: int = 0,
    ):
        self.type = type
        self.left_side_rank = left_side_rank

class PsiExecuteConfig:
    def __init__(
        self,
        protocol_conf: PsiProtocolConfig = PsiProtocolConfig(),
        input_params: InputParams = InputParams(),
        output_params: OutputParams = OutputParams(),
        join_conf: ResultJoinConfig = ResultJoinConfig(),
        checkpoint_conf: CheckpointConfig = CheckpointConfig(),
    ):
        self.protocol_conf = protocol_conf
        self.input_params = input_params
        self.output_params = output_params
        self.join_conf = join_conf
        self.checkpoint_conf = checkpoint_conf

class UbPsiMode(enum.IntEnum):
    MODE_UNSPECIFIED = 0
    MODE_OFFLINE_GEN_CACHE = 1
    MODE_OFFLINE_TRANSFER_CACHE = 2
    MODE_OFFLINE = 3
    MODE_ONLINE = 4
    MODE_FULL = 5

class UbPsiRole(enum.IntEnum):
    ROLE_UNSPECIFIED = 0
    ROLE_SERVER = 3
    ROLE_CLIENT = 4

class UbPsiServerParams:
    def __init__(self, secret_key_path: str = ""):
        self.secret_key_path = secret_key_path

class UbPsiExecuteConfig:
    def __init__(
        self,
        mode: UbPsiMode = UbPsiMode.MODE_UNSPECIFIED,
        role: UbPsiRole = UbPsiRole.ROLE_UNSPECIFIED,
        server_receive_result: bool = False,
        client_receive_result: bool = False,
        cache_path: str = "",
        input_params: InputParams = InputParams(),
        output_params: OutputParams = OutputParams(),
        server_params: UbPsiServerParams = UbPsiServerParams(),
        join_conf: ResultJoinConfig = ResultJoinConfig(),
    ):
        self.mode = mode
        self.role = role
        self.server_receive_result = server_receive_result
        self.client_receive_result = client_receive_result
        self.cache_path = cache_path
        self.input_params = input_params
        self.output_params = output_params
        self.server_params = server_params
        self.join_conf = join_conf

class PsiExecuteReport:
    def __init__(self): ...
    @property
    def original_count(self) -> int: ...
    @property
    def intersection_count(self) -> int: ...
    @property
    def original_unique_count(self) -> int: ...
    @property
    def intersection_unique_count(self) -> int: ...

def psi_execute(
    config: PsiExecuteConfig,
    lctx: link.Context,
    progress_params: ProgressParams = ProgressParams(),
) -> PsiExecuteReport: ...
def ub_psi_execute(
    config: UbPsiExecuteConfig, lctx: link.Context
) -> PsiExecuteReport: ...
