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

from .libpsi import (  # type: ignore
    CheckpointConfig,
    DpParams,
    EcdhParams,
    EllipticCurveType,
    InputParams,
    OutputParams,
    ProgressData,
    ProgressParams,
    PsiExecuteConfig,
    PsiExecuteReport,
    PsiProtocol,
    PsiProtocolConfig,
    ResultJoinConfig,
    ResultJoinType,
    Rr22Rarams,
    SourceType,
    UbPsiExecuteConfig,
    UbPsiMode,
    UbPsiRole,
    UbPsiServerParams,
    psi_execute,
    ub_psi_execute,
)


def parse_protocol(v: str) -> PsiProtocol:
    match v:
        case "PROTOCOL_ECDH":
            return PsiProtocol.PROTOCOL_ECDH
        case "PROTOCOL_KKRT":
            return PsiProtocol.PROTOCOL_KKRT
        case "PROTOCOL_RR22":
            return PsiProtocol.PROTOCOL_RR22
        case "PROTOCOL_ECDH_3PC":
            return PsiProtocol.PROTOCOL_ECDH_3PC
        case "PROTOCOL_ECDH_NPC":
            return PsiProtocol.PROTOCOL_ECDH_NPC
        case "PROTOCOL_KKRT_NPC":
            return PsiProtocol.PROTOCOL_KKRT_NPC
        case "PROTOCOL_DP":
            return PsiProtocol.PROTOCOL_DP
        case _:
            return PsiProtocol.PROTOCOL_UNSPECIFIED


def parse_curve_type(v: str) -> EllipticCurveType:
    match v:
        case "CURVE_25519":
            return EllipticCurveType.CURVE_25519
        case "CURVE_FOURQ":
            return EllipticCurveType.CURVE_FOURQ
        case "CURVE_SM2":
            return EllipticCurveType.CURVE_SM2
        case "CURVE_SECP256K1":
            return EllipticCurveType.CURVE_SECP256K1
        case "CURVE_25519_ELLIGATOR2":
            return EllipticCurveType.CURVE_25519_ELLIGATOR2
        case _:
            return EllipticCurveType.CURVE_INVALID_TYPE


def parse_source_type(v: str) -> SourceType:
    match v:
        case "SOURCE_TYPE_FILE_CSV":
            return SourceType.SOURCE_TYPE_FILE_CSV
        case _:
            return SourceType.SOURCE_TYPE_UNSPECIFIED


def parse_join_type(v: str) -> ResultJoinType:
    match v:
        case "JOIN_TYPE_INNER_JOIN":
            return ResultJoinType.JOIN_TYPE_INNER_JOIN
        case "JOIN_TYPE_LEFT_JOIN":
            return ResultJoinType.JOIN_TYPE_LEFT_JOIN
        case "JOIN_TYPE_RIGHT_JOIN":
            return ResultJoinType.JOIN_TYPE_RIGHT_JOIN
        case "JOIN_TYPE_FULL_JOIN":
            return ResultJoinType.JOIN_TYPE_FULL_JOIN
        case "JOIN_TYPE_DIFFERENCE":
            return ResultJoinType.JOIN_TYPE_DIFFERENCE
        case _:
            return ResultJoinType.JOIN_TYPE_UNSPECIFIED


def parse_ub_psi_mode(mode: str) -> UbPsiMode:
    match mode:
        case "MODE_OFFLINE_GEN_CACHE":
            return UbPsiMode.MODE_OFFLINE_GEN_CACHE
        case "MODE_OFFLINE_TRANSFER_CACHE":
            return UbPsiMode.MODE_OFFLINE_TRANSFER_CACHE
        case "MODE_OFFLINE":
            return UbPsiMode.MODE_OFFLINE
        case "MODE_ONLINE":
            return UbPsiMode.MODE_ONLINE
        case "MODE_FULL":
            return UbPsiMode.MODE_FULL
        case _:
            return UbPsiMode.MODE_UNSPECIFIED


def parse_ub_psi_role(role: str) -> UbPsiRole:
    match role:
        case "ROLE_SERVER":
            return UbPsiRole.ROLE_SERVER
        case "ROLE_CLIENT":
            return UbPsiRole.ROLE_CLIENT
        case _:
            return UbPsiRole.ROLE_UNSPECIFIED
