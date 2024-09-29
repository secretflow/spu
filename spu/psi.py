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

from . import libpsi  # type: ignore
from .libpsi.libs import ProgressData
from .libspu.link import Context  # type: ignore
from .pir_pb2 import (  # type: ignore
    ApsiReceiverConfig,
    ApsiSenderConfig,
    PirResultReport,
)
from .psi_pb2 import (  # type: ignore
    BucketPsiConfig,
    CurveType,
    InputParams,
    MemoryPsiConfig,
    OutputParams,
    PsiResultReport,
    PsiType,
)
from .psi_v2_pb2 import (
    DebugOptions,
    EcdhConfig,
    IoConfig,
    IoType,
    KkrtConfig,
    Protocol,
    ProtocolConfig,
    PsiConfig,
    RecoveryConfig,
    Role,
    Rr22Config,
    UbPsiConfig,
)


def mem_psi(
    link: Context, config: MemoryPsiConfig, input_items: List[str]
) -> List[str]:
    return libpsi.libs.mem_psi(link, config.SerializeToString(), input_items)


def bucket_psi(
    link: Context,
    config: BucketPsiConfig,
    progress_callbacks: [[libpsi.libs.ProgressData], None] = None,
    callbacks_interval_ms: int = 5 * 1000,
    ic_mode: bool = False,
) -> PsiResultReport:
    """
    Run bucket psi
    :param link: the transport layer
    :param config: psi config
    :param progress_callbacks: callbacks func for psi progress. func defined: def progress_callbacks(data: ProgressData)
    :param ic_mode: Whether to run in interconnection mode
    :return: statistical results
    """
    report_str = libpsi.libs.bucket_psi(
        link,
        config.SerializeToString(),
        progress_callbacks,
        callbacks_interval_ms,
        ic_mode,
    )
    report = PsiResultReport()
    report.ParseFromString(report_str)
    return report


def gen_cache_for_2pc_ub_psi(config: BucketPsiConfig) -> PsiResultReport:
    """This API is still experimental.

    Args:
        config (BucketPsiConfig): only the following fields are required:
            - input_params
            - output_params
            - bucket_size
            - curve_type
            - ecdh_secret_key_path

        Other fields would be overwritten.

    Returns:
        PsiResultReport: statistical results
    """
    config.psi_type = PsiType.Value('ECDH_OPRF_UB_PSI_2PC_GEN_CACHE')
    config.broadcast_result = False
    config.output_params.need_sort = False
    config.receiver_rank = 0
    report_str = libpsi.libs.bucket_psi(None, config.SerializeToString())
    report = PsiResultReport()
    report.ParseFromString(report_str)
    return report


def psi(
    config: PsiConfig,
    link: Context = None,
) -> PsiResultReport:
    """
    Run PSI with v2 API.
    Check PsiConfig at https://www.secretflow.org.cn/docs/psi/latest/en-US/reference/psi_v2_config#psiconfig.
    :param config: psi config
    :param link: the transport layer
    :return: statistical results
    """
    report_str = libpsi.libs.psi(
        config.SerializeToString(),
        link,
    )
    report = PsiResultReport()
    report.ParseFromString(report_str)
    return report


def ub_psi(
    config: UbPsiConfig,
    link: Context = None,
) -> PsiResultReport:
    """
    Run PSI with v2 API.
    Check UbPsiConfig at https://www.secretflow.org.cn/docs/psi/latest/en-US/reference/psi_v2_config#ubpsiconfig.
    :param config: ub psi config
    :param link: the transport layer
    :return: statistical results
    """
    report_str = libpsi.libs.ub_psi(
        config.SerializeToString(),
        link,
    )
    report = PsiResultReport()
    report.ParseFromString(report_str)
    return report


def apsi_send(config: ApsiSenderConfig, link: Context = None) -> PirResultReport:
    report_str = libpsi.libs.apsi_send(config.SerializeToString(), link)

    report = PirResultReport()
    report.ParseFromString(report_str)
    return report


def apsi_receive(config: ApsiReceiverConfig, link: Context = None) -> PirResultReport:
    report_str = libpsi.libs.apsi_receive(config.SerializeToString(), link)

    report = PirResultReport()
    report.ParseFromString(report_str)
    return report
