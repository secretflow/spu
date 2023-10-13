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

from . import libspu  # type: ignore
from .libspu.libs import ProgressData
from .psi_pb2 import (  # type: ignore
    BucketPsiConfig,
    CurveType,
    InputParams,
    MemoryPsiConfig,
    OutputParams,
    PsiResultReport,
    PsiType,
)


def mem_psi(
    link: libspu.link.Context, config: MemoryPsiConfig, input_items: List[str]
) -> List[str]:
    return libspu.libs.mem_psi(link, config.SerializeToString(), input_items)


def bucket_psi(
    link: libspu.link.Context,
    config: BucketPsiConfig,
    progress_callbacks: [[psi.ProgressData], None] = None,
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
    report_str = libspu.libs.bucket_psi(
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
    report_str = libspu.libs.bucket_psi(None, config.SerializeToString())
    report = PsiResultReport()
    report.ParseFromString(report_str)
    return report
