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

from . import _lib

from spu.psi.psi_pb2 import (
    PsiType,
    CurveType,
    InputParams,
    OuputParams,
    PsiResultReport,
    BucketPsiConfig,
    MemoryPsiConfig,
)


def mem_psi(
    link: _lib.link.Context, config: MemoryPsiConfig, input_items: List[str]
) -> List[str]:
    return _lib.libs.mem_psi(link, config.SerializeToString(), input_items)


def bucket_psi(
    link: _lib.link.Context, config: BucketPsiConfig, ic_mode: bool = False
) -> PsiResultReport:
    """
    Run bucket psi
    :param link: the transport layer
    :param config: psi config
    :param ic_mode: Whether to run in interconnection mode
    :return: statistical results
    """
    report_str = _lib.libs.bucket_psi(link, config.SerializeToString(), ic_mode)
    report = PsiResultReport()
    report.ParseFromString(report_str)
    return report
