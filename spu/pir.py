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

from .pir_pb2 import (  # type: ignore
    KvStoreType,
    PirProtocol,
    PirResultReport,
    PirClientConfig,
    PirServerConfig,
    PirSetupConfig,
)

from . import libspu  # type: ignore


def pir_setup(config: PirSetupConfig) -> List[str]:
    report_str = libspu.libs.pir_setup(config.SerializeToString())

    report = PirResultReport()
    report.ParseFromString(report_str)
    return report


def pir_server(link: libspu.link.Context, config: PirServerConfig) -> List[str]:
    report_str = libspu.libs.pir_server(link, config.SerializeToString())

    report = PirResultReport()
    report.ParseFromString(report_str)
    return report


def pir_client(link: libspu.link.Context, config: PirClientConfig) -> List[str]:
    report_str = libspu.libs.pir_client(link, config.SerializeToString())

    report = PirResultReport()
    report.ParseFromString(report_str)
    return report
