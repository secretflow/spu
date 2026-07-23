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


from . import experimental, psi
from .api import Io, Runtime, check_cpu_feature, compile
from .intrinsic import *
from .spu_pb2 import (  # type: ignore
    CompilerOptions,
    DataType,
    ExecutableProto,
    FieldType,
    ProtocolKind,
    PtType,
    RuntimeConfig,
    ShapeProto,
    Visibility,
)
from .utils import simulation
from .version import __version__  # type: ignor

__all__ = [
    "__version__",
    # spu_pb2
    "DataType",
    "Visibility",
    "PtType",
    "ProtocolKind",
    "FieldType",
    "ShapeProto",
    "RuntimeConfig",
    "ExecutableProto",
    "CompilerOptions",
    # spu_api
    "Io",
    "Runtime",
    "compile",
    # utils
    "simulation",
    # libs
    "psi",
    # intrinsic
] + intrinsic.__all__

check_cpu_feature()
