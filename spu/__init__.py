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


from .version import __version__  # type: ignore

from .spu_pb2 import (  # type: ignore
    DataType,
    Visibility,
    PtType,
    ProtocolKind,
    FieldType,
    ShapeProto,
    RuntimeConfig,
    ExecutableProto,
    CompilerOptions,
)

from .api import Io, Runtime, compile, check_cpu_feature
from .utils import simulation
from .intrinsic import *

from . import pir
from . import psi

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
    "pir",
    "psi",
    # intrinsic
] + intrinsic.__all__

check_cpu_feature()
