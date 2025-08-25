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
from .api import (
    Io,
    Runtime,
    check_cpu_feature,
    compile,
    parse_beaver_type,
    parse_cheetah_ot_kind,
    parse_data_type,
    parse_exp_mode,
    parse_field_type,
    parse_log_mode,
    parse_protocol_kind,
    parse_sigmoid_mode,
    parse_sort_method,
    parse_visibility,
)
from .intrinsic import *
from .libspu import (  # type: ignore
    CompilationSource,
    CompilerOptions,
    DataType,
    Executable,
    FieldType,
    ProtocolKind,
    RuntimeConfig,
    Shape,
    SourceIRType,
    ValueChunk,
    ValueMeta,
    Visibility,
    link,
)
from .utils import simulation
from .version import __version__  # type: ignor

__all__ = [
    "__version__",
    # libspu
    "CompilationSource",
    "CompilerOptions",
    "DataType",
    "Visibility",
    "ProtocolKind",
    "FieldType",
    "Shape",
    "SourceIRType",
    "ValueChunk",
    "ValueMeta",
    "RuntimeConfig",
    "Executable",
    "link",
    # spu_api
    "Io",
    "Runtime",
    "compile",
    "parse_beaver_type",
    "parse_cheetah_ot_kind",
    "parse_data_type",
    "parse_exp_mode",
    "parse_field_type",
    "parse_log_mode",
    "parse_protocol_kind",
    "parse_sigmoid_mode",
    "parse_sort_method",
    "parse_visibility",
    # utils
    "simulation",
    # libs
    "psi",
    # intrinsic
]

check_cpu_feature()
