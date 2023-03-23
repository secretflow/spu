// Copyright 2023 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <string>
#include <vector>

#include "libspu/psi/core/labeled_psi/psi_params.h"

namespace spu {
namespace pir {
namespace examples {
namespace utils {

std::vector<uint8_t> ReadEcSecretKeyFile(const std::string &file_path);

size_t CsvFileDataCount(const std::string &file_path,
                        const std::vector<std::string> &ids);

void WritePsiParams(const std::string file_path,
                    const apsi::PSIParams &psi_params);

apsi::PSIParams ReadPsiParams(const std::string &file_path);

}  // namespace utils
}  // namespace examples
}  // namespace pir
}  // namespace spu
