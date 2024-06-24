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

#include <filesystem>
#include <string>

namespace spu::device {

std::string getMetaExtension();

std::filesystem::path getRankFolder(const std::filesystem::path& base,
                                    int64_t rank);

std::filesystem::path getConfigFilePath(const std::filesystem::path& base);

std::filesystem::path getCodeFilePath(const std::filesystem::path& base);

std::filesystem::path getMetaFilePath(const std::filesystem::path& base,
                                      int64_t rank,
                                      const std::string& var_name);

std::filesystem::path getValueChunkFilePath(const std::filesystem::path& base,
                                            int64_t rank,
                                            const std::string& var_name,
                                            int64_t chunk_id);

}  // namespace spu::device
