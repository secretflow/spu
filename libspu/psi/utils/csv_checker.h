// Copyright 2022 Ant Group Co., Ltd.
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

namespace spu::psi {

class CsvChecker {
 public:
  explicit CsvChecker(const std::string& csv_path,
                      const std::vector<std::string>& schema_names,
                      const std::string& tmp_cache_dir,
                      bool skip_check = false);

  uint32_t data_count() const { return data_count_; }

  std::string hash_digest() const { return hash_digest_; }

 private:
  uint32_t data_count_;

  std::string hash_digest_;
};

}  // namespace spu::psi
