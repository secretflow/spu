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

#include <algorithm>
#include <fstream>
#include <map>
#include <set>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/str_split.h"

#include "libspu/core/prelude.h"

namespace spu::psi {

class CsvHeaderAnalyzer {
 public:
  CsvHeaderAnalyzer(const std::string& path,
                    const std::vector<std::string>& target_fields) {
    std::set<std::string> target_set = CheckAndNormalizeTokens(target_fields);

    std::ifstream in(path);
    SPU_ENFORCE(in.is_open(), "Cannot open {}", path);
    std::string line;
    SPU_ENFORCE(std::getline(in, line), "Cannot read header line in {}", path);
    SPU_ENFORCE(!CheckIfBOMExists(line),
                "The file {} starts with BOM(Byte Order Mark).", path);

    std::vector<std::string> headers = GetCsvTokens(line);
    std::map<std::string, size_t> col_index_map;
    size_t idx = 0;
    for (const std::string& header : headers) {
      headers_.push_back(header);
      col_index_map[header] = idx;
      idx++;
    }
    // Iterate by sorted order.
    for (const auto& target : target_set) {
      SPU_ENFORCE(col_index_map.find(target) != col_index_map.end(),
                  "Cannot find feature name='{}' in CSV file header='{}'",
                  target, line);
      target_indices_sorted_.push_back(col_index_map[target]);
    }
    // Iterate by target_fields sequence.
    for (std::string target : target_fields) {
      absl::StripAsciiWhitespace(&target);
      SPU_ENFORCE(col_index_map.find(target) != col_index_map.end(),
                  "Cannot find feature name='{}' in CSV file header='{}'",
                  target, line);
      target_indices_.push_back(col_index_map[target]);
    }
    headers_set_ = CheckAndNormalizeTokens(headers_);
    header_line_ = line;
  }

  // Return fields list in this csv. The sequences are same as the file header.
  const std::vector<std::string>& headers() const { return headers_; }

  // Return sorted fields set in this csv.
  const std::set<std::string>& headers_set() const { return headers_set_; }

  // Return interested fields indices. The indices are stored by sorted target
  // field names order.
  const std::vector<size_t>& target_indices_sorted() const {
    return target_indices_sorted_;
  }

  // Return interested fields indices.
  const std::vector<size_t>& target_indices() const { return target_indices_; }

  const std::string& header_line() const { return header_line_; }

  static std::set<std::string> CheckAndNormalizeTokens(
      const std::vector<std::string>& inputs) {
    std::set<std::string> ret;
    for (std::string input : inputs) {
      absl::StripAsciiWhitespace(&input);
      SPU_ENFORCE(!input.empty(),
                  "Found empty feature name, input feature names='{}'",
                  fmt::join(inputs, ","));
      ret.insert(input);
    }
    SPU_ENFORCE(ret.size() == inputs.size(), "Repeated feature name in ='{}'",
                fmt::join(inputs, ","));
    return ret;
  }

  static std::vector<std::string> GetCsvTokens(const std::string& line) {
    std::vector<std::string> headers = absl::StrSplit(line, ',');
    std::for_each(headers.begin(), headers.end(),
                  [](auto& header) { absl::StripAsciiWhitespace(&header); });
    return headers;
  }

  // Check if the first line starts with BOM(Byte Order Mark).
  static bool CheckIfBOMExists(const std::string& first_line) {
    // Only detect UTF-8 BOM right now.
    if (first_line.length() >= 3 && first_line[0] == '\xEF' &&
        first_line[1] == '\xBB' && first_line[2] == '\xBF') {
      return true;
    } else {
      return false;
    }
  }

 private:
  std::set<std::string> headers_set_;
  std::vector<std::string> headers_;
  // The indices are stored by sorted target field names.
  std::vector<size_t> target_indices_sorted_;
  std::vector<size_t> target_indices_;
  std::string header_line_;
};

}  // namespace spu::psi