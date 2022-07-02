// Copyright 2021 Ant Group Co., Ltd.
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

#include "spu/psi/provider/batch_provider_impl.h"

#include "absl/strings/escaping.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "yasl/base/exception.h"

namespace spu::psi {

std::vector<std::string> MemoryBatchProvider::ReadNextBatch(size_t batch_size) {
  std::vector<std::string> batch;
  YASL_ENFORCE(cursor_index_ <= items_.size());
  size_t n_items = std::min(batch_size, items_.size() - cursor_index_);
  batch.insert(batch.end(), items_.begin() + cursor_index_,
               items_.begin() + cursor_index_ + n_items);
  cursor_index_ += n_items;
  return batch;
}

const std::vector<std::string>& MemoryBatchProvider::items() const {
  return items_;
}

CsvBatchProvider::CsvBatchProvider(
    const std::string& path, const std::vector<std::string>& target_fields)
    : path_(path), analyzer_(path, target_fields) {
  // TODO(airu): CsvHeaderAnalyzer function can move into provider
  in_ = io::BuildInputStream(io::FileIoOptions(path_));
  std::string line;
  YASL_ENFORCE(in_->GetLine(&line), "No header line in file={}", path);
}

std::vector<std::string> CsvBatchProvider::ReadNextBatch(size_t batch_size) {
  std::vector<std::string> ret;
  std::string line;
  while (in_->GetLine(&line)) {
    // TODO(shuyan.ycf): verify dataset & fields types before PSI, probably in
    // schema verification.
    std::vector<absl::string_view> tokens = absl::StrSplit(line, ',');
    std::vector<absl::string_view> targets;
    for (size_t fidx : analyzer_.target_indices()) {
      YASL_ENFORCE(fidx < tokens.size(),
                   "Illegal line due to no field at index={}, line={}", fidx,
                   line);
      targets.push_back(absl::StripAsciiWhitespace(tokens[fidx]));
    }
    ret.push_back(absl::StrJoin(targets, "-"));
    if (ret.size() == batch_size) {
      break;
    }
  }
  return ret;
}

}  // namespace spu::psi