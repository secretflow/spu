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

#include "libspu/psi/utils/csv_checker.h"

#include <filesystem>

#include "absl/strings/escaping.h"
#include "absl/strings/str_join.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "spdlog/spdlog.h"
#include "yacl/crypto/base/hash/hash_utils.h"
#include "yacl/utils/scope_guard.h"

#include "libspu/core/prelude.h"
#include "libspu/psi/io/io.h"

namespace spu::psi {

namespace {
// Check if the first line starts with BOM(Byte Order Mark).
bool CheckIfBOMExists(const std::string& file_path) {
  std::string first_line;
  {
    io::FileIoOptions file_opts(file_path);
    auto file_is = io::BuildInputStream(file_opts);
    file_is->GetLine(&first_line, '\n');
    file_is->Close();
  }

  // Only detect UTF-8 BOM right now.
  return first_line.length() >= 3 && first_line[0] == '\xEF' &&
         first_line[1] == '\xBB' && first_line[2] == '\xBF';
}
}  // namespace

CsvChecker::CsvChecker(const std::string& csv_path,
                       const std::vector<std::string>& schema_names,
                       const std::string& tmp_cache_dir, bool skip_check)
    : data_count_(0) {
  SPU_ENFORCE(!CheckIfBOMExists(csv_path),
              "the file {} starts with BOM(Byte Order Mark).", csv_path);

  size_t duplicated_size = 0;
  std::vector<std::string> duplicated_keys;
  yacl::crypto::SslHash hash_obj(yacl::crypto::HashAlgorithm::SHA256);

  io::FileIoOptions file_opts(csv_path);
  io::CsvOptions csv_opts;
  csv_opts.read_options.file_schema.feature_names = schema_names;
  csv_opts.read_options.file_schema.feature_types.resize(schema_names.size(),
                                                         io::Schema::STRING);
  auto csv_reader = io::BuildReader(file_opts, csv_opts);

  auto timestamp_str = std::to_string(absl::ToUnixNanos(absl::Now()));
  std::string keys_file = fmt::format("selected-keys.{}", timestamp_str);

  io::FileIoOptions tmp_file_ops(keys_file);
  auto keys_os = io::BuildOutputStream(tmp_file_ops);
  ON_SCOPE_EXIT([&] {
    std::error_code ec;
    std::filesystem::remove(tmp_file_ops.file_name, ec);
    if (ec.value() != 0) {
      SPDLOG_WARN("can not remove tmp file: {}, msg: {}",
                  tmp_file_ops.file_name, ec.message());
    }
  });

  // read csv file by row
  io::ColumnVectorBatch batch;
  while (csv_reader->Next(&batch)) {
    for (size_t row = 0; row < batch.Shape().rows; row++) {
      std::vector<std::string> chosen;
      for (size_t col = 0; col < batch.Shape().cols; col++) {
        const auto& token = batch.At<std::string>(row, col);
        SPU_ENFORCE(token.size(), "empty token in row={} field={}",
                    data_count_ + row, schema_names[col]);
        chosen.push_back(token);
      }
      // if combined_id is about 128bytes
      // .keys file tasks almost 12GB for 10^8 samples.
      std::string combined_id = absl::StrJoin(chosen, "-");
      hash_obj.Update(combined_id);
      if (!skip_check) {
        keys_os->Write(combined_id.data(), combined_id.size());
        keys_os->Write("\n", 1);
      }
    }
    data_count_ += batch.Shape().rows;
  }
  keys_os->Close();

  if (!skip_check) {
    std::string duplicated_keys_file =
        fmt::format("duplicate-keys.{}", timestamp_str);
    ON_SCOPE_EXIT([&] {
      std::error_code ec;
      std::filesystem::remove(duplicated_keys_file, ec);
      if (ec.value() != 0) {
        SPDLOG_WARN("can not remove tmp file: {}, msg: {}",
                    duplicated_keys_file, ec.message());
      }
    });

    std::string cmd = fmt::format(
        "LC_ALL=C sort --buffer-size=1G --temporary-directory={} "
        "--stable {} | LC_ALL=C uniq -d > {}",
        tmp_cache_dir, keys_file, duplicated_keys_file);
    SPDLOG_INFO("Executing duplicated scripts: {}", cmd);
    int ret = system(cmd.c_str());
    SPU_ENFORCE(ret == 0, "failed to execute cmd={}, ret={}", cmd, ret);
    io::FileIoOptions dup_keys_file_opts(duplicated_keys_file);
    auto duplicated_is = io::BuildInputStream(dup_keys_file_opts);
    std::string duplicated_key;
    while (duplicated_is->GetLine(&duplicated_key)) {
      if (duplicated_size++ < 10) {
        duplicated_keys.push_back(duplicated_key);
      }
    }
    // not precise size if some key repeat more than 2 times.
    SPU_ENFORCE(duplicated_size == 0, "found duplicated keys: {}",
                fmt::join(duplicated_keys, ","));
  }

  std::vector<uint8_t> digest = hash_obj.CumulativeHash();
  hash_digest_ = absl::BytesToHexString(absl::string_view(
      reinterpret_cast<const char*>(digest.data()), digest.size()));
}

}  // namespace spu::psi
