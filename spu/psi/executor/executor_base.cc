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

#include "spu/psi/executor/executor_base.h"

#include <filesystem>
#include <numeric>

#include "absl/strings/escaping.h"
#include "absl/strings/str_join.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "spdlog/spdlog.h"
#include "yasl/base/exception.h"
#include "yasl/crypto/hash_util.h"
#include "yasl/utils/scope_guard.h"
#include "yasl/utils/serialize.h"

#include "spu/psi/io/io.h"
#include "spu/psi/provider/csv_header_analyzer.h"

namespace spu::psi {

namespace {

constexpr size_t kCsvHeaderLineCount = 1;

// Multiple-Key out-of-core sort.
// Out-of-core support reference:
//   http://vkundeti.blogspot.com/2008/03/tech-algorithmic-details-of-unix-sort.html
// Multiple-Key support reference:
//   https://stackoverflow.com/questions/9471101/sort-csv-file-by-column-priority-using-the-sort-command
// use POSIX locale for sort
//   https://unix.stackexchange.com/questions/43465/whats-the-default-order-of-linux-sort/43466
//
// NOTE:
// This implementation requires `sort` command, which is guaranteed by our
// docker-way ship.
void MultiKeySort(const std::string& in_csv, const std::string& out_csv,
                  const std::vector<std::string>& keys) {
  std::string line;
  {
    io::FileIoOptions op(in_csv);
    auto in = io::BuildInputStream(op);
    YASL_ENFORCE(in->GetLine(&line), "{}: No header line for csv file", in_csv);
  }

  {
    // Copy head line to out_csv
    // Add scope to flush write here.
    io::FileIoOptions op(out_csv);
    auto out = io::BuildOutputStream(op);
    out->Write(line);
    out->Write("\n");
    out->Close();
  }

  // Construct sort key indices.
  CsvHeaderAnalyzer analyzer(in_csv, keys);
  // NOTE: `sort` cmd starts from index 1.
  std::vector<std::string> sort_keys;
  for (size_t index : analyzer.target_indices()) {
    // About `sort --key=KEYDEF`
    //
    // KEYDEF is F[.C][OPTS][,F[.C][OPTS]] for start and stop position, where
    // F is a field number and C a character position in the field; both are
    // origin 1, and the stop position defaults to the line's end.  If neither
    // -t nor -b is in effect, characters in a field are counted from the
    // beginning of the preceding whitespace.  OPTS is one or more
    // single-letter ordering options [bdfgiMhnRrV], which override global
    // ordering options for that key.  If no key is given, use the entire line
    // as the key.
    //
    // I have already verified `sort --key=3,3 --key=1,1` will firstly sort by
    // 3rd field and then 1st field.
    constexpr size_t kOffset = 1;
    size_t key_index = kOffset + index;
    sort_keys.push_back(fmt::format("--key={},{}", key_index, key_index));
  }
  YASL_ENFORCE(sort_keys.size() == keys.size(),
               "Mismatched header, field_names={}, line={}",
               fmt::join(keys, ","), line);

  // Sort the csv body and append to out csv.
  std::string cmd = fmt::format(
      "tail -n +2 {} | LC_ALL=C sort --buffer-size=1G "
      "--temporary-directory=./ --stable --field-separator=, {} >>{}",
      in_csv, fmt::join(sort_keys, " "), out_csv);
  SPDLOG_INFO("Executing sort scripts: {}", cmd);
  int ret = system(cmd.c_str());
  SPDLOG_INFO("Finished sort scripts: {}, ret={}", cmd, ret);
  YASL_ENFORCE(ret == 0, "Failed to execute cmd={}, ret={}", cmd, ret);
}

void FilterFileByIndices(const std::string& input, const std::string& output,
                         const std::vector<unsigned>& payload_indices,
                         size_t header_line_count) {
  auto in = io::BuildInputStream(io::FileIoOptions(input));
  auto out = io::BuildOutputStream(io::FileIoOptions(output));

  std::string line;
  size_t idx = 0;
  auto indices_iter = payload_indices.begin();
  while (in->GetLine(&line)) {
    if (idx < header_line_count) {
      out->Write(line);
      out->Write("\n");
    } else {
      if (indices_iter == payload_indices.end()) {
        break;
      }
      if (*indices_iter == idx - header_line_count) {
        indices_iter++;
        out->Write(line);
        out->Write("\n");
      }
    }
    idx++;
  }
  out->Close();
}

bool HashListEqualTest(const std::vector<yasl::Buffer>& hash_list) {
  YASL_ENFORCE(!hash_list.empty(), "unsupported hash_list size={}",
               hash_list.size());

  for (size_t idx = 1; idx < hash_list.size(); idx++) {
    if (hash_list[idx] == hash_list[0]) {
      continue;
    }
    return false;
  }
  return true;
}

void DatasetPreCheck(const std::string& path,
                     const std::vector<std::string>& id_fields,
                     size_t* total_size, std::string* hash_digest) {
  size_t original_size = 0;
  size_t duplicated_size = 0;
  std::vector<std::string> duplicated_keys;
  yasl::crypto::SslHash hash_obj(yasl::crypto::HashAlgorithm::SHA256);

  io::FileIoOptions in_file_ops(path);
  io::CsvOptions csv_ops;
  csv_ops.read_options.file_schema.feature_names = id_fields;
  csv_ops.read_options.file_schema.feature_types.resize(id_fields.size(),
                                                        io::Schema::STRING);
  auto csv_reader = io::BuildReader(in_file_ops, csv_ops);

  auto timestamp_str = std::to_string(absl::ToUnixNanos(absl::Now()));
  std::string keys_file = fmt::format("{}.keys.{}", path, timestamp_str);

  io::FileIoOptions tmp_file_ops(keys_file);
  auto keys_os = io::BuildOutputStream(tmp_file_ops);
  ON_SCOPE_EXIT([&] { std::filesystem::remove_all(tmp_file_ops.file_name); });

  io::ColumnVectorBatch batch;
  while (csv_reader->Next(&batch)) {
    for (size_t row = 0; row < batch.Shape().rows; row++) {
      std::vector<std::string> chosen;
      for (size_t col = 0; col < batch.Shape().cols; col++) {
        const auto& token = batch.At<std::string>(row, col);
        YASL_ENFORCE(token.size(), "Empty token in row={} field={}",
                     original_size + row, id_fields[col]);
        chosen.push_back(token);
      }
      // if combined_id is about 128bytes
      // .keys file tasks almost 12GB for 10^8 samples.
      std::string combined_id = absl::StrJoin(chosen, "-");
      hash_obj.Update(combined_id);
      keys_os->Write(combined_id.data(), combined_id.size());
      keys_os->Write("\n", 1);
    }
    original_size += batch.Shape().rows;
  }
  keys_os->Close();

  std::string duplicated_keys_file =
      fmt::format("{}.duplicated.{}", path, timestamp_str);
  std::string cmd = fmt::format(
      "LC_ALL=C sort --buffer-size=1G --temporary-directory=./ "
      "--stable {} | LC_ALL=C uniq -d > {}",
      keys_file, duplicated_keys_file);
  SPDLOG_INFO("Executing duplicated scripts: {}", cmd);
  int ret = system(cmd.c_str());
  ON_SCOPE_EXIT([&] { std::filesystem::remove_all(duplicated_keys_file); });
  SPDLOG_INFO("Finished duplicated scripts: {}, ret={}", cmd, ret);
  YASL_ENFORCE(ret == 0, "Failed to execute cmd={}, ret={}", cmd, ret);
  io::FileIoOptions dup_file_ops(duplicated_keys_file);
  auto duplicated_is = io::BuildInputStream(dup_file_ops);
  std::string duplicated_key;
  while (duplicated_is->GetLine(&duplicated_key)) {
    if (duplicated_size++ < 10) {
      duplicated_keys.push_back(duplicated_key);
    }
  }

  // not precise size if some key repeat more than 2 times.
  YASL_ENFORCE(duplicated_size == 0, "Found duplicated keys: {}",
               fmt::join(duplicated_keys, ","));

  *total_size = original_size;

  std::vector<uint8_t> digest = hash_obj.CumulativeHash();
  *hash_digest = absl::BytesToHexString(absl::string_view(
      reinterpret_cast<const char*>(digest.data()), digest.size()));
}

}  // namespace

PsiExecutorBase::PsiExecutorBase(PsiExecBaseOptions options)
    : options_(std::move(options)), input_data_count_(0) {}

void PsiExecutorBase::Init() {
  // TODO: move common sanity check to there, e.g. check input file exists.

  OnInit();

  // Create output folder automatically.
  auto out_dir_path = std::filesystem::path(options_.out_path).parent_path();

  std::error_code ec;
  std::filesystem::create_directory(out_dir_path, ec);

  YASL_ENFORCE(ec.value() == 0,
               "failed to create output dir={} for path={}, reason = {}",
               out_dir_path.string(), options_.out_path, ec.message());
}

void PsiExecutorBase::Run(PsiReport* report) {
  // step 1: dataset pre check
  std::string hash_digest;
  {
    SPDLOG_INFO("Begin sanity check for input file: {}", options_.in_path);
    DatasetPreCheck(options_.in_path, options_.field_names, &input_data_count_,
                    &hash_digest);
    SPDLOG_INFO("End sanity check for input file: {}, size={}",
                options_.in_path, input_data_count_);
  }
  auto digest_list =
      yasl::link::AllGather(options_.link_ctx, hash_digest, "PSI:SYNC_DIGEST");
  YASL_ENFORCE(digest_list.size() >= 2);

  // step 2: run psi
  std::vector<unsigned> indices;
  bool digest_hash_equal = HashListEqualTest(digest_list);
  if (!digest_hash_equal) {
    OnRun(&indices);
  } else {
    SPDLOG_INFO("skip doing psi, because dataset has been aligned!");
    indices.resize(input_data_count_);
    std::iota(indices.begin(), indices.end(), 0);
  }

  // step 3: filter dataset
  SPDLOG_INFO("Begin post filtering, indices.size={}, should_sort={}",
              indices.size(), options_.should_sort);
  if (options_.should_sort && !digest_hash_equal) {
    std::string out_path_unsorted = options_.out_path + ".unsorted";
    // Register remove of temp file.
    ON_SCOPE_EXIT([&] {
      if (std::remove(out_path_unsorted.c_str()) != 0) {
        SPDLOG_WARN("Cannot remove {}, error={}", out_path_unsorted,
                    strerror(errno));
      }
    });
    FilterFileByIndices(options_.in_path, out_path_unsorted, indices,
                        kCsvHeaderLineCount);
    SPDLOG_INFO("End post filtering, in={}, out={}", options_.in_path,
                out_path_unsorted);
    MultiKeySort(out_path_unsorted, options_.out_path, options_.field_names);
  } else {
    FilterFileByIndices(options_.in_path, options_.out_path, indices,
                        kCsvHeaderLineCount);
    SPDLOG_INFO("End post filtering, in={}, out={}", options_.in_path,
                options_.out_path);
  }

  // fill report
  report->intersection_count = indices.size();
  report->original_count = input_data_count_;
}

void PsiExecutorBase::Stop() { OnStop(); }

}  // namespace spu::psi
