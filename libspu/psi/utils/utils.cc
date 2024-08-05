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

#include "libspu/psi/utils/utils.h"

#include "spdlog/spdlog.h"
#include "yacl/crypto/utils/rand.h"

#include "libspu/core/prelude.h"
#include "libspu/psi/io/io.h"
#include "libspu/psi/utils/csv_header_analyzer.h"
#include "libspu/psi/utils/serialize.h"

namespace spu::psi {

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
  CsvHeaderAnalyzer analyzer(in_csv, keys);

  std::string line;
  {
    // Copy head line to out_csv
    // Add scope to flush write here.
    io::FileIoOptions in_opts(in_csv);
    auto in = io::BuildInputStream(in_opts);
    in->GetLine(&line);
    in->Close();

    io::FileIoOptions out_opts(out_csv);
    auto out = io::BuildOutputStream(out_opts);
    out->Write(line);
    out->Write("\n");
    out->Close();
  }

  // Construct sort key indices.
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
  SPU_ENFORCE(sort_keys.size() == keys.size(),
              "mismatched header, field_names={}, line={}",
              fmt::join(keys, ","), line);

  // Sort the csv body and append to out csv.
  std::string cmd = fmt::format(
      "tail -n +2 {} | LC_ALL=C sort --buffer-size=3G --parallel=8 "
      "--temporary-directory=./ --stable --field-separator=, {} >>{}",
      in_csv, fmt::join(sort_keys, " "), out_csv);
  SPDLOG_INFO("Executing sort scripts: {}", cmd);
  int ret = system(cmd.c_str());
  SPDLOG_INFO("Finished sort scripts: {}, ret={}", cmd, ret);
  SPU_ENFORCE(ret == 0, "failed to execute cmd={}, ret={}", cmd, ret);
}

void FilterFileByIndices(const std::string& input, const std::string& output,
                         const std::vector<uint64_t>& indices,
                         size_t header_line_count) {
  auto in = io::BuildInputStream(io::FileIoOptions(input));
  auto out = io::BuildOutputStream(io::FileIoOptions(output));

  std::string line;
  size_t idx = 0;
  size_t target_count = 0;
  auto indices_iter = indices.begin();
  while (in->GetLine(&line)) {
    if (idx < header_line_count) {
      out->Write(line);
      out->Write("\n");
    } else {
      if (indices_iter == indices.end()) {
        break;
      }
      if (*indices_iter == idx - header_line_count) {
        indices_iter++;
        out->Write(line);
        out->Write("\n");
        ++target_count;
      }
    }
    idx++;
  }

  SPU_ENFORCE(target_count == indices.size(),
              "logstic error, indices.size={}, target_count={}, please be "
              "sure the `indices` is sorted",
              indices.size(), target_count);

  out->Close();
  in->Close();
}

std::string KeysJoin(const std::vector<absl::string_view>& keys) {
  return absl::StrJoin(keys, ",");
}

std::vector<size_t> AllGatherItemsSize(
    const std::shared_ptr<yacl::link::Context>& link_ctx, size_t self_size) {
  std::vector<size_t> items_size_list(link_ctx->WorldSize());

  std::vector<yacl::Buffer> items_size_buf_list = yacl::link::AllGather(
      link_ctx, utils::SerializeSize(self_size), "PSI:SYNC_SIZE");

  for (size_t idx = 0; idx < items_size_buf_list.size(); idx++) {
    items_size_list[idx] = utils::DeserializeSize(items_size_buf_list[idx]);
  }

  return items_size_list;
}

std::vector<size_t> GetShuffledIdx(size_t items_size) {
  std::mt19937 rng(yacl::crypto::RandU64());

  std::vector<size_t> shuffled_idx_vec(items_size);
  std::iota(shuffled_idx_vec.begin(), shuffled_idx_vec.end(), 0);
  std::shuffle(shuffled_idx_vec.begin(), shuffled_idx_vec.end(), rng);

  return shuffled_idx_vec;
}

// pad data to max_len bytes
// format len(32bit)||data||00..00
std::vector<uint8_t> PaddingData(yacl::ByteContainerView data, size_t max_len) {
  SPU_ENFORCE((data.size() + 4) <= max_len, "data_size:{} max_len:{}",
              data.size(), max_len);

  std::vector<uint8_t> data_with_padding(max_len);
  uint32_t data_size = data.size();

  std::memcpy(&data_with_padding[0], &data_size, sizeof(uint32_t));
  std::memcpy(&data_with_padding[4], data.data(), data.size());

  return data_with_padding;
}

std::string UnPaddingData(yacl::ByteContainerView data) {
  uint32_t data_len;
  std::memcpy(&data_len, data.data(), sizeof(data_len));

  SPU_ENFORCE((data_len + sizeof(uint32_t)) <= data.size());

  std::string ret(data_len, '\0');
  std::memcpy(ret.data(), data.data() + 4, data_len);

  return ret;
}

}  // namespace spu::psi
