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

#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "yacl/link/link.h"

#include "libspu/psi/utils/serializable.pb.h"

namespace spu::psi {

namespace {

static const std::string kFinishedFlag = "p_finished";
static const std::string kUnFinishedFlag = "p_unfinished";

}  // namespace

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
                  const std::vector<std::string>& keys);

// `indices` must be sorted
void FilterFileByIndices(const std::string& input, const std::string& output,
                         const std::vector<uint64_t>& indices,
                         size_t header_line_count = 1);

// join keys with "-"
std::string KeysJoin(const std::vector<absl::string_view>& keys);

std::vector<size_t> AllGatherItemsSize(
    const std::shared_ptr<yacl::link::Context>& link_ctx, size_t self_size);

template <typename T>
T SyncWait(const std::shared_ptr<yacl::link::Context>& lctx,
           std::future<T>* f) {
  std::vector<yacl::Buffer> flag_list;
  std::chrono::seconds span(5);
  while (true) {
    bool done = f->wait_for(span) == std::future_status::ready;
    auto flag = done ? kFinishedFlag : kUnFinishedFlag;
    flag_list = yacl::link::AllGather(lctx, flag, "sync wait");
    if (std::find_if(flag_list.begin(), flag_list.end(),
                     [](const yacl::Buffer& b) {
                       return std::string_view(b.data<char>(), b.size()) ==
                              kUnFinishedFlag;
                     }) == flag_list.end()) {
      // all done
      break;
    }
  }
  return f->get();
}

std::vector<size_t> GetShuffledIdx(size_t items_size);

std::vector<uint8_t> PaddingData(yacl::ByteContainerView data, size_t max_len);
std::string UnPaddingData(yacl::ByteContainerView data);

}  // namespace spu::psi
