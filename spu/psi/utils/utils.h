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

#include "yasl/link/link.h"

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

std::string KeysJoin(const std::vector<absl::string_view>& keys);

template <typename T>
T SyncWait(const std::shared_ptr<yasl::link::Context>& lctx,
           std::future<T>* f) {
  std::vector<yasl::Buffer> flag_list;
  std::chrono::seconds span(5);
  while (true) {
    bool done = f->wait_for(span) == std::future_status::ready;
    auto flag = done ? kFinishedFlag : kUnFinishedFlag;
    flag_list = yasl::link::AllGather(lctx, flag, "sync wait");
    if (std::find_if(flag_list.begin(), flag_list.end(),
                     [](const yasl::Buffer& b) {
                       return std::string_view(b.data<char>(), b.size()) ==
                              kUnFinishedFlag;
                     }) == flag_list.end()) {
      // all done
      break;
    }
  }
  return f->get();
}
}  // namespace spu::psi
