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
#include <memory>
#include <string>
#include <unordered_map>

namespace spu::device {

class Timer {
  using TimePoint = decltype(std::chrono::high_resolution_clock::now());
  TimePoint start_;

public:
  Timer() { reset(); }

  void reset() { start_ = std::chrono::high_resolution_clock::now(); }

  std::chrono::duration<double> count() const {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::duration<double>>(end -
                                                                     start_);
  }
};

class Profiler {
public:
  struct ExecutionRecord {
    // total number of executation.
    size_t count = 0;
    // total elapsed time.
    std::chrono::duration<double> time = {};
  };
  using ExecutionRecordsT = std::unordered_map<std::string, ExecutionRecord>;

  Timer start() const { return {}; }

  void end(std::string_view id, const Timer &time) {
    auto t = time.count();
    auto &record = records_[std::string(id)];
    record.count++;
    record.time += t;
  }

  const ExecutionRecordsT &getRecords() const { return records_; }

private:
  ExecutionRecordsT records_;
};

} // namespace spu::device