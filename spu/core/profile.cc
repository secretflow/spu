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

#include "spu/core/profile.h"

#include <atomic>
#include <mutex>

namespace spu {

void ProfilingContext::addRecord(ActionRecord&& record) {
  if (profiling_enabled_) {
    auto& stat_rec = action_stats_[record.name];
    stat_rec.count++;
    stat_rec.total_time +=
        std::chrono::duration_cast<Duration>(record.end - record.start);
  }
}

void ProfilingContext::clearProfilingRecords() { action_stats_.clear(); }

// global uuid, for multiple profiling context.
static inline size_t genActionUuid() {
  static std::atomic<size_t> s_counter = 0;
  return ++s_counter;
}

ProfileGuard::ProfileGuard(ProfilingContext* pctx, std::string_view name,
                           std::string_view params, bool supress_sub_profiling)
    : pctx_(pctx) {
  record_.id = genActionUuid();
  record_.name = name;
  record_.start = std::chrono::high_resolution_clock::now();

  cur_profiling_enabled_ = pctx->getProfilingEnabled();
  if (supress_sub_profiling) {
    pctx->setProfilingEnabled(false);
  }
}

ProfileGuard::~ProfileGuard() {
  record_.end = std::chrono::high_resolution_clock::now();
  pctx_->setProfilingEnabled(cur_profiling_enabled_);
  pctx_->addRecord(std::move(record_));
}

// trace related.
std::shared_ptr<spdlog::logger> spuTraceLog() {
  static std::once_flag flag;
  static const char* const kLoggerName = "spu_trace_logger";

  std::call_once(flag, []() {
    auto logger = std::make_shared<spdlog::logger>(
        kLoggerName,
        std::make_shared<spdlog::sinks::ansicolor_stdout_sink_mt>());
    logger->set_pattern("[SPU-TRACE] %v");
    spdlog::register_logger(logger);
  });
  return spdlog::get(kLoggerName);
}

}  // namespace spu
