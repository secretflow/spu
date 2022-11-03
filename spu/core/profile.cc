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
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "yasl/base/exception.h"

#ifdef __APPLE__
#include <mach/mach.h>
#endif

namespace spu {

namespace {

#ifdef __linux__

std::string ReadProcSelfStatusByKey(const std::string& key) {
  std::string ret;
  std::ifstream self_status("/proc/self/status");
  std::string line;
  while (std::getline(self_status, line)) {
    std::vector<absl::string_view> fields =
        absl::StrSplit(line, absl::ByChar(':'));
    if (fields.size() == 2 && key == absl::StripAsciiWhitespace(fields[0])) {
      ret = absl::StripAsciiWhitespace(fields[1]);
    }
  }
  return ret;
}

float ReadVMxFromProcSelfStatus(const std::string& key) {
  const std::string str_usage = ReadProcSelfStatusByKey(key);
  std::vector<absl::string_view> fields =
      absl::StrSplit(str_usage, absl::ByChar(' '));
  if (fields.size() == 2) {
    size_t ret = 0;
    if (!absl::SimpleAtoi(fields[0], &ret)) {
      return -1;
    }
    return static_cast<float>(ret) / 1024 / 1024;
  }
  return -1;
}

float GetPeakMemUsage() { return ReadVMxFromProcSelfStatus("VmHWM"); }

float GetCurrentMemUsage() { return ReadVMxFromProcSelfStatus("VmRSS"); }

#elif defined(__APPLE__)

float GetCurrentMemUsage() {
  struct mach_task_basic_info t_info;
  mach_msg_type_number_t t_info_count = MACH_TASK_BASIC_INFO_COUNT;

  auto ret = task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                       reinterpret_cast<task_info_t>(&t_info), &t_info_count);

  if (KERN_SUCCESS != ret || MACH_TASK_BASIC_INFO_COUNT != t_info_count) {
    return -1;
  }
  return static_cast<float>(t_info.resident_size) / 1024 / 1024;
}

float GetPeakMemUsage() {
  struct mach_task_basic_info t_info;
  mach_msg_type_number_t t_info_count = MACH_TASK_BASIC_INFO_COUNT;

  auto ret = task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                       reinterpret_cast<task_info_t>(&t_info), &t_info_count);

  if (KERN_SUCCESS != ret || MACH_TASK_BASIC_INFO_COUNT != t_info_count) {
    return -1;
  }
  return static_cast<float>(t_info.resident_size_max) / 1024 / 1024;
}

#endif

}  // namespace

void MemProfilingGuard::enable(int i, std::string_view m, std::string_view n) {
  indent_ = i * 2;
  module_ = m;
  name_ = n;
  start_peak_ = GetPeakMemUsage();
  enable_ = true;
  spuTraceLog()->info("{}{}.{}: before peak {:.2f}GB, current {:.2f}GB",
                      std::string(indent_, ' '), module_, name_, start_peak_,
                      GetCurrentMemUsage());
}

MemProfilingGuard::~MemProfilingGuard() {
  if (!enable_) {
    return;
  }
  auto p = GetPeakMemUsage();
  auto increase = p - start_peak_;
  if (increase >= 0.01) {
    spuTraceLog()->info(
        "{}{}.{}: peak {:.2f}GB, increase {:.2f}GB, current {:.2f}GB",
        std::string(indent_, ' '), module_, name_, p, increase,
        GetCurrentMemUsage());
  } else {
    spuTraceLog()->info("{}{}.{}: peak {:.2f}GB, current {:.2f}GB",
                        std::string(indent_, ' '), module_, name_, p,
                        GetCurrentMemUsage());
  }
}

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
