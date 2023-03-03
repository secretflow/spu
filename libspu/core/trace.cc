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

#include "libspu/core/trace.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <unordered_map>
#include <utility>

#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"

#include "libspu/core/prelude.h"

#ifdef __APPLE__
#include <mach/mach.h>
#endif

namespace spu {
namespace internal {

// global uuid, for multiple profiling context.
int64_t genActionUuid() {
  static std::atomic<int64_t> s_counter = 0;
  return ++s_counter;
}

}  // namespace internal

namespace {

#ifdef __linux__

[[maybe_unused]] std::string ReadProcSelfStatusByKey(const std::string& key) {
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

[[maybe_unused]] float ReadVMxFromProcSelfStatus(const std::string& key) {
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

[[maybe_unused]] float GetPeakMemUsage() {
  return ReadVMxFromProcSelfStatus("VmHWM");
}

[[maybe_unused]] float GetCurrentMemUsage() {
  return ReadVMxFromProcSelfStatus("VmRSS");
}

#elif defined(__APPLE__)

[[maybe_unused]] float GetCurrentMemUsage() {
  struct mach_task_basic_info t_info;
  mach_msg_type_number_t t_info_count = MACH_TASK_BASIC_INFO_COUNT;

  auto ret = task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                       reinterpret_cast<task_info_t>(&t_info), &t_info_count);

  if (KERN_SUCCESS != ret || MACH_TASK_BASIC_INFO_COUNT != t_info_count) {
    return -1;
  }
  return static_cast<float>(t_info.resident_size) / 1024 / 1024;
}

[[maybe_unused]] float GetPeakMemUsage() {
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

[[maybe_unused]] std::string getIndentString(size_t indent) {
  constexpr size_t kMaxIndent = 30;

  static std::once_flag flag;
  static std::array<std::string, kMaxIndent> cache;
  std::call_once(flag, [&]() {
    for (size_t idx = 0; idx < kMaxIndent; idx++) {
      cache[idx] = std::string(idx * 2, ' ');
    }
  });

  return cache[std::min(indent, kMaxIndent - 1)];
}

// global variables.
std::mutex g_tracer_map_mutex;
std::unordered_map<std::string, std::shared_ptr<Tracer>> g_tracers;
std::mutex g_trace_flags_map_mutex;
std::unordered_map<std::string, int64_t> g_trace_flags;
std::shared_ptr<spdlog::logger> g_trace_logger;

std::shared_ptr<spdlog::logger> defaultTraceLogger() {
  static std::once_flag flag;
  static std::shared_ptr<spdlog::logger> default_logger;

  std::call_once(flag, []() {
    auto console_sink =
        std::make_shared<spdlog::sinks::ansicolor_stdout_sink_mt>();
    default_logger = std::make_shared<spdlog::logger>("TR", console_sink);
    // default value, for reference.
    // default_logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%l] %v");
    default_logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] %v");
  });

  return default_logger;
}

void setTraceLogger(std::shared_ptr<spdlog::logger> logger) {
  g_trace_logger = std::move(logger);
}

std::shared_ptr<spdlog::logger> getTraceLogger() {
  if (!g_trace_logger) {
    g_trace_logger = defaultTraceLogger();
  }
  return g_trace_logger;
}

}  // namespace

void Tracer::logActionBegin(int64_t id, const std::string& mod,
                            const std::string& name,
                            const std::string& detail) const {
  const auto indent = getIndentString(depth_);

  if ((flag_ & TR_LOGM) != 0) {
    getTraceLogger()->info("[B] [M{}] {}{}.{}({})", GetPeakMemUsage(), indent,
                           mod, name, detail);
  } else {
    getTraceLogger()->info("[B] {}{}.{}({})", indent, mod, name, detail);
  }
}

void Tracer::logActionEnd(int64_t id, const std::string& mod,
                          const std::string& name,
                          const std::string& detail) const {
  const auto indent = getIndentString(depth_);

  if ((flag_ & TR_LOGM) != 0) {
    getTraceLogger()->info("[E] [M{}] {}{}.{}({})", GetPeakMemUsage(), indent,
                           mod, name, detail);
  } else {
    getTraceLogger()->info("[E] {}{}.{}({})", indent, mod, name, detail);
  }
}

void initTrace(const std::string& ctx_id, int64_t tr_flag,
               const std::shared_ptr<spdlog::logger>& tr_logger) {
  {
    std::unique_lock lock(g_trace_flags_map_mutex);
    g_trace_flags[ctx_id] = tr_flag;
  }
  // we may trigger initTrace several times with different rt_config in python
  {
    std::unique_lock lock(g_tracer_map_mutex);
    g_tracers.erase(ctx_id);
  }
  if (tr_logger) {
    setTraceLogger(tr_logger);
  }
}

int64_t getGlobalTraceFlag(const std::string& id) {
  std::unique_lock lock(g_trace_flags_map_mutex);
  return g_trace_flags[id];
}

std::shared_ptr<Tracer> getTracer(const std::string& id,
                                  const std::string& pid) {
  std::unique_lock lock(g_tracer_map_mutex);
  auto itr = g_tracers.find(id);
  if (itr != g_tracers.end()) {
    return itr->second;
  }

  if (pid.empty()) {
    // this is a new trace tree, use global trace flag.
    auto trace_flag = getGlobalTraceFlag(id);
    auto tracer = std::make_shared<Tracer>(trace_flag);
    g_tracers.emplace(id, tracer);
    return tracer;
  } else {
    itr = g_tracers.find(pid);
    if (itr != g_tracers.end()) {
      // if not found and parent exist, clone from it.
      auto n_tracer = std::make_shared<Tracer>(*itr->second);
      g_tracers.emplace(id, n_tracer);
      return n_tracer;
    } else {
      // parent has no tracer, maybe parent never traced an action.
      SPDLOG_WARN("parent({}) tracer never triggered", pid);

      // make a fresh tracer to let the program go.
      auto trace_flag = getGlobalTraceFlag(id);
      auto tracer = std::make_shared<Tracer>(trace_flag);
      g_tracers.emplace(id, tracer);
      return tracer;
    }
  }
}

void MemProfilingGuard::enable(int i, std::string_view m, std::string_view n) {
  indent_ = i * 2;
  module_ = m;
  name_ = n;
  start_peak_ = GetPeakMemUsage();
  enable_ = true;
  SPDLOG_DEBUG("{}{}.{}: before peak {:.2f}GB, current {:.2f}GB",
               std::string(indent_, ' '), module_, name_, start_peak_,
               GetCurrentMemUsage());
}

MemProfilingGuard::~MemProfilingGuard() {
  if (!enable_) {
    return;
  }
  auto p = GetPeakMemUsage();
  auto increase = p - start_peak_;
  if (increase >= 0.01) {  // NOLINT
    SPDLOG_DEBUG("{}{}.{}: peak {:.2f}GB, increase {:.2f}GB, current {:.2f}GB",
                 std::string(indent_, ' '), module_, name_, p, increase,
                 GetCurrentMemUsage());
  } else {
    SPDLOG_DEBUG("{}{}.{}: peak {:.2f}GB, current {:.2f}GB",
                 std::string(indent_, ' '), module_, name_, p,
                 GetCurrentMemUsage());
  }
}

}  // namespace spu
