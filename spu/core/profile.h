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

#pragma once

#include <chrono>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "fmt/format.h"
#include "fmt/ostream.h"
#include "spdlog/spdlog.h"

namespace spu {

using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
using Duration = std::chrono::nanoseconds;
using ActionUuid = size_t;

struct ActionRecord {
  // the uuid of this record.
  ActionUuid id;
  // name of this record. i.e. fmul, _mul, _mul_ss
  // note: we assume `name` is function name, which is statically stored, and we
  // dont have to own its lifetime.
  std::string_view name;
  // record of same name could have different parameters.
  // std::string params;
  // timing stuffs.
  TimePoint start;
  TimePoint end;
};

struct ActionStatistic {
  // number of actions executed.
  size_t count;
  // total duration time.
  Duration total_time = {};
};

// map type to its statistics
using ActionStatistics = std::unordered_map<std::string_view, ActionStatistic>;

// Note: this class is NOT designed to be thread-safe. A context should be used
// exactly in one thread, the context and threading is designed to be 1-1
// relationship.
//
// Warn: Please double check benchmark metrics when you change this module. The
// profiler itself may add extra overhead and cause the result inaccurate.
class ProfilingContext {
 protected:
  bool tracing_enabled_ = false;
  bool profiling_enabled_ = false;
  int64_t action_depth_ = 0;

  ActionStatistics action_stats_;

  // TODO: slow and high memory footprint, we should use a analyser to
  // reconstruct the records from logs.
  // std::vector<ActionRecord> records_;

 public:
  virtual ~ProfilingContext() = default;

  void setProfilingEnabled(bool enable) { profiling_enabled_ = enable; }
  bool getProfilingEnabled() const { return profiling_enabled_; }

  void addRecord(ActionRecord&& record);
  const ActionStatistics& getActionStats() const { return action_stats_; }
  void clearProfilingRecords();

  // The tracing related.
  // (jint) maybe it's not a good idea to mix profling & tracing together.
  void setTracingEnabled(bool enable) { tracing_enabled_ = enable; }
  bool getTracingEnabled() const { return tracing_enabled_; }

  void incTracingDepth() { action_depth_++; }
  void decTracingDepth() { action_depth_--; }
  void setTracingDepth(size_t new_depth) { action_depth_ = new_depth; }
  size_t getTracingDepth() const { return action_depth_; }
};

class ProfileGuard final {
  ProfilingContext* pctx_;
  ActionRecord record_;
  bool cur_profiling_enabled_ = false;

 public:
  explicit ProfileGuard(ProfilingContext* pctx, std::string_view name,
                        std::string_view params,
                        bool supress_sub_profiling = false);
  ~ProfileGuard();
};

class TraceGuard final {
  ProfilingContext* pctx_;

 public:
  explicit TraceGuard(ProfilingContext* pctx) : pctx_(pctx) {
    pctx_->incTracingDepth();
  }
  ~TraceGuard() { pctx_->decTracingDepth(); }
};

std::shared_ptr<spdlog::logger> spuTraceLog();

#define __PROFILE_OP1(MODULE, NAME, CTX, X) \
  ProfileGuard __profile_guard(CTX, NAME, fmt::format("{}", X));

#define __PROFILE_OP2(MODULE, NAME, CTX, X, Y) \
  ProfileGuard __profile_guard(CTX, NAME, fmt::format("{},{}", X, Y));

#define __PROFILE_OP3(MODULE, NAME, CTX, X, Y, Z) \
  ProfileGuard __profile_guard(CTX, NAME, fmt::format("{},{},{}", X, Y, Z));

#define __PROFILE_OP4(MODULE, NAME, CTX, X, Y, Z, U) \
  ProfileGuard __profile_guard(CTX, NAME,            \
                               fmt::format("{},{},{},{}", X, Y, Z, U));

#define __PROFILE_END_OP1(MODULE, NAME, CTX, X) \
  ProfileGuard __profile_guard(CTX, NAME, fmt::format("{}", X), true);

#define __PROFILE_END_OP2(MODULE, NAME, CTX, X, Y) \
  ProfileGuard __profile_guard(CTX, NAME, fmt::format("{},{}", X, Y), true);

#define __PROFILE_END_OP3(MODULE, NAME, CTX, X, Y, Z)                       \
  ProfileGuard __profile_guard(CTX, NAME, fmt::format("{},{},{}", X, Y, Z), \
                               true);

#define __PROFILE_END_OP4(MODULE, NAME, CTX, X, Y, Z, U) \
  ProfileGuard __profile_guard(CTX, NAME,                \
                               fmt::format("{},{},{},{}", X, Y, Z, U), true);

#define __TRACE_OP1(MODULE, NAME, CTX, X)                             \
  TraceGuard __trace_guard(CTX);                                      \
  if (CTX->getTracingEnabled()) {                                     \
    const auto indent = std::string(CTX->getTracingDepth() * 2, ' '); \
    spuTraceLog()->info("{}{}.{}({})", indent, MODULE, NAME, X);      \
  }

#define __TRACE_OP2(MODULE, NAME, CTX, X, Y)                           \
  TraceGuard __trace_guard(CTX);                                       \
  if (CTX->getTracingEnabled()) {                                      \
    const auto indent = std::string(CTX->getTracingDepth() * 2, ' ');  \
    spuTraceLog()->info("{}{}.{}({},{})", indent, MODULE, NAME, X, Y); \
  }

#define __TRACE_OP3(MODULE, NAME, CTX, X, Y, Z)                              \
  TraceGuard __trace_guard(CTX);                                             \
  if (CTX->getTracingEnabled()) {                                            \
    const auto indent = std::string(CTX->getTracingDepth() * 2, ' ');        \
    spuTraceLog()->info("{}{}.{}({},{},{})", indent, MODULE, NAME, X, Y, Z); \
  }

#define __TRACE_OP4(MODULE, NAME, CTX, X, Y, Z, U)                             \
  TraceGuard __trace_guard(CTX);                                               \
  if (CTX->getTracingEnabled()) {                                              \
    const auto indent = std::string(CTX->getTracingDepth() * 2, ' ');          \
    spuTraceLog()->info("{}{}.{}({},{},{},{})", indent, MODULE, NAME, X, Y, Z, \
                        U);                                                    \
  }

// https://stackoverflow.com/questions/11761703/overloading-macro-on-number-of-arguments
//
// clang-format off
// SPU_PROFILE_OP(ctx, x)        -> SELECT(ctx, x, OP3, OP2, OP1)       -> OP1
// SPU_PROFILE_OP(ctx, x, y)     -> SELECT(ctx, x, y, OP3, OP2, OP1)    -> OP2
// SPU_PROFILE_OP(ctx, x, y, z)  -> SELECT(ctx, x, y, z, OP3, OP2, OP1) -> OP3
//
// crap!!
// SPU_PROFILE_OP(ctx, x, y, z, w)  -> SELECT(ctx, x, y, z, w, OP3, OP2, OP1) -> w
// clang-format on
#define __MACRO_SELECT_WITH_CTX(CTX, _1, _2, _3, _4, NAME, ...) NAME

#define __TRACE_OP(MODULE, NAME, ...)                                         \
  __MACRO_SELECT_WITH_CTX(__VA_ARGS__, __TRACE_OP4, __TRACE_OP3, __TRACE_OP2, \
                          __TRACE_OP1)                                        \
  (MODULE, NAME, __VA_ARGS__)

#define __PROFILE_OP(MODULE, NAME, ...)                              \
  __MACRO_SELECT_WITH_CTX(__VA_ARGS__, __PROFILE_OP4, __PROFILE_OP3, \
                          __PROFILE_OP2, __PROFILE_OP1)              \
  (MODULE, NAME, __VA_ARGS__)

#define __PROFILE_END_OP(MODULE, NAME, ...)                                  \
  __MACRO_SELECT_WITH_CTX(__VA_ARGS__, __PROFILE_END_OP4, __PROFILE_END_OP3, \
                          __PROFILE_END_OP2, __PROFILE_END_OP1)              \
  (MODULE, NAME, __VA_ARGS__)

#define __TRACE_AND_PROFILE_OP(MODULE, NAME, ...) \
  __TRACE_OP(MODULE, NAME, __VA_ARGS__)           \
  __PROFILE_OP(MODULE, NAME, __VA_ARGS__)

#define __TRACE_AND_PROFILE_END_OP(MODULE, NAME, ...) \
  __TRACE_OP(MODULE, NAME, __VA_ARGS__)               \
  __PROFILE_END_OP(MODULE, NAME, __VA_ARGS__)

}  // namespace spu

namespace std {

// helper function to print indices.
inline std::ostream& operator<<(std::ostream& os,
                                const std::vector<size_t>& indices) {
  os << fmt::format("{{{}}}", fmt::join(indices, ","));
  return os;
}

}  // namespace std
