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
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "absl/types/span.h"
#include "fmt/format.h"
#include "fmt/ostream.h"
#include "spdlog/spdlog.h"

namespace std {

// helper function to print indices.
inline std::ostream& operator<<(std::ostream& os,
                                const std::vector<size_t>& indices) {
  os << fmt::format("{{{}}}", fmt::join(indices, ","));
  return os;
}

inline std::ostream& operator<<(std::ostream& os,
                                const absl::Span<int64_t const>& indices) {
  os << fmt::format("{{{}}}", fmt::join(indices, ","));
  return os;
}

}  // namespace std

namespace spu {
namespace internal {

inline void variadicToStringImpl(std::stringstream& ss) {}

template <typename T>
void variadicToStringImpl(std::stringstream& ss, const T& t) {
  ss << t;
}

template <typename First, typename... Rest>
void variadicToStringImpl(std::stringstream& ss, const First& first,
                          const Rest&... tail) {
  ss << first << ", ";
  variadicToStringImpl(ss, tail...);
}

// convert parameter list `x, y, z` to string "(x, y, z)"
template <typename... Args>
std::string variadicToString(const Args&... args) {
  std::stringstream ss;
  variadicToStringImpl(ss, args...);
  return ss.str();
}

int64_t genActionUuid();

}  // namespace internal

/// Trace module macros
// Module divide action into groups.
//
// For each traced action, it's only recorded when:
// 1. the action belongs the module.
// 2. the tracer enables the module.
//
// |-MOD1--|-MOD2--|
// f0
// |- g0
// |  |- h0
// |  |  |- r1
// |  |  |  |- s0
// |  |  |  |  |- t0
// |  |  |  |  |- t1
// |  |  |  |- s1
// |  |- h1
// |  g1
// |  |- h2
// |  |- h3
#define TR_MOD1 0x0001  // module 1
#define TR_MOD2 0x0002  // module 2
#define TR_MOD3 0x0004  // module 3
#define TR_MOD4 0x0008  // module 4
#define TR_MOD5 0x0010  // module 5
#define TR_MOD6 0x0020  // module 6
#define TR_MODALL (TR_MOD1 | TR_MOD2 | TR_MOD3 | TR_MOD4 | TR_MOD5 | TR_MOD6)

// Trace action macros
//
// Note: action statistics could always be reconstruct from log, when recording
// is enabled, the statistics could be queried from runtime.
#define TR_LOGB 0x0100              // log action begin
#define TR_LOGE 0x0200              // log action end
#define TR_LOGM 0x0400              // log current memory usage
#define TR_REC 0x0800               // record the action
#define TR_LOG (TR_LOGB | TR_LOGE)  // log action begin & end
#define TR_LAR (TR_LOG | TR_REC)    // log and record the action

using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
using Duration = std::chrono::nanoseconds;

struct ActionRecord {
  // the uuid of this action.
  int64_t id;
  // name of the action, the name should be static allocated.
  std::string name;
  // detail of the action
  std::string detail;
  // the flag of this action.
  int64_t flag;
  // the action timing information.
  TimePoint start;
  TimePoint end;
};

class Tracer final {
  // name of this tracer
  const std::string name_;

  // current tracer mask.
  int64_t mask_;

  // the logger
  std::shared_ptr<spdlog::logger> logger_;

  // the recorded action, at ending time.
  std::vector<ActionRecord> records_;

 public:
  explicit Tracer(std::string name, int64_t mask,
                  std::shared_ptr<spdlog::logger> logger)
      : name_(std::move(name)), mask_(mask), logger_(std::move(logger)) {}

  const std::string& name() const { return name_; }

  void setMask(int64_t new_mask) { mask_ = new_mask; }
  int64_t getMask() const { return mask_; }

  /// log begin of an action.
  // @id, unique action id.
  // @flag, various attributes of the action.
  // @name, name of the action.
  // @detail, detail of the action.
  void logActionBegin(int64_t id, int64_t flag, const std::string& name,
                      const std::string& detail = "");
  void logActionEnd(int64_t id, int64_t flag, const std::string& name,
                    const std::string& detail = "");

  void addRecord(ActionRecord&& rec) {
    if ((rec.flag & mask_ & TR_MODALL) != 0 && (mask_ & TR_REC) != 0) {
      records_.push_back(std::move(rec));
    }
  }
  const std::vector<ActionRecord>& getRecords() const { return records_; }
  void clearRecords() { records_.clear(); }
};

class TraceAction final {
  std::shared_ptr<Tracer> const tracer_;
  int64_t const flag_;
  int64_t const mask_;

  // the uuid of this action.
  int64_t id_;

  // name of the action.
  std::string name_;

  // detail of the action.
  std::string detail_;

  // the action timing information.
  TimePoint start_;
  TimePoint end_;

  int64_t saved_tracer_mask_;

  template <typename... Args>
  void begin(const std::string& name, Args&&... args) {
    name_ = name;
    start_ = std::chrono::high_resolution_clock::now();

    if ((flag_ & TR_LOGB) != 0) {
      // request for logging begin of the acion
      detail_ = internal::variadicToString(std::forward<Args>(args)...);
      tracer_->logActionBegin(id_, flag_, name_, detail_);
    }

    // set new mask to the tracer.
    saved_tracer_mask_ = tracer_->getMask();
    tracer_->setMask(saved_tracer_mask_ & mask_);
  }

  void end() {
    // recover mask of the tracer.
    tracer_->setMask(saved_tracer_mask_);

    //
    end_ = std::chrono::high_resolution_clock::now();

    if ((flag_ & TR_LOGE) != 0) {
      // request for logging end of the acion
      tracer_->logActionEnd(id_, flag_, name_, detail_);
    }

    if ((flag_ & TR_REC) != 0) {
      // request for recording this action.
      tracer_->addRecord(
          ActionRecord{id_, name_, std::move(detail_), flag_, start_, end_});
    }
  }

 public:
  /// define a trace action.
  //
  // The action will be traced by the tracer according to the action's flag and
  // tracer's mask. For example:
  //
  //   Tracer tracer = ...;
  //   TraceAction ta(tracer, TR_MOD2 | TR_LOG, ~TR_MOD2, "func", ...);
  //
  // Which means:
  //   flag = TR_MOD2|TR_LOG, means it belongs MOD2, request for logging.
  //   mask = ~TR_MOD2,       means disable further TR_MOD2 tracing.
  template <typename... Args>
  explicit TraceAction(std::shared_ptr<Tracer> tracer, int64_t flag,
                       int64_t mask, const std::string& name, Args&&... args)
      : tracer_(std::move(tracer)), flag_(flag), mask_(mask) {
    id_ = internal::genActionUuid();
    begin(name, std::forward<Args>(args)...);
  }

  ~TraceAction() { end(); }
};

// global setting
void initTrace(int64_t tr_flag,
               std::shared_ptr<spdlog::logger> tr_logger = nullptr);

std::shared_ptr<Tracer> getTracer(const std::string& name);

void registerTracer(std::shared_ptr<Tracer> tracer);

#define SPU_ENABLE_TRACE

// TODO: support per-context trace.
// #define GET_CTX_NAME(CTX) "CTX:0"
#define GET_CTX_NAME(CTX) ((CTX)->name())

#ifdef SPU_ENABLE_TRACE

// Why add `##` to __VA_ARGS__, please see
// https://stackoverflow.com/questions/5891221/variadic-macros-with-zero-arguments
#define SPU_TRACE_ACTION(TR_NAME, FLAG, MASK, NAME, ...)           \
  TraceAction __trace_action(getTracer(TR_NAME), FLAG, MASK, NAME, \
                             ##__VA_ARGS__);

#else

#define SPU_TRACE_ACTION(TR_NAME, FLAG, MASK, NAME, ...) (void)NAME;

#endif

/////////////////////////////////////////////////////////////
/// Helper macros for modules.
/////////////////////////////////////////////////////////////

#define TR_HLO TR_MOD1
#define TR_HAL TR_MOD2
#define TR_MPC TR_MOD3

// trace a hal layer dispatch
#define SPU_TRACE_HLO_DISP(CTX, ...)                                     \
  SPU_TRACE_ACTION(GET_CTX_NAME(CTX), (TR_HLO | TR_LOG), (~0), __func__, \
                   ##__VA_ARGS__)

// trace a hal layer leaf
#define SPU_TRACE_HLO_LEAF(CTX, ...)                                          \
  SPU_TRACE_ACTION(GET_CTX_NAME(CTX), (TR_HLO | TR_LAR), (~TR_HLO), __func__, \
                   ##__VA_ARGS__)

// trace a hal layer dispatch
#define SPU_TRACE_HAL_DISP(CTX, ...)                                     \
  SPU_TRACE_ACTION(GET_CTX_NAME(CTX), (TR_HAL | TR_LOG), (~0), __func__, \
                   ##__VA_ARGS__)

// trace a hal layer leaf
#define SPU_TRACE_HAL_LEAF(CTX, ...)                                          \
  SPU_TRACE_ACTION(GET_CTX_NAME(CTX), (TR_HAL | TR_LAR), (~TR_HAL), __func__, \
                   ##__VA_ARGS__)

// trace a mpc layer dispatch
#define SPU_TRACE_MPC_DISP(CTX, ...)                                      \
  SPU_TRACE_ACTION(GET_CTX_NAME(CTX), (TR_MPC | TR_LOG), (~0), kBindName, \
                   ##__VA_ARGS__)

// trace a mpc layer leaf
#define SPU_TRACE_MPC_LEAF(CTX, ...)                                           \
  SPU_TRACE_ACTION(GET_CTX_NAME(CTX), (TR_MPC | TR_LAR), (~TR_MPC), kBindName, \
                   ##__VA_ARGS__)

// TODO: remove this.
class MemProfilingGuard {
 private:
  int indent_ = 0;
  std::string_view module_;
  std::string_view name_;
  float start_peak_ = 0.0F;
  bool enable_ = false;

 public:
  void enable(int i, std::string_view m, std::string_view n);
  ~MemProfilingGuard();
};

// Debug purpose only.
class LogActionGuard final {
  int64_t id_ = 0;

  void end() {
    SPDLOG_INFO("{} end.", id_);
    id_ = 0;
  }

 public:
  template <typename... Args>
  explicit LogActionGuard(Args&&... args) {
    id_ = internal::genActionUuid();

    SPDLOG_INFO("{} {} begins: {}", __func__, id_,
                internal::variadicToString(std::forward<Args>(args)...));
  }

  ~LogActionGuard() {
    if (id_ != 0) {
      end();
    }
  }
};

}  // namespace spu
