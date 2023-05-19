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
#include <mutex>
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

/// Design of tracing system.
//
// Each action has two attributes:
// - flag: describes the expected behavior of the action, i.e. (MOD1|LOG|REC)
//         means the action belongs to MOD1, expected to be logged and recorded.
// - mask: used to mask out the current flag.
//
// The behavior is defined by two attributes: the action's flag and the current
// context's flag. i.e. action's flag is (LOG | REC) and current context's flag
// is LOG, the final flag is (LOG | REC) & LOG = LOG.
//
// Module divide action into groups. For example, let's consider the following
// call stack: f0 calls g0, g0 calls h0, etc.
//
//   |--- M1 ---|--- M2 ---|
//   f0->g0->h0->r0->s0->t0
//
// (f0, g0, h0) belongs to module 1, (r0, s0, t0) belongs to module 2.
//
// The call stack looks like this:
//
// |   M1  |  M2      | flag,       mask,   cur           | action
// --------|----------|-----------------------------------|-------
// f0                 | M1|LOG,     ~0,     M1|M2|LOG|REC | M1|LOG
// |- g0              | M1|REG,     ~M1,    M1|M2|LOG|REC | M1|LOG|REC
// |  |- h0           | M1|LOG|REC, ~0,     M2|LOG|REC    | -
// |  |  |- r0        | M2|LOG,     ~0,     M2|LOG|REC    | M2|LOG
// |  |  |  |- s0     | M2,         ~M2,    M2|LOG|REC    | M2
// |  |  |  |  |- t0  | M2|REC,     ~0,     LOG|REC       | -
// |  |  |  |  |- t1  | M2|REC,     ~0,     LOG|REC       | -
// |  |  |  |- s1     | M2|REC,     ~0,     M2|LOG|REC    | M2|REC
// |  |- h1           | M1|LOG,     ~0,     M1|M2|LOG|REC | M1|LOG
//
// The current flag is a thread-local (per thread) state.

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

/////////////////////////////////////////////////////////////
/// Helper macros for modules.
/////////////////////////////////////////////////////////////

#define TR_HLO TR_MOD1
#define TR_HAL TR_MOD2
#define TR_MPC TR_MOD3

using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
using Duration = std::chrono::nanoseconds;

struct ActionRecord final {
  // the uuid of this action.
  int64_t id;
  // name of the action, the name should be static allocated.
  std::string name;
  // detail of the action.
  std::string detail;
  // the flag of the action.
  int64_t flag;
  // the action timing information.
  TimePoint start;
  TimePoint end;
};

class ProfState final {
  // the recorded action, at ending time.
  std::vector<ActionRecord> records_;
  // the records_ mutex.
  std::mutex mutex_;

 public:
  void addRecord(ActionRecord&& rec) {
    std::unique_lock lk(mutex_);
    records_.push_back(std::move(rec));
  }
  const std::vector<ActionRecord>& getRecords() const { return records_; }
  void clearRecords() { records_.clear(); }
};

// A tracer is a 'single thread'
class Tracer final {
  // current tracer's flag.
  int64_t flag_;
  // current depth
  int64_t depth_ = 0;
  // trace from multi-thread shares a profile state.
  std::shared_ptr<ProfState> prof_state_ = nullptr;

 public:
  explicit Tracer(int64_t flag)
      : flag_(flag), prof_state_(std::make_shared<ProfState>()) {}

  void setFlag(int64_t new_flag) { flag_ = new_flag; }
  int64_t getFlag() const { return flag_; }

  int64_t getDepth() const { return depth_; }

  void incDepth() { depth_++; }

  void decDepth() { depth_--; }

  const std::shared_ptr<ProfState>& getProfState() { return prof_state_; }

  // TODO: drop these two functions.
  void logActionBegin(int64_t id, const std::string& mod,
                      const std::string& name,
                      const std::string& detail = "") const;

  void logActionEnd(int64_t id, const std::string& mod, const std::string& name,
                    const std::string& detail = "") const;
};

class TraceAction final {
  // The tracer.
  std::shared_ptr<Tracer> const tracer_;

  // The static expected behavior of this action.
  int64_t const flag_;

  // The mask to suppress current tracer's flag.
  int64_t const mask_;

  // the uuid of this action.
  int64_t id_;

  // the module of this action.
  std::string mod_;

  // name of the action.
  std::string name_;

  // detail of the action.
  std::string detail_;

  // the action timing information.
  TimePoint start_;
  TimePoint end_;

  int64_t saved_tracer_flag_;

  template <typename... Args>
  void begin(Args&&... args) {
    start_ = std::chrono::high_resolution_clock::now();

    const auto flag = flag_ & tracer_->getFlag();
    if ((flag & TR_LOGB) != 0) {
      detail_ = internal::variadicToString(std::forward<Args>(args)...);
      tracer_->logActionBegin(id_, mod_, name_, detail_);
      tracer_->incDepth();
    }

    // set new flag to the tracer.
    saved_tracer_flag_ = tracer_->getFlag();
    tracer_->setFlag(saved_tracer_flag_ & mask_);
  }

  void end() {
    // recover mask of the tracer.
    tracer_->setFlag(saved_tracer_flag_);

    //
    end_ = std::chrono::high_resolution_clock::now();

    const auto flag = flag_ & tracer_->getFlag();
    if ((flag & TR_LOGE) != 0) {
      tracer_->decDepth();
      tracer_->logActionEnd(id_, mod_, name_, detail_);
    }
    if ((flag & TR_REC) != 0 && (flag & TR_MODALL) != 0) {
      tracer_->getProfState()->addRecord(
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
  explicit TraceAction(
      std::shared_ptr<Tracer> tracer,  //
      int64_t flag,      // the static expected behaviour flag of action.
      int64_t mask,      // the suppress mask of the action.
      std::string name,  // name of this action.
      Args&&... args)
      : tracer_(std::move(tracer)),
        flag_(flag),
        mask_(mask),
        name_(std::move(name)) {
    id_ = internal::genActionUuid();
    if (flag_ & TR_MPC) {
      mod_ = "mpc";
    } else if (flag_ & TR_HAL) {
      mod_ = "hal";
    } else {
      mod_ = "hlo";
    }
    begin(std::forward<Args>(args)...);
  }

  ~TraceAction() { end(); }
};

// global setting
void initTrace(const std::string& ctx_id, int64_t tr_flag,
               const std::shared_ptr<spdlog::logger>& tr_logger = nullptr);

int64_t getGlobalTraceFlag(const std::string& id);

// get the trace state by current (virtual thread) id, if there is no
// corresponding Tracer found, try to clone a state from the Tracer
// corresponding to the parent id.
std::shared_ptr<Tracer> getTracer(const std::string& id,
                                  const std::string& pid);

/// The helper macros
#define SPU_ENABLE_TRACE

// TODO: support per-context trace.
#define GET_TRACER(CTX) getTracer((CTX)->id(), (CTX)->pid())

#ifdef SPU_ENABLE_TRACE

// Why add `##` to __VA_ARGS__, please see
// https://stackoverflow.com/questions/5891221/variadic-macros-with-zero-arguments
#define SPU_TRACE_ACTION(TRACER, FLAG, MASK, NAME, ...) \
  TraceAction __trace_action(TRACER, FLAG, MASK, NAME, ##__VA_ARGS__);

#else

#define SPU_TRACE_ACTION(TRACER, FLAG, MASK, NAME, ...) (void)NAME;

#endif

// trace an hlo layer dispatch
#define SPU_TRACE_HLO_DISP(CTX, ...)                                   \
  SPU_TRACE_ACTION(GET_TRACER(CTX), (TR_HLO | TR_LOG), (~0), __func__, \
                   ##__VA_ARGS__)

// trace an hlo layer leaf
#define SPU_TRACE_HLO_LEAF(CTX, ...)                                        \
  SPU_TRACE_ACTION(GET_TRACER(CTX), (TR_HLO | TR_LAR), (~TR_HLO), __func__, \
                   ##__VA_ARGS__)

// trace an hal layer dispatch
#define SPU_TRACE_HAL_DISP(CTX, ...)                                   \
  SPU_TRACE_ACTION(GET_TRACER(CTX), (TR_HAL | TR_LOG), (~0), __func__, \
                   ##__VA_ARGS__)

// trace an hal layer leaf
#define SPU_TRACE_HAL_LEAF(CTX, ...)                                        \
  SPU_TRACE_ACTION(GET_TRACER(CTX), (TR_HAL | TR_LAR), (~TR_HAL), __func__, \
                   ##__VA_ARGS__)

// trace an mpc layer dispatch
#define SPU_TRACE_MPC_DISP(CTX, ...)                                   \
  SPU_TRACE_ACTION(GET_TRACER(CTX), (TR_MPC | TR_LOG), (~0), __func__, \
                   ##__VA_ARGS__)

// trace an mpc layer leaf
#define SPU_TRACE_MPC_LEAF(CTX, ...)                                        \
  SPU_TRACE_ACTION(GET_TRACER(CTX), (TR_MPC | TR_LAR), (~TR_MPC), __func__, \
                   ##__VA_ARGS__)

// Debug purpose only.
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
