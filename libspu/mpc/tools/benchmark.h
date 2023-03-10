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

#include <functional>

#include "benchmark/benchmark.h"
#include "yacl/link/link.h"

#include "libspu/core/bit_utils.h"
#include "libspu/core/shape_util.h"  // calcNumel
#include "libspu/mpc/api.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/object.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::bench {

struct ParamEntry {
  std::string name;
  std::string value;
  ParamEntry(std::string n, std::string v)
      : name(std::move(n)), value(std::move(v)) {}
};

std::map<uint64_t, std::string> field_name{
    {static_cast<uint64_t>(FieldType::FM32), "32"},
    {static_cast<uint64_t>(FieldType::FM64), "64"},
    {static_cast<uint64_t>(FieldType::FM128), "128"}};

using CreateComputeFn = std::function<std::unique_ptr<Object>(
    const RuntimeConfig& conf,
    const std::shared_ptr<yacl::link::Context>& lctx)>;

class BenchConfig {
 public:
  inline static std::shared_ptr<yacl::link::Context> bench_lctx = nullptr;
  inline static CreateComputeFn bench_factory = {};
  inline static uint32_t bench_npc = 0;
  inline static std::string bench_mode = "standalone";
  inline static std::string bench_parties = {};
  inline static std::vector<int64_t> bench_numel_range = {1u << 10, 1u << 20};
  inline static std::vector<int64_t> bench_shift_range = {2};
  inline static std::vector<int64_t> bench_matrix_m_range = {10};
  inline static std::vector<int64_t> bench_matrix_k_range = {10};
  inline static std::vector<int64_t> bench_field_range = {FieldType::FM64,
                                                          FieldType::FM128};
};

template <typename BenchOp>
void MPCBenchMark(benchmark::State& state) {
  std::string label = BenchOp::op_name;
  label = std::string("op_name:") + label;
  for (auto& entry : BenchOp::Paras(state)) {
    label += '/' + entry.name + ':' + entry.value;
  }
  state.SetLabel(label);

  for (auto _ : state) {
    const size_t npc = BenchConfig::bench_npc;
    const auto field = static_cast<spu::FieldType>(state.range(0));
    RuntimeConfig conf;
    conf.set_field(field);
    auto func = [&](std::shared_ptr<yacl::link::Context> lctx) {
      auto obj = BenchConfig::bench_factory(conf, lctx);
      auto* comm = obj->getState<Communicator>();

      BenchOp op(obj.get(), state);

      /* WHEN */
      if (lctx->Rank() == 0) {
        auto prev = comm->getStats();
        // state.ResumeTiming() will record start time,
        // and state.PauseTiming() uses now to substract start time.
        // If we use state to manage time, we need call
        // state.PauseTiming() in every iteration to avoid initialization
        // overhand, so we need a state.ResumeTiming() to mathch it.
        // If we call state.ResumeTiming() in this thread,
        // time is more, because we record two pairs of state operation:
        //  (this)   |------R--P---R--|
        //  >----P---|                |----->
        //  (other)  |----------------|
        //  (another)|----------------|
        // What if we call state.ResumeTiming() at the end of every thread? The
        // last Resume will make the second pair insignificant, but under this
        // condition:
        //  (this)   |------R----P--R|
        //  >----P---|         |---------->
        //  (other)  |--------R|
        //  (another)  |--------R|
        // time is less. So we use manual time.
        auto start = std::chrono::high_resolution_clock::now();
        benchmark::DoNotOptimize(op.Exec());
        auto end = std::chrono::high_resolution_clock::now();
        auto cost = comm->getStats() - prev;
        state.counters["latency"] = cost.latency;
        state.counters["comm"] = cost.comm;
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end -
                                                                      start);
        state.SetIterationTime(elapsed_seconds.count());
      } else {
        op.Exec();
        state.SetIterationTime(0);
      }
    };

    if (BenchConfig::bench_mode == "standalone") {
      utils::simulate(npc, func);
    } else {
      func(BenchConfig::bench_lctx);
    }
  }
}

class BenchOpSP {
 protected:
  ArrayRef p0;
  ArrayRef p1;
  ArrayRef s0;
  ArrayRef s1;
  Object* obj_{nullptr};
  benchmark::State& state;

 public:
  BenchOpSP(Object* obj, benchmark::State& st) : obj_(obj), state(st) {
    /* GIVEN */
    p0 = rand_p(obj, state.range(1));
    p1 = rand_p(obj, state.range(1));
    s0 = p2s(obj, p0);
    s1 = p2s(obj, p1);
  }
  static std::vector<ParamEntry> Paras(benchmark::State& st) {
    std::vector<ParamEntry> paras;
    paras.emplace_back("field_type", field_name[st.range(0)]);
    paras.emplace_back("buf_len", std::to_string(st.range(1)));
    return paras;
  }
};

class BenchAddSS : public BenchOpSP {
 public:
  using BenchOpSP::BenchOpSP;
  ArrayRef Exec() { return add_ss(obj_, s0, s1); };
  static inline std::string op_name = "add_ss";
};

class BenchMulSS : public BenchOpSP {
 public:
  using BenchOpSP::BenchOpSP;
  ArrayRef Exec() { return mul_ss(obj_, s0, s1); }
  static inline std::string op_name = "mul_ss";
};

class BenchAndSS : public BenchOpSP {
 public:
  using BenchOpSP::BenchOpSP;
  ArrayRef Exec() { return and_ss(obj_, s0, s1); }
  static inline std::string op_name = "and_ss";
};

class BenchXorSS : public BenchOpSP {
 public:
  using BenchOpSP::BenchOpSP;
  ArrayRef Exec() { return xor_ss(obj_, s0, s1); }
  static inline std::string op_name = "xor_ss";
};

class BenchAddSP : public BenchOpSP {
 public:
  using BenchOpSP::BenchOpSP;
  ArrayRef Exec() { return add_sp(obj_, s0, p1); }
  static inline std::string op_name = "add_sp";
};

class BenchMulSP : public BenchOpSP {
 public:
  using BenchOpSP::BenchOpSP;
  ArrayRef Exec() { return mul_sp(obj_, s0, p1); }
  static inline std::string op_name = "mul_sp";
};

class BenchAndSP : public BenchOpSP {
 public:
  using BenchOpSP::BenchOpSP;
  ArrayRef Exec() { return and_sp(obj_, s0, p1); }
  static inline std::string op_name = "and_sp";
};

class BenchXorSP : public BenchOpSP {
 public:
  using BenchOpSP::BenchOpSP;
  ArrayRef Exec() { return xor_sp(obj_, s0, p1); }
  static inline std::string op_name = "xor_sp";
};

class BenchNotS : public BenchOpSP {
 public:
  using BenchOpSP::BenchOpSP;
  ArrayRef Exec() { return not_s(obj_, s0); }
  static inline std::string op_name = "not_s";
};

class BenchNotP : public BenchOpSP {
 public:
  using BenchOpSP::BenchOpSP;
  ArrayRef Exec() { return not_p(obj_, p0); }
  static inline std::string op_name = "not_p";
};

class BenchOpShift : public BenchOpSP {
 public:
  using BenchOpSP::BenchOpSP;
  static std::vector<ParamEntry> Paras(benchmark::State& st) {
    std::vector<ParamEntry> paras = BenchOpSP::Paras(st);
    paras.emplace_back("shift_bit", std::to_string(st.range(2)));
    return paras;
  }
};

class BenchLShiftS : public BenchOpShift {
 public:
  using BenchOpShift::BenchOpShift;
  ArrayRef Exec() { return lshift_s(obj_, s0, state.range(2)); }
  static inline std::string op_name = "lshift_s";
};

class BenchLShiftP : public BenchOpShift {
 public:
  using BenchOpShift::BenchOpShift;
  ArrayRef Exec() { return lshift_p(obj_, p0, state.range(2)); }
  static inline std::string op_name = "lshift_p";
};

class BenchRShiftS : public BenchOpShift {
 public:
  using BenchOpShift::BenchOpShift;
  ArrayRef Exec() { return rshift_s(obj_, s0, state.range(2)); }
  static inline std::string op_name = "rshift_s";
};

class BenchRShiftP : public BenchOpShift {
 public:
  using BenchOpShift::BenchOpShift;
  ArrayRef Exec() { return rshift_p(obj_, p0, state.range(2)); }
  static inline std::string op_name = "rshift_p";
};

class BenchARShiftS : public BenchOpShift {
 public:
  using BenchOpShift::BenchOpShift;
  ArrayRef Exec() { return arshift_s(obj_, s0, state.range(2)); }
  static inline std::string op_name = "arshift_s";
};

class BenchARShiftP : public BenchOpShift {
 public:
  using BenchOpShift::BenchOpShift;
  ArrayRef Exec() { return arshift_p(obj_, p0, state.range(2)); }
  static inline std::string op_name = "arshift_p";
};

class BenchTruncS : public BenchOpSP {
 public:
  using BenchOpSP::BenchOpSP;
  ArrayRef Exec() { return trunc_s(obj_, s0, state.range(2)); }
  static inline std::string op_name = "trunc_s";
  static std::vector<ParamEntry> Paras(benchmark::State& st) {
    std::vector<ParamEntry> paras = BenchOpSP::Paras(st);
    paras.emplace_back("trunc_bit", std::to_string(st.range(2)));
    return paras;
  }
};

class BenchS2P : public BenchOpSP {
 public:
  using BenchOpSP::BenchOpSP;
  ArrayRef Exec() { return s2p(obj_, s0); }
  static inline std::string op_name = "s2p";
};

class BenchP2S : public BenchOpSP {
 public:
  using BenchOpSP::BenchOpSP;
  ArrayRef Exec() { return p2s(obj_, p0); }
  static inline std::string op_name = "p2s";
};

class BenchOpMat {
 protected:
  int64_t M{};
  int64_t K{};
  int64_t N{};
  ArrayRef p0;
  ArrayRef p1;
  ArrayRef s0;
  ArrayRef s1;
  Object* obj_{nullptr};
  benchmark::State& state;

 public:
  BenchOpMat(Object* obj, benchmark::State& st) : obj_(obj), state(st) {
    /* GIVEN */
    M = state.range(1);
    K = state.range(2);
    N = state.range(1);
    const std::vector<int64_t> shape_A{M, K};
    const std::vector<int64_t> shape_B{K, N};
    p0 = rand_p(obj_, calcNumel(shape_A));
    p1 = rand_p(obj_, calcNumel(shape_B));
    s0 = p2s(obj_, p0);
    s1 = p2s(obj_, p1);
  }
  static std::vector<ParamEntry> Paras(benchmark::State& st) {
    std::vector<ParamEntry> paras;
    paras.emplace_back("field_type", field_name[st.range(0)]);
    paras.emplace_back("matrix_size", fmt::format("{{{0}, {1}}}*{{{1}, {0}}}",
                                                  st.range(1), st.range(2)));
    return paras;
  }
};

class BenchMMulSP : public BenchOpMat {
 public:
  using BenchOpMat::BenchOpMat;
  ArrayRef Exec() { return mmul_sp(obj_, s0, p1, M, N, K); };
  static inline std::string op_name = "mmul_sp";
};

class BenchMMulSS : public BenchOpMat {
 public:
  using BenchOpMat::BenchOpMat;
  ArrayRef Exec() { return mmul_ss(obj_, s0, s1, M, N, K); };
  static inline std::string op_name = "mmul_ss";
};

}  // namespace spu::mpc::bench
