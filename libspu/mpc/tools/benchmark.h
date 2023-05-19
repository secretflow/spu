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
#include "libspu/core/context.h"
#include "libspu/core/shape_util.h"  // calcNumel
#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/api.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::bench {

struct ParamEntry {
  std::string name;
  std::string value;
  ParamEntry(std::string n, std::string v)
      : name(std::move(n)), value(std::move(v)) {}
};

using CreateComputeFn = std::function<std::unique_ptr<SPUContext>(
    const RuntimeConfig& conf,
    const std::shared_ptr<yacl::link::Context>& lctx)>;

class BenchConfig {
 public:
  inline static std::shared_ptr<yacl::link::Context> bench_lctx = nullptr;
  inline static CreateComputeFn bench_factory = {};
  inline static uint32_t bench_npc = 0;
  inline static std::string bench_mode = "standalone";
  inline static std::string bench_parties = {};
  inline static std::vector<int64_t> bench_numel_range = {1U << 10, 1U << 20};
  inline static std::vector<int64_t> bench_shift_range = {2};
  inline static std::vector<int64_t> bench_matrix_range = {10, 100};
  inline static std::vector<int64_t> bench_field_range = {FieldType::FM64,
                                                          FieldType::FM128};
};

template <typename OpData, typename ArgsInfo>
void MPCBenchMark(benchmark::State& state) {
  state.SetLabel(ArgsInfo(OpData::op_name, state).Label());

  for (auto _ : state) {
    const size_t npc = BenchConfig::bench_npc;
    const auto field = static_cast<spu::FieldType>(state.range(0));
    RuntimeConfig conf;
    conf.set_field(field);
    auto func = [&](std::shared_ptr<yacl::link::Context> lctx) {
      auto obj = BenchConfig::bench_factory(conf, lctx);
      if (!obj->hasKernel(OpData::op_name)) {
        return;
      }

      auto* comm = obj->getState<Communicator>();

      OpData op(obj.get(), state);

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

#define MPC_BENCH_DEFINE(CLASS, DATA, OP, ...)     \
  class CLASS : public DATA {                      \
   public:                                         \
    using DATA::DATA;                              \
    static inline std::string op_name = #OP;       \
    Value Exec() { return OP(obj_, __VA_ARGS__); } \
  };

template <size_t P = 0, size_t S = 0, size_t A = 0, size_t B = 0, size_t MP = 0,
          size_t MS = 0, size_t MA = 0, size_t MB = 0, size_t B1 = 0>
class OpData {
 protected:
  SPUContext* obj_{nullptr};
  benchmark::State& state;
  std::array<Value, P> ps;
  std::array<Value, S> ss;
  std::array<Value, A> as;
  std::array<Value, B> bs;
  std::array<Value, MP> mps;
  std::array<Value, MS> mss;
  std::array<Value, MA> mas;
  std::array<Value, MB> mbs;
  std::array<Value, B1> b1s;

 public:
  OpData(SPUContext* obj, benchmark::State& st) : obj_(obj), state(st) {
    for (auto& p : ps) {
      p = rand_p(obj_, Shape{state.range(1)});
    }
    for (auto& s : ss) {
      s = p2s(obj_, rand_p(obj, Shape{state.range(1)}));
    }
    for (auto& a : as) {
      a = p2a(obj_, rand_p(obj_, Shape{state.range(1)}));
    }
    for (auto& b : bs) {
      b = p2b(obj_, rand_p(obj_, Shape{state.range(1)}));
    }
    auto matrix_size = calcNumel({state.range(1), state.range(1)});
    for (auto& mp : mps) {
      mp = rand_p(obj_, Shape{matrix_size});
    }
    for (auto& ms : mss) {
      ms = p2s(obj_, rand_p(obj_, Shape{matrix_size}));
    }
    for (auto& ma : mas) {
      ma = p2a(obj_, rand_p(obj_, Shape{matrix_size}));
    }
    for (auto& mb : mbs) {
      mb = p2b(obj_, rand_p(obj_, Shape{matrix_size}));
    }
    for (auto& b1 : b1s) {
      b1 = p2b(obj_, rand_p(obj_, Shape{state.range(1)}));
      b1 = lshift_b(obj_, b1,
                    SizeOf(static_cast<FieldType>(state.range(0))) * 8 - 1);
      b1 = rshift_b(obj_, b1,
                    SizeOf(static_cast<FieldType>(state.range(0))) * 8 - 1);
    }
  }
  virtual ~OpData() = default;
};

using OpDataBasic = OpData<>;
using OpData1P = OpData<1>;
using OpData1S = OpData<0, 1>;
using OpData2S = OpData<0, 2>;
using OpData1S1P = OpData<1, 1>;
using OpData1A = OpData<0, 0, 1>;
using OpData2A = OpData<0, 0, 2>;
using OpData1A1P = OpData<1, 0, 1>;
using OpData1B = OpData<0, 0, 0, 1>;
using OpData2B = OpData<0, 0, 0, 2>;
using OpData1B1P = OpData<1, 0, 0, 1>;
using OpData2MS = OpData<0, 0, 0, 0, 0, 2>;
using OpData2MA = OpData<0, 0, 0, 0, 0, 0, 2>;
using OpData1MS1MP = OpData<0, 0, 0, 0, 1, 1>;
using OpData1MA1MP = OpData<0, 0, 0, 0, 1, 0, 1>;
using OpData1A1B1 = OpData<0, 0, 1, 0, 0, 0, 0, 0, 1>;

MPC_BENCH_DEFINE(BenchAddSS, OpData2S, add_ss, ss[0], ss[1])
MPC_BENCH_DEFINE(BenchMulSS, OpData2S, mul_ss, ss[0], ss[1])
MPC_BENCH_DEFINE(BenchAndSS, OpData2S, and_ss, ss[0], ss[1])
MPC_BENCH_DEFINE(BenchXorSS, OpData2S, xor_ss, ss[0], ss[1])
MPC_BENCH_DEFINE(BenchAddSP, OpData1S1P, add_sp, ss[0], ps[0])
MPC_BENCH_DEFINE(BenchMulSP, OpData1S1P, mul_sp, ss[0], ps[0])
MPC_BENCH_DEFINE(BenchAndSP, OpData1S1P, and_sp, ss[0], ps[0])
MPC_BENCH_DEFINE(BenchXorSP, OpData1S1P, xor_sp, ss[0], ps[0])
MPC_BENCH_DEFINE(BenchNotS, OpData1S, not_s, ss[0])
MPC_BENCH_DEFINE(BenchNotP, OpData1P, not_p, ps[0])
MPC_BENCH_DEFINE(BenchLShiftS, OpData1S, lshift_s, ss[0], state.range(2))
MPC_BENCH_DEFINE(BenchLShiftP, OpData1P, lshift_p, ps[0], state.range(2))
MPC_BENCH_DEFINE(BenchRShiftS, OpData1S, rshift_s, ss[0], state.range(2))
MPC_BENCH_DEFINE(BenchRShiftP, OpData1P, rshift_p, ps[0], state.range(2))
MPC_BENCH_DEFINE(BenchARShiftS, OpData1S, arshift_s, ss[0], state.range(2))
MPC_BENCH_DEFINE(BenchARShiftP, OpData1P, arshift_p, ps[0], state.range(2))
MPC_BENCH_DEFINE(BenchTruncS, OpData1S, trunc_s, ss[0], state.range(2))
MPC_BENCH_DEFINE(BenchS2P, OpData1S, s2p, ss[0])
MPC_BENCH_DEFINE(BenchP2S, OpData1P, p2s, ps[0])
MPC_BENCH_DEFINE(BenchMMulSP, OpData1MS1MP, mmul_sp, mss[0], mps[0],
                 state.range(1), state.range(1), state.range(1))
MPC_BENCH_DEFINE(BenchMMulSS, OpData2MS, mmul_ss, mss[0], mss[1],
                 state.range(1), state.range(1), state.range(1))

MPC_BENCH_DEFINE(BenchRandA, OpDataBasic, rand_a, Shape{state.range(1)})
MPC_BENCH_DEFINE(BenchRandB, OpDataBasic, rand_b, Shape{state.range(1)})
MPC_BENCH_DEFINE(BenchP2A, OpData1P, p2a, ps[0])
MPC_BENCH_DEFINE(BenchA2P, OpData1A, a2p, as[0])
MPC_BENCH_DEFINE(BenchMsbA2b, OpData1A, msb_a2b, as[0])
MPC_BENCH_DEFINE(BenchNotA, OpData1A, not_a, as[0])
MPC_BENCH_DEFINE(BenchAddAP, OpData1A1P, add_ap, as[0], ps[0])
MPC_BENCH_DEFINE(BenchMulAP, OpData1A1P, mul_ap, as[0], ps[0])
MPC_BENCH_DEFINE(BenchAddAA, OpData2A, add_aa, as[0], as[1])
MPC_BENCH_DEFINE(BenchMulAA, OpData2A, mul_aa, as[0], as[1])
MPC_BENCH_DEFINE(BenchMulA1B, OpData1A1B1, mul_a1b, as[0], b1s[0])
MPC_BENCH_DEFINE(BenchLShiftA, OpData1A, lshift_a, as[0], state.range(2))
MPC_BENCH_DEFINE(BenchTruncA, OpData1A, trunc_a, as[0], state.range(2))
MPC_BENCH_DEFINE(BenchMMulAP, OpData1MA1MP, mmul_ap, mas[0], mps[0],
                 state.range(1), state.range(1), state.range(1))
MPC_BENCH_DEFINE(BenchMMulAA, OpData2MA, mmul_aa, mas[0], mas[1],
                 state.range(1), state.range(1), state.range(1))
MPC_BENCH_DEFINE(BenchB2P, OpData1B, b2p, bs[0])
MPC_BENCH_DEFINE(BenchP2B, OpData1P, p2b, ps[0])
MPC_BENCH_DEFINE(BenchA2B, OpData1A, a2b, as[0])
MPC_BENCH_DEFINE(BenchB2A, OpData1B, b2a, bs[0])
MPC_BENCH_DEFINE(BenchAndBP, OpData1B1P, and_bp, bs[0], ps[0])
MPC_BENCH_DEFINE(BenchAndBB, OpData2B, and_bb, bs[0], bs[1])
MPC_BENCH_DEFINE(BenchXorBP, OpData1B1P, xor_bp, bs[0], ps[0])
MPC_BENCH_DEFINE(BenchXorBB, OpData2B, xor_bb, bs[0], bs[1])
MPC_BENCH_DEFINE(BenchLShiftB, OpData1B, lshift_b, bs[0], state.range(2))
MPC_BENCH_DEFINE(BenchRShiftB, OpData1B, rshift_b, bs[0], state.range(2))
MPC_BENCH_DEFINE(BenchARShiftB, OpData1B, arshift_b, bs[0], state.range(2))
MPC_BENCH_DEFINE(BenchBitRevB, OpData1B, bitrev_b, bs[0], 0,
                 SizeOf(static_cast<FieldType>(state.range(0))))
MPC_BENCH_DEFINE(BenchBitIntlB, OpData1B, bitintl_b, bs[0], 0)
MPC_BENCH_DEFINE(BenchBitDentlB, OpData1B, bitdeintl_b, bs[0], 0)
MPC_BENCH_DEFINE(BenchAddBB, OpData2B, add_bb, bs[0], bs[1])

}  // namespace spu::mpc::bench
