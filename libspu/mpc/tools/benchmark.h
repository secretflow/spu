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
  inline static std::vector<int64_t> bench_numel_range =
      benchmark::CreateRange(10, 1000, /*multi=*/10);
  inline static std::vector<int64_t> bench_shift_range =
      benchmark::CreateRange(2, 8, /*multi=*/2);
  inline static std::vector<int64_t> bench_matrix_m_range =
      benchmark::CreateDenseRange(3, 9, /*step=*/3);
  inline static std::vector<int64_t> bench_matrix_k_range =
      benchmark::CreateDenseRange(4, 16, /*step=*/4);
  inline static std::vector<int64_t> bench_field_range = {
      FieldType::FM32, FieldType::FM64, FieldType::FM128};
};

template <typename BenchOp>
void MPCBenchMark(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
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
        state.ResumeTiming();

        benchmark::DoNotOptimize(op.Exec());

        state.PauseTiming();
        auto cost = comm->getStats() - prev;
        state.counters["latency"] = cost.latency;
        state.counters["comm"] = cost.comm;
        state.ResumeTiming();
      } else {
        op.Exec();

        state.ResumeTiming();
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
};

class BenchAddSS : public BenchOpSP {
 public:
  using BenchOpSP::BenchOpSP;
  ArrayRef Exec() { return add_ss(obj_, s0, s1); };
};

class BenchMulSS : public BenchOpSP {
 public:
  using BenchOpSP::BenchOpSP;
  ArrayRef Exec() { return mul_ss(obj_, s0, s1); }
};

class BenchAndSS : public BenchOpSP {
 public:
  using BenchOpSP::BenchOpSP;
  ArrayRef Exec() { return and_ss(obj_, s0, s1); }
};

class BenchXorSS : public BenchOpSP {
 public:
  using BenchOpSP::BenchOpSP;
  ArrayRef Exec() { return xor_ss(obj_, s0, s1); }
};

class BenchAddSP : public BenchOpSP {
 public:
  using BenchOpSP::BenchOpSP;
  ArrayRef Exec() { return add_sp(obj_, s0, p1); }
};

class BenchMulSP : public BenchOpSP {
 public:
  using BenchOpSP::BenchOpSP;
  ArrayRef Exec() { return mul_sp(obj_, s0, p1); }
};

class BenchAndSP : public BenchOpSP {
 public:
  using BenchOpSP::BenchOpSP;
  ArrayRef Exec() { return and_sp(obj_, s0, p1); }
};

class BenchXorSP : public BenchOpSP {
 public:
  using BenchOpSP::BenchOpSP;
  ArrayRef Exec() { return xor_sp(obj_, s0, p1); }
};

class BenchNotS : public BenchOpSP {
 public:
  using BenchOpSP::BenchOpSP;
  ArrayRef Exec() { return not_s(obj_, s0); }
};

class BenchNotP : public BenchOpSP {
 public:
  using BenchOpSP::BenchOpSP;
  ArrayRef Exec() { return not_p(obj_, p0); }
};

class BenchLShiftS : public BenchOpSP {
 public:
  using BenchOpSP::BenchOpSP;
  ArrayRef Exec() { return lshift_s(obj_, s0, state.range(2)); }
};

class BenchLShiftP : public BenchOpSP {
 public:
  using BenchOpSP::BenchOpSP;
  ArrayRef Exec() { return lshift_p(obj_, p0, state.range(2)); }
};

class BenchRShiftS : public BenchOpSP {
 public:
  using BenchOpSP::BenchOpSP;
  ArrayRef Exec() { return rshift_s(obj_, s0, state.range(2)); }
};

class BenchRShiftP : public BenchOpSP {
 public:
  using BenchOpSP::BenchOpSP;
  ArrayRef Exec() { return rshift_p(obj_, p0, state.range(2)); }
};

class BenchARShiftS : public BenchOpSP {
 public:
  using BenchOpSP::BenchOpSP;
  ArrayRef Exec() { return arshift_s(obj_, s0, state.range(2)); }
};

class BenchARShiftP : public BenchOpSP {
 public:
  using BenchOpSP::BenchOpSP;
  ArrayRef Exec() { return arshift_p(obj_, p0, state.range(2)); }
};

class BenchTruncS : public BenchOpSP {
 public:
  using BenchOpSP::BenchOpSP;
  ArrayRef Exec() { return trunc_s(obj_, s0, state.range(2)); }
};

class BenchS2P : public BenchOpSP {
 public:
  using BenchOpSP::BenchOpSP;
  ArrayRef Exec() { return s2p(obj_, s0); }
};

class BenchP2S : public BenchOpSP {
 public:
  using BenchOpSP::BenchOpSP;
  ArrayRef Exec() { return p2s(obj_, p0); }
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
};

class BenchMMulSP : public BenchOpMat {
 public:
  using BenchOpMat::BenchOpMat;
  ArrayRef Exec() { return mmul_sp(obj_, s0, p1, M, N, K); };
};

class BenchMMulSS : public BenchOpMat {
 public:
  using BenchOpMat::BenchOpMat;
  ArrayRef Exec() { return mmul_ss(obj_, s0, s1, M, N, K); };
};

}  // namespace spu::mpc::bench
