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

#include <functional>

#include "benchmark/benchmark.h"
#include "yasl/link/link.h"

#include "spu/core/shape_util.h"  // calcNumel
#include "spu/mpc/interfaces.h"
#include "spu/mpc/object.h"
#include "spu/mpc/util/communicator.h"
#include "spu/mpc/util/ring_ops.h"

namespace spu::mpc::bench {

void BenchmarkPrint(uint32_t rank, std::vector<std::string>& parties,
                    uint32_t numel, uint32_t shiftbit) {}

using CreateComputeFn = std::function<std::unique_ptr<Object>(
    const std::shared_ptr<yasl::link::Context>& lctx)>;

class ComputeBench : public benchmark::Fixture {
 public:
  static std::shared_ptr<yasl::link::Context> bench_lctx;
  static uint32_t bench_numel;
  static uint32_t bench_shiftbit;
  static CreateComputeFn bench_factory;
};

std::shared_ptr<yasl::link::Context> ComputeBench::bench_lctx = nullptr;
uint32_t ComputeBench::bench_numel = 7;
uint32_t ComputeBench::bench_shiftbit = 2;
CreateComputeFn ComputeBench::bench_factory = {};

/*
 * Benchmark Defines
 */

#define SPU_BM_SECTION_START(COMM) \
  auto prev = COMM->getStats();    \
  state.ResumeTiming();

#define SPU_BM_SECTION_END(COMM)            \
  state.PauseTiming();                      \
  auto cost = COMM->getStats() - prev;      \
  state.counters["latency"] = cost.latency; \
  state.counters["comm"] = cost.comm;       \
  state.ResumeTiming();

#define SPU_BM_SECTION(COMM, CODE) \
  SPU_BM_SECTION_START(COMM)       \
  CODE;                            \
  SPU_BM_SECTION_END(COMM)

#define SPU_BM_DEFINE_F(BaseClass, Method) BENCHMARK_DEFINE_F(BaseClass, Method)

#define SPU_BM_DEFINE_BINARY_OP_SS(OP)                                \
  SPU_BM_DEFINE_F(ComputeBench, OP##_ss)                              \
  (benchmark::State & state) {                                        \
    for (auto _ : state) {                                            \
      state.PauseTiming();                                            \
      auto obj = bench_factory(bench_lctx);                           \
      auto comm = obj->getState<Communicator>();                      \
      const auto field = static_cast<spu::FieldType>(state.range(0)); \
      {                                                               \
        /* GIVEN */                                                   \
        auto p0 = rand_p(obj.get(), field, bench_numel);              \
        auto p1 = rand_p(obj.get(), field, bench_numel);              \
        auto s0 = p2s(obj.get(), p0);                                 \
        auto s1 = p2s(obj.get(), p1);                                 \
                                                                      \
        /* WHEN */                                                    \
        SPU_BM_SECTION(comm, OP##_ss(obj.get(), s0, s1))              \
      }                                                               \
    }                                                                 \
  }

#define SPU_BM_DEFINE_BINARY_OP_SP(OP)                                \
  SPU_BM_DEFINE_F(ComputeBench, OP##_sp)                              \
  (benchmark::State & state) {                                        \
    for (auto _ : state) {                                            \
      state.PauseTiming();                                            \
      const auto field = static_cast<spu::FieldType>(state.range(0)); \
      {                                                               \
        auto obj = bench_factory(bench_lctx);                         \
        auto comm = obj->getState<Communicator>();                    \
                                                                      \
        /* GIVEN */                                                   \
        auto p0 = rand_p(obj.get(), field, bench_numel);              \
        auto p1 = rand_p(obj.get(), field, bench_numel);              \
        auto s0 = p2s(obj.get(), p0);                                 \
                                                                      \
        /* WHEN */                                                    \
        SPU_BM_SECTION(comm, OP##_sp(obj.get(), s0, p1))              \
      }                                                               \
    }                                                                 \
  }

#define SPU_BM_DEFINE_BINARY_OP(OP) \
  SPU_BM_DEFINE_BINARY_OP_SS(OP)    \
  SPU_BM_DEFINE_BINARY_OP_SP(OP)

SPU_BM_DEFINE_BINARY_OP(add)
SPU_BM_DEFINE_BINARY_OP(mul)
SPU_BM_DEFINE_BINARY_OP(and)
SPU_BM_DEFINE_BINARY_OP(xor)

#define SPU_BM_DEFINE_UNARY_OP_S(OP)                                  \
  SPU_BM_DEFINE_F(ComputeBench, OP##_s)                               \
  (benchmark::State & state) {                                        \
    for (auto _ : state) {                                            \
      state.PauseTiming();                                            \
      const auto field = static_cast<spu::FieldType>(state.range(0)); \
      {                                                               \
        auto obj = bench_factory(bench_lctx);                         \
        auto comm = obj->getState<Communicator>();                    \
                                                                      \
        /* GIVEN */                                                   \
        auto p0 = rand_p(obj.get(), field, bench_numel);              \
        auto s0 = p2s(obj.get(), p0);                                 \
                                                                      \
        /* WHEN */                                                    \
        SPU_BM_SECTION(comm, OP##_s(obj.get(), s0));                  \
      }                                                               \
    }                                                                 \
  }

#define SPU_BM_DEFINE_UNARY_OP_P(OP)                                  \
  SPU_BM_DEFINE_F(ComputeBench, OP##_p)                               \
  (benchmark::State & state) {                                        \
    for (auto _ : state) {                                            \
      state.PauseTiming();                                            \
      const auto field = static_cast<spu::FieldType>(state.range(0)); \
      {                                                               \
        auto obj = bench_factory(bench_lctx);                         \
        auto comm = obj->getState<Communicator>();                    \
                                                                      \
        /* GIVEN */                                                   \
        auto p0 = rand_p(obj.get(), field, bench_numel);              \
                                                                      \
        /* WHEN */                                                    \
        SPU_BM_SECTION(comm, OP##_p(obj.get(), p0);)                  \
      }                                                               \
    }                                                                 \
  }

#define SPU_BM_DEFINE_UNARY_OP(OP) \
  SPU_BM_DEFINE_UNARY_OP_S(OP)     \
  SPU_BM_DEFINE_UNARY_OP_P(OP)

SPU_BM_DEFINE_UNARY_OP(not )

#define SPU_BM_DEFINE_UNARY_OP_WITH_BIT_S(OP, BIT)                    \
  SPU_BM_DEFINE_F(ComputeBench, OP##_s)                               \
  (benchmark::State & state) {                                        \
    for (auto _ : state) {                                            \
      state.PauseTiming();                                            \
      const auto field = static_cast<spu::FieldType>(state.range(0)); \
      {                                                               \
        auto obj = bench_factory(bench_lctx);                         \
        auto comm = obj->getState<Communicator>();                    \
                                                                      \
        /* GIVEN */                                                   \
        auto p0 = rand_p(obj.get(), field, bench_numel);              \
        auto s0 = p2s(obj.get(), p0);                                 \
                                                                      \
        /* WHEN */                                                    \
        SPU_BM_SECTION(comm, OP##_s(obj.get(), s0, BIT));             \
      }                                                               \
    }                                                                 \
  }

#define SPU_BM_DEFINE_UNARY_OP_WITH_BIT_P(OP, BIT)                    \
  SPU_BM_DEFINE_F(ComputeBench, OP##_p)                               \
  (benchmark::State & state) {                                        \
    for (auto _ : state) {                                            \
      state.PauseTiming();                                            \
      const auto field = static_cast<spu::FieldType>(state.range(0)); \
      {                                                               \
        auto obj = bench_factory(bench_lctx);                         \
        auto comm = obj->getState<Communicator>();                    \
                                                                      \
        /* GIVEN */                                                   \
        auto p0 = rand_p(obj.get(), field, bench_numel);              \
                                                                      \
        /* WHEN */                                                    \
        SPU_BM_SECTION(comm, OP##_p(obj.get(), p0, BIT));             \
      }                                                               \
    }                                                                 \
  }

#define SPU_BM_DEFINE_UNARY_OP_WITH_BIT(OP)             \
  SPU_BM_DEFINE_UNARY_OP_WITH_BIT_S(OP, bench_shiftbit) \
  SPU_BM_DEFINE_UNARY_OP_WITH_BIT_P(OP, bench_shiftbit)

SPU_BM_DEFINE_UNARY_OP_WITH_BIT(lshift)
SPU_BM_DEFINE_UNARY_OP_WITH_BIT(rshift)
SPU_BM_DEFINE_UNARY_OP_WITH_BIT(arshift)

SPU_BM_DEFINE_F(ComputeBench, truncpr_s)
(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    const auto field = static_cast<spu::FieldType>(state.range(0));

    auto p0 = ring_rand_range(field, bench_numel, /*min*/ 0,
                              /*max*/ 10000);
    {
      auto obj = bench_factory(bench_lctx);
      auto* comm = obj->getState<Communicator>();

      const size_t bits = 2;
      auto s0 = p2s(obj.get(), p0);

      SPU_BM_SECTION(comm, truncpr_s(obj.get(), s0, bits));
    }
  }
}

SPU_BM_DEFINE_F(ComputeBench, mmul_ss)
(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    const auto field = static_cast<spu::FieldType>(state.range(0));

    const int64_t M = 3;
    const int64_t K = 4;
    const int64_t N = 3;
    const std::vector<int64_t> shape_A{M, K};
    const std::vector<int64_t> shape_B{K, N};

    {
      auto obj = bench_factory(bench_lctx);
      auto* comm = obj->getState<Communicator>();

      /* GIVEN */
      auto p0 = rand_p(obj.get(), field, calcNumel(shape_A));
      auto p1 = rand_p(obj.get(), field, calcNumel(shape_B));
      auto s0 = p2s(obj.get(), p0);
      auto s1 = p2s(obj.get(), p1);

      /* WHEN */
      SPU_BM_SECTION(comm, mmul_ss(obj.get(), s0, s1, M, N, K));
    }
  }
}

SPU_BM_DEFINE_F(ComputeBench, mmul_sp)
(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    const auto field = static_cast<spu::FieldType>(state.range(0));

    const int64_t M = 3;
    const int64_t K = 4;
    const int64_t N = 3;
    const std::vector<int64_t> shape_A{M, K};
    const std::vector<int64_t> shape_B{K, N};

    {
      auto obj = bench_factory(bench_lctx);
      auto* comm = obj->getState<Communicator>();

      /* GIVEN */
      auto p0 = rand_p(obj.get(), field, calcNumel(shape_A));
      auto p1 = rand_p(obj.get(), field, calcNumel(shape_B));
      auto s0 = p2s(obj.get(), p0);

      /* WHEN */
      SPU_BM_SECTION(comm, mmul_sp(obj.get(), s0, p1, M, N, K));
    }
  }
}

SPU_BM_DEFINE_F(ComputeBench, p2s)
(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    const auto field = static_cast<spu::FieldType>(state.range(0));

    {
      auto obj = bench_factory(bench_lctx);
      auto* comm = obj->getState<Communicator>();

      /* GIVEN */
      auto p0 = rand_p(obj.get(), field, bench_numel);

      /* WHEN */
      SPU_BM_SECTION(comm, p2s(obj.get(), p0));
    }
  }
}

SPU_BM_DEFINE_F(ComputeBench, s2p)
(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    const auto field = static_cast<spu::FieldType>(state.range(0));

    {
      auto obj = bench_factory(bench_lctx);
      auto* comm = obj->getState<Communicator>();

      /* GIVEN */
      auto p0 = rand_p(obj.get(), field, bench_numel);
      auto s0 = p2s(obj.get(), p0);

      /* WHEN */
      SPU_BM_SECTION(comm, s2p(obj.get(), s0));
    }
  }
}

/*
 * Benchmark Registers
 */

#define SPU_BM_REGISTER_BINARY_OP(OP, Arguments) \
  SPU_BM_REGISTER_OP(OP##_ss, Arguments);        \
  SPU_BM_REGISTER_OP(OP##_sp, Arguments);

#define SPU_BM_REGISTER_UNARY_OP(OP, Arguments) \
  SPU_BM_REGISTER_OP(OP##_s, Arguments);        \
  SPU_BM_REGISTER_OP(OP##_p, Arguments);

#define SPU_BM_REGISTER_OP(OP, Arguments) \
  BENCHMARK_REGISTER_F(ComputeBench, OP)->Apply(Arguments);

#define SPU_BM_PROTOCOL_REGISTER(Arguments)    \
  SPU_BM_REGISTER_BINARY_OP(add, Arguments)    \
  SPU_BM_REGISTER_BINARY_OP(mul, Arguments)    \
  SPU_BM_REGISTER_BINARY_OP(and, Arguments)    \
  SPU_BM_REGISTER_BINARY_OP(xor, Arguments)    \
  SPU_BM_REGISTER_UNARY_OP(not, Arguments)     \
  SPU_BM_REGISTER_UNARY_OP(lshift, Arguments)  \
  SPU_BM_REGISTER_UNARY_OP(rshift, Arguments)  \
  SPU_BM_REGISTER_UNARY_OP(arshift, Arguments) \
  SPU_BM_REGISTER_OP(truncpr_s, Arguments)     \
  SPU_BM_REGISTER_OP(mmul_ss, Arguments)       \
  SPU_BM_REGISTER_OP(mmul_sp, Arguments)       \
  SPU_BM_REGISTER_OP(p2s, Arguments)           \
  SPU_BM_REGISTER_OP(s2p, Arguments)

}  // namespace spu::mpc::bench
