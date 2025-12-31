// Copyright 2024 Ant Group Co., Ltd.
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

#include <random>

#include "benchmark/benchmark.h"
#include "yacl/base/aligned_vector.h"

#include "libspu/mpc/cheetah/ot/matrix_transpose.h"

namespace {

yacl::UninitAlignedVector<uint128_t> GenerateRandomMatrixU128(size_t m,
                                                              size_t n) {
  yacl::UninitAlignedVector<uint128_t> mat(m * n);

  std::random_device rd;
  std::seed_seq seed({rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()});
  auto random = std::mt19937(seed);

  for (size_t i = 0; i < m * n; i++) {
    mat[i] = yacl::MakeUint128(random(), random());
  }

  return mat;
}

yacl::UninitAlignedVector<uint64_t> GenerateRandomMatrixU64(size_t m,
                                                            size_t n) {
  yacl::UninitAlignedVector<uint64_t> mat(m * n);

  std::default_random_engine rdv;
  std::uniform_int_distribution<uint64_t> uniform(0, -1);
  std::generate_n(mat.begin(), m * n, [&]() -> uint64_t {
    return static_cast<uint64_t>(uniform(rdv));
  });

  return mat;
}

yacl::UninitAlignedVector<uint32_t> GenerateRandomMatrixU32(size_t m,
                                                            size_t n) {
  yacl::UninitAlignedVector<uint32_t> mat(m * n);

  std::default_random_engine rdv;
  std::uniform_int_distribution<uint32_t> uniform(0, -1);
  std::generate_n(mat.begin(), m * n, [&]() -> uint32_t {
    return static_cast<uint32_t>(uniform(rdv));
  });

  return mat;
}

}  // namespace

[[maybe_unused]] static void BM_naive_transpose_128(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();

    // get matrix m,n
    size_t m = state.range(0);
    size_t n = state.range(1);

    const auto inp = GenerateRandomMatrixU128(m, n);
    yacl::UninitAlignedVector<uint128_t> oup(m * n);
    state.ResumeTiming();

    spu::mpc::cheetah::naive_transpose(inp.data(), oup.data(), m, n);
  }
}

[[maybe_unused]] static void BM_naive_transpose_64(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();

    // get matrix m,n
    size_t m = state.range(0);
    size_t n = state.range(1);

    const auto inp = GenerateRandomMatrixU64(m, n);
    yacl::UninitAlignedVector<uint64_t> oup(m * n);
    state.ResumeTiming();

    spu::mpc::cheetah::naive_transpose(inp.data(), oup.data(), m, n);
  }
}

[[maybe_unused]] static void BM_naive_transpose_32(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();

    // get matrix m,n
    size_t m = state.range(0);
    size_t n = state.range(1);

    const auto inp = GenerateRandomMatrixU32(m, n);
    yacl::UninitAlignedVector<uint32_t> oup(m * n);
    state.ResumeTiming();

    spu::mpc::cheetah::naive_transpose(inp.data(), oup.data(), m, n);
  }
}

[[maybe_unused]] static void BM_cache_friendly_transpose_128(
    benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();

    // get matrix m,n
    size_t m = state.range(0);
    size_t n = state.range(1);

    const auto inp = GenerateRandomMatrixU128(m, n);
    yacl::UninitAlignedVector<uint128_t> oup(m * n);
    state.ResumeTiming();

    spu::mpc::cheetah::cache_friendly_transpose(inp.data(), oup.data(), m, n);
  }
}

[[maybe_unused]] static void BM_cache_friendly_transpose_64(
    benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();

    // get matrix m,n
    size_t m = state.range(0);
    size_t n = state.range(1);

    const auto inp = GenerateRandomMatrixU64(m, n);
    yacl::UninitAlignedVector<uint64_t> oup(m * n);
    state.ResumeTiming();

    spu::mpc::cheetah::cache_friendly_transpose(inp.data(), oup.data(), m, n);
  }
}

[[maybe_unused]] static void BM_cache_friendly_transpose_32(
    benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();

    // get matrix m,n
    size_t m = state.range(0);
    size_t n = state.range(1);

    const auto inp = GenerateRandomMatrixU32(m, n);
    yacl::UninitAlignedVector<uint32_t> oup(m * n);
    state.ResumeTiming();

    spu::mpc::cheetah::cache_friendly_transpose(inp.data(), oup.data(), m, n);
  }
}

[[maybe_unused]] static void BM_sse_transpose_128(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();

    // get matrix m,n
    size_t m = state.range(0);
    size_t n = state.range(1);

    const auto inp = GenerateRandomMatrixU128(m, n);
    yacl::UninitAlignedVector<uint128_t> oup(m * n);
    state.ResumeTiming();

    spu::mpc::cheetah::sse_transpose(inp.data(), oup.data(), m, n);
  }
}

[[maybe_unused]] static void BM_sse_transpose_64(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();

    // get matrix m,n
    size_t m = state.range(0);
    size_t n = state.range(1);

    const auto inp = GenerateRandomMatrixU64(m, n);
    yacl::UninitAlignedVector<uint64_t> oup(m * n);
    state.ResumeTiming();

    spu::mpc::cheetah::sse_transpose(inp.data(), oup.data(), m, n);
  }
}

[[maybe_unused]] static void BM_sse_transpose_32(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();

    // get matrix m,n
    size_t m = state.range(0);
    size_t n = state.range(1);

    const auto inp = GenerateRandomMatrixU32(m, n);
    yacl::UninitAlignedVector<uint32_t> oup(m * n);
    state.ResumeTiming();

    spu::mpc::cheetah::sse_transpose(inp.data(), oup.data(), m, n);
  }
}

// n should be multiple of 4
// we now only need bw * n matrix to transpose

// clang-format off
// BM_naive_transpose_128/10/8192                0.643 ms        0.643 ms         1127
// BM_naive_transpose_128/23/8192                 1.47 ms         1.47 ms          505
// BM_naive_transpose_128/32/8192                 2.93 ms         2.93 ms          292
// BM_naive_transpose_128/64/8192                 6.08 ms         6.08 ms          121
// BM_naive_transpose_128/128/8192                13.0 ms         13.0 ms           51

// BM_cache_friendly_transpose_128/10/8192       0.070 ms        0.070 ms        10296
// BM_cache_friendly_transpose_128/23/8192       0.141 ms        0.141 ms         4910
// BM_cache_friendly_transpose_128/32/8192       0.216 ms        0.216 ms         3013
// BM_cache_friendly_transpose_128/64/8192       0.561 ms        0.561 ms          974
// BM_cache_friendly_transpose_128/128/8192       11.4 ms         11.4 ms           65

// BM_sse_transpose_128/10/8192                  0.106 ms        0.106 ms         6539
// BM_sse_transpose_128/23/8192                  0.188 ms        0.188 ms         3837
// BM_sse_transpose_128/32/8192                  0.502 ms        0.502 ms         1334
// BM_sse_transpose_128/64/8192                   1.38 ms         1.38 ms          509
// BM_sse_transpose_128/128/8192                  13.6 ms         13.6 ms           55
// clang-format on

// U128 test
BENCHMARK(BM_naive_transpose_128)
    ->Unit(benchmark::kMillisecond)
    ->Args({10, 8192})
    ->Args({23, 8192})
    ->Args({32, 8192})  // m,n
    ->Args({64, 8192})
    ->Args({128, 8192});

BENCHMARK(BM_cache_friendly_transpose_128)
    ->Unit(benchmark::kMillisecond)
    ->Args({10, 8192})
    ->Args({23, 8192})
    ->Args({32, 8192})  // m,n
    ->Args({64, 8192})
    ->Args({128, 8192});

BENCHMARK(BM_sse_transpose_128)
    ->Unit(benchmark::kMillisecond)
    ->Args({10, 8192})
    ->Args({23, 8192})
    ->Args({32, 8192})  // m,n
    ->Args({64, 8192})
    ->Args({128, 8192});

// clang-format off
// BM_naive_transpose_64/10/8192                0.336 ms        0.336 ms         2086
// BM_naive_transpose_64/24/8192                0.773 ms        0.773 ms          889
// BM_naive_transpose_64/32/8192                 1.14 ms         1.14 ms          611
// BM_naive_transpose_64/64/8192                 3.45 ms         3.45 ms          210
// BM_naive_transpose_64/128/8192                7.23 ms         7.23 ms           93
// 
// BM_cache_friendly_transpose_64/10/8192       0.046 ms        0.046 ms        15644
// BM_cache_friendly_transpose_64/24/8192       0.128 ms        0.128 ms         5402
// BM_cache_friendly_transpose_64/32/8192       0.183 ms        0.183 ms         3654
// BM_cache_friendly_transpose_64/64/8192       0.796 ms        0.794 ms         1557
// BM_cache_friendly_transpose_64/128/8192       7.54 ms         7.47 ms          103
// 
// BM_sse_transpose_64/10/8192                  0.065 ms        0.054 ms        12394
// BM_sse_transpose_64/24/8192                  0.099 ms        0.099 ms         7146
// BM_sse_transpose_64/32/8192                  0.146 ms        0.146 ms         4600
// BM_sse_transpose_64/64/8192                  0.753 ms        0.753 ms          929
// BM_sse_transpose_64/128/8192                  5.68 ms         5.68 ms          134
// clang-format on

// U64 test
BENCHMARK(BM_naive_transpose_64)
    ->Unit(benchmark::kMillisecond)
    ->Args({10, 8192})
    ->Args({24, 8192})
    ->Args({32, 8192})  // m,n
    ->Args({64, 8192})
    ->Args({128, 8192});

BENCHMARK(BM_cache_friendly_transpose_64)
    ->Unit(benchmark::kMillisecond)
    ->Args({10, 8192})
    ->Args({24, 8192})
    ->Args({32, 8192})  // m,n
    ->Args({64, 8192})
    ->Args({128, 8192});

BENCHMARK(BM_sse_transpose_64)
    ->Unit(benchmark::kMillisecond)
    ->Args({10, 8192})
    ->Args({24, 8192})
    ->Args({32, 8192})  // m,n
    ->Args({64, 8192})
    ->Args({128, 8192});

// clang-format off
// BM_naive_transpose_32/10/8192                0.161 ms        0.161 ms         4327
// BM_naive_transpose_32/24/8192                0.505 ms        0.505 ms         1000
// BM_naive_transpose_32/32/8192                0.698 ms        0.698 ms          755
// BM_naive_transpose_32/64/8192                 1.33 ms         1.33 ms          534
// BM_naive_transpose_32/128/8192                4.57 ms         4.57 ms          152
// 
// BM_cache_friendly_transpose_32/10/8192       0.031 ms        0.031 ms        23880
// BM_cache_friendly_transpose_32/24/8192       0.093 ms        0.093 ms         7384
// BM_cache_friendly_transpose_32/32/8192       0.158 ms        0.158 ms         4687
// BM_cache_friendly_transpose_32/64/8192       0.564 ms        0.562 ms         1677
// BM_cache_friendly_transpose_32/128/8192       4.20 ms         4.14 ms          164
// 
// BM_sse_transpose_32/10/8192                  0.030 ms        0.030 ms        22263
// BM_sse_transpose_32/24/8192                  0.066 ms        0.066 ms        10926
// BM_sse_transpose_32/32/8192                  0.095 ms        0.095 ms         7662
// BM_sse_transpose_32/64/8192                  0.156 ms        0.156 ms         4858
// BM_sse_transpose_32/128/8192                  2.62 ms         2.62 ms          247
// clang-format on

// U32 test
BENCHMARK(BM_naive_transpose_32)
    ->Unit(benchmark::kMillisecond)
    ->Args({10, 8192})
    ->Args({24, 8192})
    ->Args({32, 8192})  // m,n
    ->Args({64, 8192})
    ->Args({128, 8192});

BENCHMARK(BM_cache_friendly_transpose_32)
    ->Unit(benchmark::kMillisecond)
    ->Args({10, 8192})
    ->Args({24, 8192})
    ->Args({32, 8192})  // m,n
    ->Args({64, 8192})
    ->Args({128, 8192});

BENCHMARK(BM_sse_transpose_32)
    ->Unit(benchmark::kMillisecond)
    ->Args({10, 8192})
    ->Args({24, 8192})
    ->Args({32, 8192})  // m,n
    ->Args({64, 8192})
    ->Args({128, 8192});
