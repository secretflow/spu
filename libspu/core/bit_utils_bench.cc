// Copyright 2023 Ant Group Co., Ltd.
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

#include "libspu/core/bit_utils.h"

namespace spu {

namespace {

template <typename T>
void BMBitIntlWithPdepext(benchmark::State& state) {
  auto count = static_cast<size_t>(state.range(0));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<T> distrib(0, std::numeric_limits<T>::max());

  std::vector<T> values;
  values.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    values.push_back(distrib(gen));
  }
  while (state.KeepRunningBatch(static_cast<int64_t>(count))) {
    for (size_t i = 0; i < count; ++i) {
      benchmark::DoNotOptimize(detail::BitIntlWithPdepext(values[i], 0));
    }
  }
}
BENCHMARK_TEMPLATE(BMBitIntlWithPdepext, uint64_t)->Range(1, 1 << 20);

template <typename T>
void BMBitDeintlWithPdepext(benchmark::State& state) {
  auto count = static_cast<size_t>(state.range(0));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<T> distrib(0, std::numeric_limits<T>::max());

  std::vector<T> values;
  values.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    values.push_back(distrib(gen));
  }
  while (state.KeepRunningBatch(static_cast<int64_t>(count))) {
    for (size_t i = 0; i < count; ++i) {
      benchmark::DoNotOptimize(detail::BitDeintlWithPdepext(values[i], 0));
    }
  }
}
BENCHMARK_TEMPLATE(BMBitDeintlWithPdepext, uint64_t)->Range(1, 1 << 20);

template <typename T>
void BMBitIntl(benchmark::State& state) {
  auto count = static_cast<size_t>(state.range(0));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<T> distrib(0, std::numeric_limits<T>::max());

  std::vector<T> values;
  values.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    values.push_back(distrib(gen));
  }
  while (state.KeepRunningBatch(static_cast<int64_t>(count))) {
    for (size_t i = 0; i < count; ++i) {
      benchmark::DoNotOptimize(BitIntl(values[i], 0));
    }
  }
}
BENCHMARK_TEMPLATE(BMBitIntl, uint32_t)->Range(1, 1 << 20);
BENCHMARK_TEMPLATE(BMBitIntl, uint64_t)->Range(1, 1 << 20);
BENCHMARK_TEMPLATE(BMBitIntl, uint128_t)->Range(1, 1 << 20);

template <typename T>
void BMBitDeintl(benchmark::State& state) {
  auto count = static_cast<size_t>(state.range(0));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<T> distrib(0, std::numeric_limits<T>::max());

  std::vector<T> values;
  values.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    values.push_back(distrib(gen));
  }
  while (state.KeepRunningBatch(static_cast<int64_t>(count))) {
    for (size_t i = 0; i < count; ++i) {
      benchmark::DoNotOptimize(BitDeintl(values[i], 0));
    }
  }
}
BENCHMARK_TEMPLATE(BMBitDeintl, uint32_t)->Range(1, 1 << 20);
BENCHMARK_TEMPLATE(BMBitDeintl, uint64_t)->Range(1, 1 << 20);
BENCHMARK_TEMPLATE(BMBitDeintl, uint128_t)->Range(1, 1 << 20);

}  // namespace

}  // namespace spu
