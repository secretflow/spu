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

#include <iostream>
#include <random>

#include "benchmark/benchmark.h"

#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::utils {

static ArrayRef makeRandomArray(FieldType field, size_t numel, size_t stride) {
  const Type ty = makeType<RingTy>(field);
  const size_t buf_size = SizeOf(field) * numel * stride;
  auto buf = std::make_shared<yacl::Buffer>(buf_size);
  const int64_t offset = 0;
  return ArrayRef(buf, ty, numel, stride, offset);
}

static void makeUnaryArgs(benchmark::internal::Benchmark* b) {
  b->ArgsProduct({
      benchmark::CreateRange(8, 9182, /*multi=*/8),   // numel
      benchmark::CreateDenseRange(1, 2, /*step=*/1),  // stride
      {FM32, FM64, FM128},                            // field
  });
}

static void BM_RingAdd(benchmark::State& state) {
  const int64_t numel = state.range(0);
  const int64_t stride = state.range(1);
  const auto field = static_cast<spu::FieldType>(state.range(2));

  const ArrayRef x = makeRandomArray(field, numel, stride);
  const ArrayRef y = makeRandomArray(field, numel, stride);

  for (auto _ : state) {
    ring_add(x, y);
  }
}

static void BM_RingAdd_(benchmark::State& state) {  // NOLINT
  const int64_t numel = state.range(0);
  const int64_t stride = state.range(1);
  const auto field = static_cast<spu::FieldType>(state.range(2));

  const ArrayRef y = makeRandomArray(field, numel, stride);
  ArrayRef x = makeRandomArray(field, numel, stride);

  for (auto _ : state) {
    ring_add_(x, y);
  }
}

BENCHMARK(BM_RingAdd)->Apply(makeUnaryArgs);
BENCHMARK(BM_RingAdd_)->Apply(makeUnaryArgs);

}  // namespace spu::mpc::utils

BENCHMARK_MAIN();
