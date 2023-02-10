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

#include "libspu/mpc/cheetah/arith/cheetah_mul.h"

#include "gtest/gtest.h"

#include "libspu/core/type_util.h"
#include "libspu/core/xt_helper.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah::test {

class CheetahMulTest
    : public ::testing::TestWithParam<std::tuple<FieldType, size_t>> {};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, CheetahMulTest,
    testing::Combine(testing::Values(FieldType::FM32, FieldType::FM64,
                                     FieldType::FM128),
                     testing::Values(10, 128, 8192)),
    [](const testing::TestParamInfo<CheetahMulTest::ParamType>& p) {
      return fmt::format("{}x{}", std::get<0>(p.param), std::get<1>(p.param));
    });

TEST_P(CheetahMulTest, Basic) {
  size_t kWorldSize = 2;
  auto field = std::get<0>(GetParam());
  auto n = std::get<1>(GetParam());

  std::vector<ArrayRef> arr(kWorldSize);
  // NOTE(juhou): now Cheetah supports strided ArrayRef
  for (size_t stride : {1, 2, 3}) {
    for (size_t i = 0; i < kWorldSize; ++i) {
      arr[i] = ring_rand(field, n);
      arr[i] = arr[i].slice(1, n, stride);
    }

    std::vector<ArrayRef> result(kWorldSize);
    utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> lctx) {
      int rank = lctx->Rank();
      auto mul = std::make_shared<CheetahMul>(lctx);
      result[rank] = mul->MulOLE(arr[rank], rank == 0);
    });

    auto expected = ring_mul(arr[0], arr[1]);
    auto computed = ring_add(result[0], result[1]);

    const int64_t kMaxDiff = 1;
    DISPATCH_ALL_FIELDS(field, "_", [&]() {
      auto e = ArrayView<ring2k_t>(expected);
      auto c = ArrayView<ring2k_t>(computed);

      for (auto idx = 0; idx < expected.numel(); idx++) {
        auto err = e[idx] > c[idx] ? e[idx] - c[idx] : c[idx] - e[idx];
        EXPECT_LE(err, kMaxDiff);
      }
    });
  }
}

#if 0
TEST_P(CheetahMulTest, Fork) {
  size_t kWorldSize = 2;
  auto field = std::get<0>(GetParam());
  auto n = std::get<1>(GetParam());

  std::vector<ArrayRef> arr(kWorldSize);
  for (size_t i = 0; i < kWorldSize; ++i) {
    arr[i] = ring_rand(field, n);
  }

  std::vector<ArrayRef> result(kWorldSize);

  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> lctx) {
    int rank = lctx->Rank();
    auto mul = std::make_shared<CheetahMul>(lctx);
    // warm up using n=1
    result[rank] = mul->MulOLE(arr[rank].slice(0, 1), rank == 0);

    auto mul_f = mul->Fork();
    result[rank] = mul_f.MulOLE(arr[rank].slice(1, n), rank == 1);
  });

  auto expected = ring_mul(arr[0], arr[1]);
  auto computed = ring_add(result[0], result[1]);

  const int64_t kMaxDiff = 1;
  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    auto e = ArrayView<ring2k_t>(expected);
    auto c = ArrayView<ring2k_t>(computed);

    for (auto idx = 1; idx < expected.numel(); idx++) {
      auto err =
          e[idx] > c[idx - 1] ? e[idx] - c[idx - 1] : c[idx - 1] - e[idx];
      EXPECT_LE(err, kMaxDiff);
    }
  });
}
#endif

}  // namespace spu::mpc::cheetah::test
