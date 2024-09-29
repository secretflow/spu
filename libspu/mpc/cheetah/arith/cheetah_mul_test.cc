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
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah::test {

class CheetahMulTest
    : public ::testing::TestWithParam<std::tuple<size_t, size_t, bool>> {};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, CheetahMulTest,
    testing::Combine(testing::Values(32, 64, 128), testing::Values(1024, 10000),
                     testing::Values(true, false)),
    [](const testing::TestParamInfo<CheetahMulTest::ParamType>& p) {
      return fmt::format("{}x{}x{}", std::get<0>(p.param), std::get<1>(p.param),
                         std::get<2>(p.param) ? "Approx" : "Exact");
    });

TEST_P(CheetahMulTest, Basic) {
  size_t kWorldSize = 2;
  auto field = std::get<0>(GetParam());
  int64_t n = std::get<1>(GetParam());
  bool allow_approx = std::get<2>(GetParam());

  MemRef a_bits(makeType<RingTy>(SE_INVALID, field), {n});
  MemRef b_bits(makeType<RingTy>(SE_INVALID, field), {n});
  ring_rand(a_bits);
  ring_rand(b_bits);

  std::vector<MemRef> a_shr(kWorldSize);
  std::vector<MemRef> b_shr(kWorldSize);
  a_shr[0] = MemRef(makeType<RingTy>(SE_INVALID, field), {n});
  ring_rand(a_shr[0]);
  b_shr[0] = MemRef(makeType<RingTy>(SE_INVALID, field), {n});
  ring_rand(b_shr[0]);
  a_shr[1] = ring_sub(a_bits, a_shr[0]);
  b_shr[1] = ring_sub(b_bits, b_shr[0]);

  std::vector<MemRef> result(kWorldSize);
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> lctx) {
    int rank = lctx->Rank();
    // (a0 + a1) * (b0 + b1)
    // a0*b0 + a0*b1 + a1*b0 + a1*b1
    auto mul = std::make_shared<CheetahMul>(lctx, allow_approx);

    MemRef cross0;
    MemRef cross1;
    if (rank == 0) {
      cross0 = mul->MulOLE(a_shr[0], true);
      cross1 = mul->MulOLE(b_shr[0], true);
    } else {
      cross0 = mul->MulOLE(b_shr[1], false);
      cross1 = mul->MulOLE(a_shr[1], false);
    }

    result[rank] = ring_mul(a_shr[rank], b_shr[rank]);
    ring_add_(result[rank], cross0);
    ring_add_(result[rank], cross1);
  });

  auto expected = ring_mul(a_bits, b_bits);
  auto computed = ring_add(result[0], result[1]);

  const int64_t kMaxDiff = allow_approx ? 1 : 0;
  EXPECT_TRUE(ring_all_equal(expected, computed, kMaxDiff));
}

TEST_P(CheetahMulTest, BasicBinary) {
  size_t kWorldSize = 2;
  auto field = std::get<0>(GetParam());
  int64_t n = std::get<1>(GetParam());
  bool allow_approx = std::get<2>(GetParam());

  MemRef a_bits(makeType<RingTy>(SE_INVALID, field), {n});
  ring_rand_range(a_bits, 0, 1);
  MemRef b_bits(makeType<RingTy>(SE_INVALID, field), {n});
  ring_rand_range(b_bits, 0, 1);

  std::vector<MemRef> a_shr(kWorldSize);
  std::vector<MemRef> b_shr(kWorldSize);
  a_shr[0] = MemRef(makeType<RingTy>(SE_INVALID, field), {n});
  ring_rand(a_shr[0]);
  b_shr[0] = MemRef(makeType<RingTy>(SE_INVALID, field), {n});
  ring_rand(b_shr[0]);
  a_shr[1] = ring_sub(a_bits, a_shr[0]);
  b_shr[1] = ring_sub(b_bits, b_shr[0]);

  std::vector<MemRef> result(kWorldSize);
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> lctx) {
    int rank = lctx->Rank();
    // (a0 + a1) * (b0 + b1)
    // a0*b0 + a0*b1 + a1*b0 + a1*b1
    auto mul = std::make_shared<CheetahMul>(lctx, allow_approx);

    MemRef cross0;
    MemRef cross1;
    if (rank == 0) {
      cross0 = mul->MulOLE(a_shr[0], true);
      cross1 = mul->MulOLE(b_shr[0], true);
    } else {
      cross0 = mul->MulOLE(b_shr[1], false);
      cross1 = mul->MulOLE(a_shr[1], false);
    }

    result[rank] = ring_mul(a_shr[rank], b_shr[rank]);
    ring_add_(result[rank], cross0);
    ring_add_(result[rank], cross1);
  });

  auto expected = ring_mul(a_bits, b_bits);
  auto computed = ring_add(result[0], result[1]);

  const int64_t kMaxDiff = allow_approx ? 1 : 0;
  EXPECT_TRUE(ring_all_equal(expected, computed, kMaxDiff));
}

TEST_P(CheetahMulTest, MixedRingSizeMul) {
  size_t kWorldSize = 2;
  auto field = std::get<0>(GetParam());
  int64_t n = std::get<1>(GetParam());
  bool allow_approx = std::get<2>(GetParam());
  // Compute Mul on field then on field2
  size_t field2;
  if (field == 32) {
    field2 = 64;
  } else if (field == 64) {
    field2 = 128;
  } else {
    field2 = 32;
    std::swap(field, field2);
  }

  MemRef a_bits(makeType<RingTy>(SE_INVALID, field), {n});
  MemRef b_bits(makeType<RingTy>(SE_INVALID, field), {n});
  MemRef c_bits(makeType<RingTy>(SE_INVALID, field2), {n});
  MemRef d_bits(makeType<RingTy>(SE_INVALID, field2), {n});
  ring_rand(a_bits);
  ring_rand(b_bits);
  ring_rand(c_bits);
  ring_rand(d_bits);

  std::vector<MemRef> a_shr(kWorldSize);
  std::vector<MemRef> b_shr(kWorldSize);
  std::vector<MemRef> c_shr(kWorldSize);
  std::vector<MemRef> d_shr(kWorldSize);

  a_shr[0] = MemRef(makeType<RingTy>(SE_INVALID, field), {n});
  ring_rand(a_shr[0]);
  b_shr[0] = MemRef(makeType<RingTy>(SE_INVALID, field), {n});
  ring_rand(b_shr[0]);
  a_shr[1] = ring_sub(a_bits, a_shr[0]);
  b_shr[1] = ring_sub(b_bits, b_shr[0]);

  c_shr[0] = MemRef(makeType<RingTy>(SE_INVALID, field2), {n});
  ring_rand(c_shr[0]);
  d_shr[0] = MemRef(makeType<RingTy>(SE_INVALID, field2), {n});
  ring_rand(d_shr[0]);
  c_shr[1] = ring_sub(c_bits, c_shr[0]);
  d_shr[1] = ring_sub(d_bits, d_shr[0]);

  std::vector<MemRef> result(kWorldSize);
  std::vector<MemRef> result2(kWorldSize);
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> lctx) {
    int rank = lctx->Rank();
    // (a0 + a1) * (b0 + b1)
    // a0*b0 + a0*b1 + a1*b0 + a1*b1
    auto mul = std::make_shared<CheetahMul>(lctx, allow_approx);

    MemRef cross0;
    MemRef cross1;
    if (rank == 0) {
      cross0 = mul->MulOLE(a_shr[0], true);
      cross1 = mul->MulOLE(b_shr[0], true);
    } else {
      cross0 = mul->MulOLE(b_shr[1], false);
      cross1 = mul->MulOLE(a_shr[1], false);
    }

    result[rank] = ring_mul(a_shr[rank], b_shr[rank]);
    ring_add_(result[rank], cross0);
    ring_add_(result[rank], cross1);

    if (rank == 0) {
      cross0 = mul->MulOLE(c_shr[0], true);
      cross1 = mul->MulOLE(d_shr[0], true);
    } else {
      cross1 = mul->MulOLE(d_shr[1], false);
      cross0 = mul->MulOLE(c_shr[1], false);
    }

    result2[rank] = ring_mul(c_shr[rank], d_shr[rank]);
    ring_add_(result2[rank], cross0);
    ring_add_(result2[rank], cross1);
  });

  auto expected = ring_mul(a_bits, b_bits);
  auto computed = ring_add(result[0], result[1]);

  auto expected2 = ring_mul(c_bits, d_bits);
  auto computed2 = ring_add(result2[0], result2[1]);

  const int64_t kMaxDiff = allow_approx ? 1 : 0;

  EXPECT_TRUE(ring_all_equal(expected, computed, kMaxDiff));

  EXPECT_TRUE(ring_all_equal(expected2, computed2, kMaxDiff));
}

TEST_P(CheetahMulTest, MulShare) {
  size_t kWorldSize = 2;
  auto field = std::get<0>(GetParam());
  int64_t n = std::get<1>(GetParam());
  bool allow_approx = std::get<2>(GetParam());

  MemRef a_bits(makeType<RingTy>(SE_INVALID, field), {n});
  MemRef b_bits(makeType<RingTy>(SE_INVALID, field), {n});

  ring_rand(a_bits);
  ring_rand(b_bits);

  std::vector<MemRef> a_shr(kWorldSize);
  std::vector<MemRef> b_shr(kWorldSize);
  a_shr[0] = MemRef(makeType<RingTy>(SE_INVALID, field), {n});
  b_shr[0] = MemRef(makeType<RingTy>(SE_INVALID, field), {n});

  ring_rand(a_shr[0]);
  ring_rand(b_shr[0]);
  a_shr[1] = ring_sub(a_bits, a_shr[0]);
  b_shr[1] = ring_sub(b_bits, b_shr[0]);

  std::vector<MemRef> result(kWorldSize);
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> lctx) {
    int rank = lctx->Rank();
    // (a0 + a1) * (b0 + b1)
    // a0*b0 + a0*b1 + a1*b0 + a1*b1
    auto mul = std::make_shared<CheetahMul>(lctx, allow_approx);

    result[rank] = mul->MulShare(a_shr[rank], b_shr[rank], rank == 0);
  });

  auto expected = ring_mul(a_bits, b_bits);
  auto computed = ring_add(result[0], result[1]);

  const int64_t kMaxDiff = allow_approx ? 1 : 0;
  EXPECT_TRUE(ring_all_equal(expected, computed, kMaxDiff));
}

}  // namespace spu::mpc::cheetah::test
