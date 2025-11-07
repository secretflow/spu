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

#include <random>

#include "gtest/gtest.h"
#include "yacl/utils/elapsed_timer.h"

#include "libspu/core/type_util.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah::test {

class CheetahMulTest
    : public ::testing::TestWithParam<std::tuple<FieldType, size_t, bool>> {};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, CheetahMulTest,
    testing::Combine(testing::Values(FieldType::FM8, FieldType::FM16,
                                     FieldType::FM32, FieldType::FM64,
                                     FieldType::FM128),
                     testing::Values(1024, 10000),
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

  auto a_bits = ring_rand(field, {n});
  auto b_bits = ring_rand(field, {n});

  std::vector<NdArrayRef> a_shr(kWorldSize);
  std::vector<NdArrayRef> b_shr(kWorldSize);
  a_shr[0] = ring_rand(field, {n});
  b_shr[0] = ring_rand(field, {n});
  a_shr[1] = ring_sub(a_bits, a_shr[0]);
  b_shr[1] = ring_sub(b_bits, b_shr[0]);

  std::vector<NdArrayRef> result(kWorldSize);
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> lctx) {
    int rank = lctx->Rank();
    // (a0 + a1) * (b0 + b1)
    // a0*b0 + a0*b1 + a1*b0 + a1*b1
    auto mul = std::make_shared<CheetahMul>(lctx, allow_approx);

    NdArrayRef cross0, cross1;
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

namespace {
// just a copy of ring_rand_range (without the type check)
template <typename T>
NdArrayRef ring_rand_small_range(FieldType field, const Shape& shape, T min,
                                 T max) {
  using S = std::make_signed_t<T>;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<S> dis(min, max);

  NdArrayRef x(makeType<RingTy>(field), shape);
  auto numel = x.numel();

  DISPATCH_ALL_FIELDS(field, [&]() {
    auto iter = x.begin();
    for (auto idx = 0; idx < numel; ++idx, ++iter) {
      iter.getScalarValue<ring2k_t>() = static_cast<ring2k_t>(dis(gen));
    }
  });

  return x;
}
}  // namespace

TEST_P(CheetahMulTest, BasicBinary) {
  size_t kWorldSize = 2;
  auto field = std::get<0>(GetParam());
  int64_t n = std::get<1>(GetParam());
  bool allow_approx = std::get<2>(GetParam());

  NdArrayRef a_bits = ring_rand_small_range(field, {n}, 0, 1);
  NdArrayRef b_bits = ring_rand_small_range(field, {n}, 0, 1);

  std::vector<NdArrayRef> a_shr(kWorldSize);
  std::vector<NdArrayRef> b_shr(kWorldSize);
  a_shr[0] = ring_rand(field, {n});
  b_shr[0] = ring_rand(field, {n});
  a_shr[1] = ring_sub(a_bits, a_shr[0]);
  b_shr[1] = ring_sub(b_bits, b_shr[0]);

  std::vector<NdArrayRef> result(kWorldSize);
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> lctx) {
    int rank = lctx->Rank();
    // (a0 + a1) * (b0 + b1)
    // a0*b0 + a0*b1 + a1*b0 + a1*b1
    auto mul = std::make_shared<CheetahMul>(lctx, allow_approx);

    NdArrayRef cross0, cross1;
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
  FieldType field2;
  if (field == FM32) {
    field2 = FM64;
  } else if (field == FM64) {
    field2 = FM128;
  } else {
    field2 = FM32;
    std::swap(field, field2);
  }

  auto a_bits = ring_rand(field, {n});
  auto b_bits = ring_rand(field, {n});
  auto c_bits = ring_rand(field2, {n});
  auto d_bits = ring_rand(field2, {n});

  std::vector<NdArrayRef> a_shr(kWorldSize);
  std::vector<NdArrayRef> b_shr(kWorldSize);
  std::vector<NdArrayRef> c_shr(kWorldSize);
  std::vector<NdArrayRef> d_shr(kWorldSize);

  a_shr[0] = ring_rand(field, {n});
  b_shr[0] = ring_rand(field, {n});
  a_shr[1] = ring_sub(a_bits, a_shr[0]);
  b_shr[1] = ring_sub(b_bits, b_shr[0]);

  c_shr[0] = ring_rand(field2, {n});
  d_shr[0] = ring_rand(field2, {n});
  c_shr[1] = ring_sub(c_bits, c_shr[0]);
  d_shr[1] = ring_sub(d_bits, d_shr[0]);

  std::vector<NdArrayRef> result(kWorldSize);
  std::vector<NdArrayRef> result2(kWorldSize);
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> lctx) {
    int rank = lctx->Rank();
    // (a0 + a1) * (b0 + b1)
    // a0*b0 + a0*b1 + a1*b0 + a1*b1
    auto mul = std::make_shared<CheetahMul>(lctx, allow_approx);

    NdArrayRef cross0, cross1;
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

  auto a_bits = ring_rand(field, {n});
  auto b_bits = ring_rand(field, {n});

  std::vector<NdArrayRef> a_shr(kWorldSize);
  std::vector<NdArrayRef> b_shr(kWorldSize);
  a_shr[0] = ring_rand(field, {n});
  b_shr[0] = ring_rand(field, {n});
  a_shr[1] = ring_sub(a_bits, a_shr[0]);
  b_shr[1] = ring_sub(b_bits, b_shr[0]);

  std::vector<NdArrayRef> result(kWorldSize);
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

class CheetahGeneralMulTest
    : public ::testing::TestWithParam<std::tuple<size_t, int64_t, bool>> {};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, CheetahGeneralMulTest,
    testing::Combine(testing::Values(8, 16, 32, 64, 128,  // full ring
                                     7, 13, 23, 48),
                     testing::Values(1024, 10000),
                     testing::Values(true, false)),
    [](const testing::TestParamInfo<CheetahGeneralMulTest::ParamType>& p) {
      return fmt::format("{}x{}x{}", std::get<0>(p.param), std::get<1>(p.param),
                         std::get<2>(p.param) ? "Approx" : "Exact");
    });

TEST_P(CheetahGeneralMulTest, Work) {
  size_t kWorldSize = 2;
  auto bw = std::get<0>(GetParam());
  auto n = std::get<1>(GetParam());
  bool allow_approx = std::get<2>(GetParam());

  auto field = FixGetProperFiled(bw);

  auto a_bits = ring_rand(field, {n});
  auto b_bits = ring_rand(field, {n});

  std::vector<NdArrayRef> a_shr(kWorldSize);
  std::vector<NdArrayRef> b_shr(kWorldSize);
  a_shr[0] = ring_rand(field, {n});
  b_shr[0] = ring_rand(field, {n});
  a_shr[1] = ring_sub(a_bits, a_shr[0]);
  b_shr[1] = ring_sub(b_bits, b_shr[0]);

  ring_reduce_(a_bits, bw);
  ring_reduce_(b_bits, bw);
  ring_reduce_(a_shr[0], bw);
  ring_reduce_(b_shr[0], bw);
  ring_reduce_(a_shr[1], bw);
  ring_reduce_(b_shr[1], bw);

  SPU_ENFORCE(
      ring_all_equal(ring_reduce(ring_add(a_shr[0], a_shr[1]), bw), a_bits));
  SPU_ENFORCE(
      ring_all_equal(ring_reduce(ring_add(b_shr[0], b_shr[1]), bw), b_bits));

  std::vector<NdArrayRef> result(kWorldSize);
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> lctx) {
    int rank = lctx->Rank();
    // (a0 + a1) * (b0 + b1)
    // a0*b0 + a0*b1 + a1*b0 + a1*b1
    auto mul = std::make_shared<CheetahMul>(lctx, allow_approx);
    // call this can reduce cost.
    mul->LazyInitKeys(field, bw);

    size_t b0 = lctx->GetStats()->sent_bytes;
    size_t r0 = lctx->GetStats()->sent_actions;
    yacl::ElapsedTimer pack_timer;

    NdArrayRef cross0;
    NdArrayRef cross1;
    if (rank == 0) {
      cross0 = mul->MulOLE(a_shr[0], true, bw);
      cross1 = mul->MulOLE(b_shr[0], true, bw);
    } else {
      cross0 = mul->MulOLE(b_shr[1], false, bw);
      cross1 = mul->MulOLE(a_shr[1], false, bw);
    }

    double pack_time = pack_timer.CountMs() * 1.0;
    size_t b1 = lctx->GetStats()->sent_bytes;
    size_t r1 = lctx->GetStats()->sent_actions;

    std::string exact_str = allow_approx ? "approx" : "exact";

    SPDLOG_INFO(
        "Rank {}, {} samples, OLE mul {} bits, {}, sent {} bits per element. "
        "Actions total {}, elapsed total time: {} ms.",
        rank, n, bw, exact_str, (b1 - b0) * 8. / n, (r1 - r0) * 1.0, pack_time);

    result[rank] = ring_mul(a_shr[rank], b_shr[rank]);
    ring_add_(result[rank], cross0);
    ring_add_(result[rank], cross1);
  });

  auto expected = ring_mul(a_bits, b_bits);
  ring_reduce_(expected, bw);

  auto computed = ring_add(result[0], result[1]);
  ring_reduce_(computed, bw);

  const int64_t kMaxDiff = allow_approx ? 1 : 0;
  EXPECT_TRUE(ring_all_equal_val(expected, computed, false, kMaxDiff));
}

}  // namespace spu::mpc::cheetah::test
