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

#include "gtest/gtest.h"
#include "yacl/link/link.h"

#include "libspu/core/type_util.h"
#include "libspu/mpc/spdz2k/beaver/beaver_tfp.h"
#include "libspu/mpc/spdz2k/beaver/beaver_tinyot.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::spdz2k {

class BeaverTest
    : public ::testing::TestWithParam<std::tuple<
          std::pair<std::function<std::unique_ptr<Beaver>(
                        const std::shared_ptr<yacl::link::Context>& lctx)>,
                    std::string>,
          size_t, FieldType, long, size_t, size_t>> {
 public:
  using Pair = typename Beaver::Pair;
  using PairPair = typename Beaver::Pair_Pair;
  using TriplePair = typename Beaver::Triple_Pair;
};

INSTANTIATE_TEST_SUITE_P(
    BeaverTestSuite, BeaverTest,
    testing::Values(
        std::tuple{std::make_pair(
                       [](const std::shared_ptr<yacl::link::Context>& lctx) {
                         return std::make_unique<BeaverTfpUnsafe>(lctx);
                       },
                       "BeaverTfpUnsafe"),
                   2, FieldType::FM128, 0, 64, 64},
        std::tuple{std::make_pair(
                       [](const std::shared_ptr<yacl::link::Context>& lctx) {
                         return std::make_unique<BeaverTfpUnsafe>(lctx);
                       },
                       "BeaverTfpUnsafe"),
                   2, FieldType::FM64, 0, 32, 32},
        std::tuple{std::make_pair(
                       [](const std::shared_ptr<yacl::link::Context>& lctx) {
                         return std::make_unique<BeaverTinyOt>(lctx);
                       },
                       "BeaverTinyOt"),
                   2, FieldType::FM64, 0, 32, 32},
        std::tuple{std::make_pair(
                       [](const std::shared_ptr<yacl::link::Context>& lctx) {
                         return std::make_unique<BeaverTinyOt>(lctx);
                       },
                       "BeaverTinyOt"),
                   2, FieldType::FM128, 0, 64, 64}),
    [](const testing::TestParamInfo<BeaverTest::ParamType>& p) {
      return fmt::format("{}x{}x{}", std::get<0>(p.param).second,
                         std::get<1>(p.param), std::get<2>(p.param));
    });

TEST_P(BeaverTest, AuthAnd) {
  const auto factory = std::get<0>(GetParam()).first;
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const int64_t kMaxDiff = std::get<3>(GetParam());
  const size_t s = std::get<5>(GetParam());
  const int64_t kNumel = 10;

  std::vector<uint128_t> keys(kWorldSize);
  std::vector<Beaver::Triple_Pair> triples(kWorldSize);

  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> lctx) {
    auto beaver = factory(lctx);
    keys[lctx->Rank()] = beaver->InitSpdzKey(kField, s);
    triples[lctx->Rank()] = beaver->AuthAnd(kField, {kNumel}, s);
  });

  uint128_t sum_key = 0;
  auto sum_a = ring_zeros(kField, {kNumel});
  auto sum_b = ring_zeros(kField, {kNumel});
  auto sum_c = ring_zeros(kField, {kNumel});
  auto sum_a_mac = ring_zeros(kField, {kNumel});
  auto sum_b_mac = ring_zeros(kField, {kNumel});
  auto sum_c_mac = ring_zeros(kField, {kNumel});
  for (Rank r = 0; r < kWorldSize; r++) {
    sum_key += keys[r];

    const auto& [vec, mac_vec] = triples[r];
    const auto& [a, b, c] = vec;
    const auto& [a_mac, b_mac, c_mac] = mac_vec;

    EXPECT_EQ(a.numel(), kNumel);
    EXPECT_EQ(b.numel(), kNumel);
    EXPECT_EQ(c.numel(), kNumel);
    EXPECT_EQ(a_mac.numel(), kNumel);
    EXPECT_EQ(b_mac.numel(), kNumel);
    EXPECT_EQ(c_mac.numel(), kNumel);

    ring_add_(sum_a, a);
    ring_add_(sum_b, b);
    ring_add_(sum_c, c);
    ring_add_(sum_a_mac, a_mac);
    ring_add_(sum_b_mac, b_mac);
    ring_add_(sum_c_mac, c_mac);
  }

  auto valid_a = ring_bitmask(sum_a, 0, 1);
  auto valid_b = ring_bitmask(sum_b, 0, 1);
  auto valid_c = ring_bitmask(sum_c, 0, 1);

  EXPECT_TRUE(ring_all_equal(ring_mul(valid_a, valid_b), valid_c))
      << sum_a << sum_b << sum_c;
  EXPECT_TRUE(ring_all_equal(ring_mul(sum_a, sum_key), sum_a_mac))
      << sum_a << sum_key << sum_a_mac;
  EXPECT_TRUE(ring_all_equal(ring_mul(sum_b, sum_key), sum_b_mac))
      << sum_b << sum_key << sum_b_mac;
  EXPECT_TRUE(ring_all_equal(ring_mul(sum_c, sum_key), sum_c_mac))
      << sum_c << sum_key << sum_c_mac;

  DISPATCH_ALL_FIELDS(kField, "_", [&]() {
    for (auto idx = 0; idx < sum_a.numel(); idx++) {
      auto t = valid_a.at<ring2k_t>(idx) * valid_b.at<ring2k_t>(idx);
      auto err = t > valid_c.at<ring2k_t>(idx) ? t - valid_c.at<ring2k_t>(idx)
                                               : valid_c.at<ring2k_t>(idx) - t;
      EXPECT_LE(err, kMaxDiff);
    }
  });
}

TEST_P(BeaverTest, AuthArrayRef) {
  const auto factory = std::get<0>(GetParam()).first;
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const size_t k = std::get<4>(GetParam());
  const size_t s = std::get<5>(GetParam());
  const int64_t kNumel = 10;

  std::vector<NdArrayRef> values(kWorldSize);
  std::vector<uint128_t> keys(kWorldSize);
  std::vector<NdArrayRef> macs(kWorldSize);

  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> lctx) {
    auto beaver = factory(lctx);
    keys[lctx->Rank()] = beaver->InitSpdzKey(kField, s);
    values[lctx->Rank()] = ring_rand(kField, {kNumel});
    macs[lctx->Rank()] =
        beaver->AuthArrayRef(values[lctx->Rank()], kField, k, s);
  });

  uint128_t sum_key = 0;
  auto sum_a = ring_zeros(kField, {kNumel});
  auto sum_a_mac = ring_zeros(kField, {kNumel});
  for (Rank r = 0; r < kWorldSize; r++) {
    sum_key += keys[r];

    const auto& a = values[r];
    const auto& a_mac = macs[r];
    EXPECT_EQ(a.numel(), kNumel);
    EXPECT_EQ(a_mac.numel(), kNumel);

    ring_add_(sum_a, a);
    ring_add_(sum_a_mac, a_mac);
  }

  EXPECT_TRUE(ring_all_equal(ring_mul(sum_a, sum_key), sum_a_mac))
      << sum_a << sum_key << sum_a_mac;
}

TEST_P(BeaverTest, AuthCoinTossing) {
  const auto factory = std::get<0>(GetParam()).first;
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const size_t k = std::get<4>(GetParam());
  const size_t s = std::get<5>(GetParam());
  const int64_t kNumel = 10;

  std::vector<uint128_t> keys(kWorldSize);
  std::vector<Beaver::Pair> pairs(kWorldSize);

  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> lctx) {
    auto beaver = factory(lctx);
    keys[lctx->Rank()] = beaver->InitSpdzKey(kField, s);
    pairs[lctx->Rank()] = beaver->AuthCoinTossing(kField, {kNumel}, k, s);
  });

  uint128_t sum_key = 0;
  auto sum_a = ring_zeros(kField, {kNumel});
  auto sum_a_mac = ring_zeros(kField, {kNumel});
  for (Rank r = 0; r < kWorldSize; r++) {
    sum_key += keys[r];

    const auto& [a, a_mac] = pairs[r];
    EXPECT_EQ(a.numel(), kNumel);
    EXPECT_EQ(a_mac.numel(), kNumel);

    ring_add_(sum_a, a);
    ring_add_(sum_a_mac, a_mac);
  }

  EXPECT_TRUE(ring_all_equal(ring_mul(sum_a, sum_key), sum_a_mac))
      << sum_a << sum_key << sum_a_mac;
}

TEST_P(BeaverTest, AuthMul) {
  const auto factory = std::get<0>(GetParam()).first;
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const size_t k = std::get<4>(GetParam());
  const size_t s = std::get<5>(GetParam());
  const int64_t kNumel = 10;

  std::vector<uint128_t> keys(kWorldSize);
  std::vector<Beaver::Triple_Pair> triples(kWorldSize);

  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> lctx) {
    auto beaver = factory(lctx);
    keys[lctx->Rank()] = beaver->InitSpdzKey(kField, s);
    triples[lctx->Rank()] = beaver->AuthMul(kField, {kNumel}, k, s);
  });

  uint128_t sum_key = 0;
  auto sum_a = ring_zeros(kField, {kNumel});
  auto sum_b = ring_zeros(kField, {kNumel});
  auto sum_c = ring_zeros(kField, {kNumel});
  auto sum_a_mac = ring_zeros(kField, {kNumel});
  auto sum_b_mac = ring_zeros(kField, {kNumel});
  auto sum_c_mac = ring_zeros(kField, {kNumel});
  for (Rank r = 0; r < kWorldSize; r++) {
    sum_key += keys[r];

    const auto& [vec, mac_vec] = triples[r];
    const auto& [a, b, c] = vec;
    const auto& [a_mac, b_mac, c_mac] = mac_vec;

    EXPECT_EQ(a.numel(), kNumel);
    EXPECT_EQ(b.numel(), kNumel);
    EXPECT_EQ(c.numel(), kNumel);
    EXPECT_EQ(a_mac.numel(), kNumel);
    EXPECT_EQ(b_mac.numel(), kNumel);
    EXPECT_EQ(c_mac.numel(), kNumel);

    ring_add_(sum_a, a);
    ring_add_(sum_b, b);
    ring_add_(sum_c, c);
    ring_add_(sum_a_mac, a_mac);
    ring_add_(sum_b_mac, b_mac);
    ring_add_(sum_c_mac, c_mac);
  }

  EXPECT_TRUE(ring_all_equal(ring_mul(sum_a, sum_b), sum_c))
      << sum_a << sum_b << sum_c;
  EXPECT_TRUE(ring_all_equal(ring_mul(sum_a, sum_key), sum_a_mac))
      << sum_a << sum_key << sum_a_mac;
  EXPECT_TRUE(ring_all_equal(ring_mul(sum_b, sum_key), sum_b_mac))
      << sum_b << sum_key << sum_b_mac;
  EXPECT_TRUE(ring_all_equal(ring_mul(sum_c, sum_key), sum_c_mac))
      << sum_c << sum_key << sum_c_mac;
}

TEST_P(BeaverTest, AuthTrunc) {
  const auto factory = std::get<0>(GetParam()).first;
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const size_t k = std::get<4>(GetParam());
  const size_t s = std::get<5>(GetParam());
  const int64_t kNumel = 7;
  const size_t kBits = 4;

  std::vector<uint128_t> keys(kWorldSize);
  std::vector<Beaver::Pair_Pair> pairs(kWorldSize);

  utils::simulate(
      kWorldSize, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
        auto beaver = factory(lctx);
        keys[lctx->Rank()] = beaver->InitSpdzKey(kField, s);
        pairs[lctx->Rank()] = beaver->AuthTrunc(kField, {kNumel}, kBits, k, s);
      });

  EXPECT_EQ(pairs.size(), kWorldSize);
  uint128_t sum_key = 0;
  auto sum_a = ring_zeros(kField, {kNumel});
  auto sum_b = ring_zeros(kField, {kNumel});
  auto sum_a_mac = ring_zeros(kField, {kNumel});
  auto sum_b_mac = ring_zeros(kField, {kNumel});
  for (Rank r = 0; r < kWorldSize; r++) {
    sum_key += keys[r];

    const auto& [vec, mac_vec] = (pairs[r]);
    const auto& [a, b] = vec;
    const auto& [a_mac, b_mac] = mac_vec;

    EXPECT_EQ(a.numel(), kNumel);
    EXPECT_EQ(b.numel(), kNumel);
    EXPECT_EQ(a_mac.numel(), kNumel);
    EXPECT_EQ(b_mac.numel(), kNumel);

    ring_add_(sum_a, a);
    ring_add_(sum_b, b);
    ring_add_(sum_a_mac, a_mac);
    ring_add_(sum_b_mac, b_mac);
  }

  const size_t bit_len = SizeOf(kField) * 8;
  auto trunc_sum_a =
      ring_arshift(ring_lshift(sum_a, bit_len - k), bit_len - k + kBits);
  ring_bitmask_(trunc_sum_a, 0, k);

  EXPECT_TRUE(ring_all_equal(trunc_sum_a, ring_bitmask(sum_b, 0, k)))
      << trunc_sum_a << sum_b;
  EXPECT_TRUE(ring_all_equal(ring_mul(sum_a, sum_key), sum_a_mac))
      << sum_a << sum_key << sum_a_mac;
  EXPECT_TRUE(ring_all_equal(ring_mul(sum_b, sum_key), sum_b_mac))
      << sum_b << sum_key << sum_b_mac;
}

TEST_P(BeaverTest, AuthDot) {
  const auto factory = std::get<0>(GetParam()).first;
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const int64_t kMaxDiff = std::get<3>(GetParam());
  const size_t k = std::get<4>(GetParam());
  const size_t s = std::get<5>(GetParam());
  // M > N
  const size_t M = 17;
  const size_t N = 8;
  const size_t K = 13;

  std::vector<uint128_t> keys(kWorldSize);
  std::vector<Beaver::Triple_Pair> triples(kWorldSize);

  utils::simulate(
      kWorldSize, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
        auto beaver = factory(lctx);
        keys[lctx->Rank()] = beaver->InitSpdzKey(kField, s);
        triples[lctx->Rank()] = beaver->AuthDot(kField, M, N, K, k, s);
        yacl::link::Barrier(lctx, "BeaverUT");
      });

  EXPECT_EQ(triples.size(), kWorldSize);
  uint128_t sum_key = 0;
  auto sum_a = ring_zeros(kField, {M, K});
  auto sum_b = ring_zeros(kField, {K, N});
  auto sum_c = ring_zeros(kField, {M, N});
  auto sum_a_mac = ring_zeros(kField, {M, K});
  auto sum_b_mac = ring_zeros(kField, {K, N});
  auto sum_c_mac = ring_zeros(kField, {M, N});
  for (Rank r = 0; r < kWorldSize; r++) {
    sum_key += keys[r];

    const auto& [vec, mac_vec] = triples[r];
    const auto& [a, b, c] = vec;
    const auto& [a_mac, b_mac, c_mac] = mac_vec;

    EXPECT_EQ(a.numel(), M * K);
    EXPECT_EQ(b.numel(), K * N);
    EXPECT_EQ(c.numel(), M * N);
    EXPECT_EQ(a_mac.numel(), M * K);
    EXPECT_EQ(b_mac.numel(), K * N);
    EXPECT_EQ(c_mac.numel(), M * N);

    ring_add_(sum_a, a);
    ring_add_(sum_b, b);
    ring_add_(sum_c, c);
    ring_add_(sum_a_mac, a_mac);
    ring_add_(sum_b_mac, b_mac);
    ring_add_(sum_c_mac, c_mac);
  }

  EXPECT_TRUE(ring_all_equal(ring_mul(sum_a, sum_key), sum_a_mac))
      << sum_a << sum_key << sum_a_mac;
  EXPECT_TRUE(ring_all_equal(ring_mul(sum_b, sum_key), sum_b_mac))
      << sum_b << sum_key << sum_b_mac;
  EXPECT_TRUE(ring_all_equal(ring_mul(sum_c, sum_key), sum_c_mac))
      << sum_c << sum_key << sum_c_mac;

  auto res = ring_mmul(sum_a, sum_b);
  DISPATCH_ALL_FIELDS(kField, "_", [&]() {
    for (auto idx = 0; idx < res.numel(); idx++) {
      auto err = res.at<ring2k_t>(idx) > sum_c.at<ring2k_t>(idx)
                     ? res.at<ring2k_t>(idx) - sum_c.at<ring2k_t>(idx)
                     : sum_c.at<ring2k_t>(idx) - res.at<ring2k_t>(idx);
      EXPECT_LE(err, kMaxDiff);
    }
  });
}

}  // namespace spu::mpc::spdz2k
