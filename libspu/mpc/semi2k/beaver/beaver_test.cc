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

#include "fmt/format.h"
#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "yacl/link/link.h"

#include "libspu/core/type_util.h"
#include "libspu/core/xt_helper.h"
#include "libspu/mpc/semi2k/beaver/beaver_tfp.h"
#include "libspu/mpc/semi2k/beaver/beaver_ttp.h"
#include "libspu/mpc/semi2k/beaver/ttp_server/beaver_server.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::semi2k {

class BeaverTest
    : public ::testing::TestWithParam<
          std::tuple<std::pair<std::function<std::unique_ptr<Beaver>(
                                   const std::shared_ptr<yacl::link::Context>&,
                                   const BeaverTtp::Options&)>,
                               std::string>,
                     size_t, FieldType, int64_t>> {
 private:
  static std::unique_ptr<brpc::Server> server_;

 public:
  using Triple = typename Beaver::Triple;
  using Pair = typename Beaver::Pair;

  static void SetUpTestSuite() { server_ = beaver::ttp_server::RunServer(0); }

  static void TearDownTestSuite() {
    server_->Stop(0);
    server_.reset();
  }

 protected:
  BeaverTtp::Options ttp_options_;
  void SetUp() override {
    auto server_host =
        fmt::format("127.0.0.1:{}", server_->listen_address().port);
    ttp_options_.server_host = server_host;
    ttp_options_.adjust_rank = 1;
    ttp_options_.session_id = "beaver_test";
  }
};

std::unique_ptr<brpc::Server> BeaverTest::server_;

INSTANTIATE_TEST_SUITE_P(
    BeaverTfpUnsafeTest, BeaverTest,
    testing::Combine(
        testing::Values(std::make_pair(
                            [](const std::shared_ptr<yacl::link::Context>& lctx,
                               const BeaverTtp::Options&) {
                              return std::make_unique<BeaverTfpUnsafe>(lctx);
                            },
                            "BeaverTfpUnsafe"),
                        std::make_pair(
                            [](const std::shared_ptr<yacl::link::Context>& lctx,
                               const BeaverTtp::Options& ops) {
                              return std::make_unique<BeaverTtp>(lctx, ops);
                            },
                            "BeaverTtp")),
        testing::Values(4, 3, 2),
        testing::Values(FieldType::FM32, FieldType::FM64, FieldType::FM128),
        testing::Values(0)),  // max beaver diff,
    [](const testing::TestParamInfo<BeaverTest::ParamType>& p) {
      return fmt::format("{}x{}x{}", std::get<0>(p.param).second,
                         std::get<1>(p.param), std::get<2>(p.param));
    });

TEST_P(BeaverTest, Mul_large) {
  const auto factory = std::get<0>(GetParam()).first;
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const int64_t kMaxDiff = std::get<3>(GetParam());
  const size_t kNumel = 10000;

  std::vector<Triple> triples;
  triples.resize(kWorldSize);

  utils::simulate(kWorldSize,
                  [&](const std::shared_ptr<yacl::link::Context>& lctx) {
                    auto beaver = factory(lctx, ttp_options_);
                    triples[lctx->Rank()] = beaver->Mul(kField, kNumel);
                    yacl::link::Barrier(lctx, "BeaverUT");
                  });

  auto sum_a = ring_zeros(kField, kNumel);
  auto sum_b = ring_zeros(kField, kNumel);
  auto sum_c = ring_zeros(kField, kNumel);
  for (Rank r = 0; r < kWorldSize; r++) {
    const auto& [a, b, c] = triples[r];
    EXPECT_EQ(a.numel(), kNumel);
    EXPECT_EQ(b.numel(), kNumel);
    EXPECT_EQ(c.numel(), kNumel);

    ring_add_(sum_a, a);
    ring_add_(sum_b, b);
    ring_add_(sum_c, c);
  }

  DISPATCH_ALL_FIELDS(kField, "_", [&]() {
    auto _a = ArrayView<ring2k_t>(sum_a);
    auto _b = ArrayView<ring2k_t>(sum_b);
    auto _c = ArrayView<ring2k_t>(sum_c);
    for (auto idx = 0; idx < sum_a.numel(); idx++) {
      auto t = _a[idx] * _b[idx];
      auto err = t > _c[idx] ? t - _c[idx] : _c[idx] - t;
      EXPECT_LE(err, kMaxDiff);
    }
  });
}

TEST_P(BeaverTest, Mul) {
  const auto factory = std::get<0>(GetParam()).first;
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const int64_t kMaxDiff = std::get<3>(GetParam());
  const size_t kNumel = 7;

  std::vector<Triple> triples;
  triples.resize(kWorldSize);

  utils::simulate(kWorldSize,
                  [&](const std::shared_ptr<yacl::link::Context>& lctx) {
                    auto beaver = factory(lctx, ttp_options_);
                    triples[lctx->Rank()] = beaver->Mul(kField, kNumel);
                    yacl::link::Barrier(lctx, "BeaverUT");
                  });

  auto sum_a = ring_zeros(kField, kNumel);
  auto sum_b = ring_zeros(kField, kNumel);
  auto sum_c = ring_zeros(kField, kNumel);
  for (Rank r = 0; r < kWorldSize; r++) {
    const auto& [a, b, c] = triples[r];
    EXPECT_EQ(a.numel(), kNumel);
    EXPECT_EQ(b.numel(), kNumel);
    EXPECT_EQ(c.numel(), kNumel);

    ring_add_(sum_a, a);
    ring_add_(sum_b, b);
    ring_add_(sum_c, c);
  }

  DISPATCH_ALL_FIELDS(kField, "_", [&]() {
    auto _a = ArrayView<ring2k_t>(sum_a);
    auto _b = ArrayView<ring2k_t>(sum_b);
    auto _c = ArrayView<ring2k_t>(sum_c);
    for (auto idx = 0; idx < sum_a.numel(); idx++) {
      auto t = _a[idx] * _b[idx];
      auto err = t > _c[idx] ? t - _c[idx] : _c[idx] - t;
      EXPECT_LE(err, kMaxDiff);
    }
  });
}

TEST_P(BeaverTest, And) {
  const auto factory = std::get<0>(GetParam()).first;
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const size_t kNumel = 7;

  std::vector<Triple> triples;
  triples.resize(kWorldSize);

  utils::simulate(kWorldSize,
                  [&](const std::shared_ptr<yacl::link::Context>& lctx) {
                    auto beaver = factory(lctx, ttp_options_);
                    triples[lctx->Rank()] = beaver->And(kField, kNumel);
                    yacl::link::Barrier(lctx, "BeaverUT");
                  });

  EXPECT_EQ(triples.size(), kWorldSize);
  auto sum_a = ring_zeros(kField, kNumel);
  auto sum_b = ring_zeros(kField, kNumel);
  auto sum_c = ring_zeros(kField, kNumel);
  for (Rank r = 0; r < kWorldSize; r++) {
    const auto& [a, b, c] = triples[r];
    EXPECT_EQ(a.numel(), kNumel);
    EXPECT_EQ(b.numel(), kNumel);
    EXPECT_EQ(c.numel(), kNumel);

    ring_xor_(sum_a, a);
    ring_xor_(sum_b, b);
    ring_xor_(sum_c, c);
  }
  EXPECT_EQ(ring_and(sum_a, sum_b), sum_c) << sum_a << sum_b << sum_c;
}

TEST_P(BeaverTest, Dot) {
  const auto factory = std::get<0>(GetParam()).first;
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const int64_t kMaxDiff = std::get<3>(GetParam());
  // M > N
  const size_t M = 17;
  const size_t N = 8;
  const size_t K = 1024;

  std::vector<Triple> triples;
  triples.resize(kWorldSize);

  utils::simulate(kWorldSize,
                  [&](const std::shared_ptr<yacl::link::Context>& lctx) {
                    auto beaver = factory(lctx, ttp_options_);
                    triples[lctx->Rank()] = beaver->Dot(kField, M, N, K);
                    yacl::link::Barrier(lctx, "BeaverUT");
                  });

  EXPECT_EQ(triples.size(), kWorldSize);
  auto sum_a = ring_zeros(kField, M * K);
  auto sum_b = ring_zeros(kField, K * N);
  auto sum_c = ring_zeros(kField, M * N);
  for (Rank r = 0; r < kWorldSize; r++) {
    const auto& [a, b, c] = triples[r];
    EXPECT_EQ(a.numel(), M * K);
    EXPECT_EQ(b.numel(), K * N);
    EXPECT_EQ(c.numel(), M * N);

    ring_add_(sum_a, a);
    ring_add_(sum_b, b);
    ring_add_(sum_c, c);
  }

  auto res = ring_mmul(sum_a, sum_b, M, N, K);
  DISPATCH_ALL_FIELDS(kField, "_", [&]() {
    auto _r = ArrayView<ring2k_t>(res);
    auto _c = ArrayView<ring2k_t>(sum_c);
    for (auto idx = 0; idx < _r.numel(); idx++) {
      auto err = _r[idx] > _c[idx] ? _r[idx] - _c[idx] : _c[idx] - _r[idx];
      EXPECT_LE(err, kMaxDiff);
    }
  });
}

TEST_P(BeaverTest, Dot_large) {
  const auto factory = std::get<0>(GetParam()).first;
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const int64_t kMaxDiff = std::get<3>(GetParam());
  // M < N
  const size_t M = 11;
  const size_t N = 20;
  const size_t K = 1023;

  std::vector<Triple> triples;
  triples.resize(kWorldSize);

  utils::simulate(kWorldSize,
                  [&](const std::shared_ptr<yacl::link::Context>& lctx) {
                    auto beaver = factory(lctx, ttp_options_);
                    triples[lctx->Rank()] = beaver->Dot(kField, M, N, K);
                    yacl::link::Barrier(lctx, "BeaverUT");
                  });

  EXPECT_EQ(triples.size(), kWorldSize);
  auto sum_a = ring_zeros(kField, M * K);
  auto sum_b = ring_zeros(kField, K * N);
  auto sum_c = ring_zeros(kField, M * N);
  for (Rank r = 0; r < kWorldSize; r++) {
    const auto& [a, b, c] = triples[r];
    EXPECT_EQ(a.numel(), M * K);
    EXPECT_EQ(b.numel(), K * N);
    EXPECT_EQ(c.numel(), M * N);

    ring_add_(sum_a, a);
    ring_add_(sum_b, b);
    ring_add_(sum_c, c);
  }

  auto res = ring_mmul(sum_a, sum_b, M, N, K);
  DISPATCH_ALL_FIELDS(kField, "_", [&]() {
    auto _r = ArrayView<ring2k_t>(res);
    auto _c = ArrayView<ring2k_t>(sum_c);
    for (auto idx = 0; idx < _r.numel(); idx++) {
      auto err = _r[idx] > _c[idx] ? _r[idx] - _c[idx] : _c[idx] - _r[idx];
      EXPECT_LE(err, kMaxDiff);
    }
  });
}

TEST_P(BeaverTest, Trunc) {
  const auto factory = std::get<0>(GetParam()).first;
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const size_t kNumel = 7;
  const size_t kBits = 5;

  std::vector<Pair> pairs;
  pairs.resize(kWorldSize);

  utils::simulate(kWorldSize,
                  [&](const std::shared_ptr<yacl::link::Context>& lctx) {
                    auto beaver = factory(lctx, ttp_options_);
                    pairs[lctx->Rank()] = beaver->Trunc(kField, kNumel, kBits);
                    yacl::link::Barrier(lctx, "BeaverUT");
                  });

  EXPECT_EQ(pairs.size(), kWorldSize);
  auto sum_a = ring_zeros(kField, kNumel);
  auto sum_b = ring_zeros(kField, kNumel);
  for (Rank r = 0; r < kWorldSize; r++) {
    const auto& [a, b] = pairs[r];
    EXPECT_EQ(a.numel(), kNumel);
    EXPECT_EQ(b.numel(), kNumel);

    ring_add_(sum_a, a);
    ring_add_(sum_b, b);
  }
  EXPECT_EQ(ring_arshift(sum_a, kBits), sum_b) << sum_a << sum_b;
}

TEST_P(BeaverTest, TruncPr) {
  const auto factory = std::get<0>(GetParam()).first;
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const size_t kNumel = 7;
  const size_t kBits = 5;
  const size_t kRingSize = SizeOf(kField) * 8;

  std::vector<Triple> rets;
  rets.resize(kWorldSize);

  utils::simulate(kWorldSize,
                  [&](const std::shared_ptr<yacl::link::Context>& lctx) {
                    auto beaver = factory(lctx, ttp_options_);
                    rets[lctx->Rank()] = beaver->TruncPr(kField, kNumel, kBits);
                    yacl::link::Barrier(lctx, "BeaverUT");
                  });

  EXPECT_EQ(rets.size(), kWorldSize);
  auto sum_r = ring_zeros(kField, kNumel);
  auto sum_rc = ring_zeros(kField, kNumel);
  auto sum_rb = ring_zeros(kField, kNumel);
  for (Rank rank = 0; rank < kWorldSize; rank++) {
    const auto& [r, rc, rb] = rets[rank];
    EXPECT_EQ(r.numel(), kNumel);
    EXPECT_EQ(rc.numel(), kNumel);
    EXPECT_EQ(rb.numel(), kNumel);

    ring_add_(sum_r, r);
    ring_add_(sum_rc, rc);
    ring_add_(sum_rb, rb);
  }

  DISPATCH_ALL_FIELDS(kField, "semi2k.truncpr.ut", [&]() {
    using T = ring2k_t;
    for (int64_t i = 0; i < sum_r.numel(); i++) {
      auto r = sum_r.at<T>(i);
      auto rc = sum_rc.at<T>(i);
      auto rb = sum_rb.at<T>(i);

      EXPECT_EQ((r << 1) >> (kBits + 1), rc)
          << fmt::format("error: {0:X} {1:X}\n", (r << 1) >> (kBits + 1), rc);
      EXPECT_EQ(r >> (kRingSize - 1), rb)
          << fmt::format("error: {0:X} {1:X}\n", r >> (kRingSize - 1), rb);
    }
  });
}

TEST_P(BeaverTest, Randbit) {
  const auto factory = std::get<0>(GetParam()).first;
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const size_t kNumel = 51;

  std::vector<ArrayRef> shares(kWorldSize);

  utils::simulate(kWorldSize,
                  [&](const std::shared_ptr<yacl::link::Context>& lctx) {
                    auto beaver = factory(lctx, ttp_options_);
                    shares[lctx->Rank()] = beaver->RandBit(kField, kNumel);
                    yacl::link::Barrier(lctx, "BeaverUT");
                  });

  EXPECT_EQ(shares.size(), kWorldSize);
  auto sum = ring_zeros(kField, kNumel);
  for (Rank r = 0; r < kWorldSize; r++) {
    EXPECT_EQ(shares[r].numel(), kNumel);
    ring_add_(sum, shares[r]);
  }

  DISPATCH_ALL_FIELDS(kField, "_", [&]() {
    using scalar_t = typename Ring2kTrait<_kField>::scalar_t;
    auto x = xt_adapt<scalar_t>(sum);
    EXPECT_TRUE(xt::all(x <= xt::ones_like(x)));
    EXPECT_TRUE(xt::all(x >= xt::zeros_like(x)));
    EXPECT_TRUE(x != xt::zeros_like(x));
    return;
  });
}

}  // namespace spu::mpc::semi2k
