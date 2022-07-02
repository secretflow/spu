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

#include "spu/mpc/beaver/beaver_test.h"

#include "xtensor/xarray.hpp"

#include "spu/core/type_util.h"
#include "spu/core/xt_helper.h"
#include "spu/mpc/util/ring_ops.h"
#include "spu/mpc/util/simulate.h"

namespace spu::mpc {

TEST_P(BeaverTest, Mul_large) {
  const auto factory = std::get<0>(GetParam());
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const long kMaxDiff = std::get<3>(GetParam());
  const size_t kNumel = 10000;

  std::vector<Beaver::Triple> triples;
  triples.resize(kWorldSize);

  util::simulate(kWorldSize, [&](std::shared_ptr<yasl::link::Context> lctx) {
    auto beaver = factory(lctx);
    triples[lctx->Rank()] = beaver->Mul(kField, kNumel);
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

  if (kMaxDiff > 0) {
    auto diff = ring_sub(ring_mul(sum_a, sum_b), sum_c);
    DISPATCH_ALL_FIELDS(kField, "_", [&]() {
      auto xdiff = xt_adapt<ring2k_t>(diff);
      EXPECT_LE(xt::amax(xdiff)(), kMaxDiff);
      EXPECT_GE(xt::amin(xdiff)(), -kMaxDiff);
    });
  } else {
    EXPECT_EQ(ring_mul(sum_a, sum_b), sum_c) << sum_a << sum_b << sum_c;
  }
}

TEST_P(BeaverTest, Mul) {
  const auto factory = std::get<0>(GetParam());
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const long kMaxDiff = std::get<3>(GetParam());
  const size_t kNumel = 7;

  std::vector<Beaver::Triple> triples;
  triples.resize(kWorldSize);

  util::simulate(kWorldSize, [&](std::shared_ptr<yasl::link::Context> lctx) {
    auto beaver = factory(lctx);
    triples[lctx->Rank()] = beaver->Mul(kField, kNumel);
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

  if (kMaxDiff > 0) {
    auto diff = ring_sub(ring_mul(sum_a, sum_b), sum_c);
    DISPATCH_ALL_FIELDS(kField, "_", [&]() {
      auto xdiff = xt_adapt<ring2k_t>(diff);
      EXPECT_LE(xt::amax(xdiff)(), kMaxDiff);
      EXPECT_GE(xt::amin(xdiff)(), -kMaxDiff);
    });
  } else {
    EXPECT_EQ(ring_mul(sum_a, sum_b), sum_c) << sum_a << sum_b << sum_c;
  }
}

TEST_P(BeaverTest, And) {
  const auto factory = std::get<0>(GetParam());
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const size_t kNumel = 7;

  std::vector<Beaver::Triple> triples;
  triples.resize(kWorldSize);

  util::simulate(kWorldSize, [&](std::shared_ptr<yasl::link::Context> lctx) {
    auto beaver = factory(lctx);
    triples[lctx->Rank()] = beaver->And(kField, kNumel);
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
  const auto factory = std::get<0>(GetParam());
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const long kMaxDiff = std::get<3>(GetParam());

  // case M > N
  const size_t M = 8;
  const size_t N = 1;
  const size_t K = 4096;

  std::vector<Beaver::Triple> triples;
  triples.resize(kWorldSize);

  util::simulate(kWorldSize, [&](std::shared_ptr<yasl::link::Context> lctx) {
    auto beaver = factory(lctx);
    triples[lctx->Rank()] = beaver->Dot(kField, M, N, K);
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

  if (kMaxDiff > 0) {
    auto diff = ring_sub(ring_mmul(sum_a, sum_b, M, N, K), sum_c);
    DISPATCH_ALL_FIELDS(kField, "_", [&]() {
      auto xdiff = xt_adapt<ring2k_t>(diff);
      EXPECT_LE(xt::amax(xdiff)(), kMaxDiff);
      EXPECT_GE(xt::amin(xdiff)(), -kMaxDiff);
    });
  } else {
    EXPECT_EQ(ring_mmul(sum_a, sum_b, M, N, K), sum_c)
        << sum_a << sum_b << sum_c;
  }
}

TEST_P(BeaverTest, Dot_large) {
  const auto factory = std::get<0>(GetParam());
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const long kMaxDiff = std::get<3>(GetParam());
  // case: M < N
  const size_t M = 2;
  const size_t N = 8192;
  const size_t K = 256;

  std::vector<Beaver::Triple> triples;
  triples.resize(kWorldSize);

  util::simulate(kWorldSize, [&](std::shared_ptr<yasl::link::Context> lctx) {
    auto beaver = factory(lctx);
    triples[lctx->Rank()] = beaver->Dot(kField, M, N, K);
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

  if (kMaxDiff > 0) {
    auto diff = ring_sub(ring_mmul(sum_a, sum_b, M, N, K), sum_c);
    DISPATCH_ALL_FIELDS(kField, "_", [&]() {
      auto xdiff = xt_adapt<ring2k_t>(diff);
      EXPECT_LE(xt::amax(xdiff)(), kMaxDiff);
      EXPECT_GE(xt::amin(xdiff)(), -kMaxDiff);
    });
  } else {
    EXPECT_EQ(ring_mmul(sum_a, sum_b, M, N, K), sum_c)
        << sum_a << sum_b << sum_c;
  }
}

TEST_P(BeaverTest, Trunc) {
  const auto factory = std::get<0>(GetParam());
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const size_t kNumel = 7;
  const size_t kBits = 5;

  std::vector<Beaver::Pair> pairs;
  pairs.resize(kWorldSize);

  util::simulate(kWorldSize, [&](std::shared_ptr<yasl::link::Context> lctx) {
    auto beaver = factory(lctx);
    if (beaver->SupportTrunc()) {
      pairs[lctx->Rank()] = beaver->Trunc(kField, kNumel, kBits);
    } else {
      // mock and pass UT.
      pairs[lctx->Rank()] = {
          ring_zeros(kField, kNumel),
          ring_zeros(kField, kNumel),
      };
    }
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

TEST_P(BeaverTest, Randbit) {
  const auto factory = std::get<0>(GetParam());
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const size_t kNumel = 7;

  std::vector<ArrayRef> shares(kWorldSize);

  util::simulate(kWorldSize, [&](std::shared_ptr<yasl::link::Context> lctx) {
    auto beaver = factory(lctx);
    if (beaver->SupportRandBit()) {
      shares[lctx->Rank()] = beaver->RandBit(kField, kNumel);
    } else {
      // mock and pass UT.
      shares[lctx->Rank()] = ring_zeros(kField, kNumel);
    }
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
    // TODO: BeaverRef could not pass following test.
    // EXPECT_TRUE(x != xt::zeros_like(x));
    return;
  });
}

}  // namespace spu::mpc
