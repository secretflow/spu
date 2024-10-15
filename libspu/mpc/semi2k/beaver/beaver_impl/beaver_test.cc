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

#include <random>

#include "gtest/gtest.h"
#include "yacl/crypto/key_utils.h"
#include "yacl/link/algorithm/barrier.h"
#include "yacl/link/context.h"

#include "libspu/core/type_util.h"
#include "libspu/core/xt_helper.h"
#include "libspu/mpc/semi2k/beaver/beaver_impl/beaver_tfp.h"
#include "libspu/mpc/semi2k/beaver/beaver_impl/beaver_ttp.h"
#include "libspu/mpc/semi2k/beaver/beaver_impl/ttp_server/beaver_server.h"
#include "libspu/mpc/utils/gfmp.h"
#include "libspu/mpc/utils/permute.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::semi2k {

class BeaverTest
    : public ::testing::TestWithParam<
          std::tuple<std::pair<std::function<std::unique_ptr<Beaver>(
                                   const std::shared_ptr<yacl::link::Context>&,
                                   BeaverTtp::Options, size_t)>,
                               std::string>,
                     size_t, FieldType, int64_t, size_t>> {
 private:
  static std::pair<yacl::Buffer, yacl::Buffer> asym_crypto_key_;
  static beaver::ttp_server::ServerOptions options_;
  static std::unique_ptr<brpc::Server> server_;

 public:
  using Triple = typename Beaver::Triple;
  using Pair = typename Beaver::Pair;

  static void SetUpTestSuite() {
    asym_crypto_key_ = yacl::crypto::GenSm2KeyPairToPemBuf();
    options_.asym_crypto_schema = "sm2";
    options_.server_private_key = asym_crypto_key_.second;
    options_.port = 0;
    server_ = beaver::ttp_server::RunServer(options_);
  }

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
    ttp_options_.asym_crypto_schema = "sm2";
    ttp_options_.server_public_key = asym_crypto_key_.first;
  }
};

std::unique_ptr<brpc::Server> BeaverTest::server_;
std::pair<yacl::Buffer, yacl::Buffer> BeaverTest::asym_crypto_key_;
beaver::ttp_server::ServerOptions BeaverTest::options_;

INSTANTIATE_TEST_SUITE_P(
    BeaverTfpUnsafeTest, BeaverTest,
    testing::Combine(
        testing::Values(std::make_pair(
                            [](const std::shared_ptr<yacl::link::Context>& lctx,
                               BeaverTtp::Options, size_t) {
                              return std::make_unique<BeaverTfpUnsafe>(lctx);
                            },
                            "BeaverTfpUnsafe"),
                        std::make_pair(
                            [](const std::shared_ptr<yacl::link::Context>& lctx,
                               BeaverTtp::Options ops, size_t adjust_rank) {
                              ops.adjust_rank = adjust_rank;
                              return std::make_unique<BeaverTtp>(lctx, ops);
                            },
                            "BeaverTtp")),
        testing::Values(4, 3, 2),
        testing::Values(FieldType::FM32, FieldType::FM64, FieldType::FM128),
        testing::Values(0),    // max beaver diff,
        testing::Values(0, 1)  // adjust_rank
        ),
    [](const testing::TestParamInfo<BeaverTest::ParamType>& p) {
      return fmt::format("{}x{}x{}x{}", std::get<0>(p.param).second,
                         std::get<1>(p.param), std::get<2>(p.param),
                         std::get<4>(p.param));
    });

namespace {
template <class T>
std::vector<NdArrayRef> open_buffer(std::vector<T>& in_buffers,
                                    FieldType k_field,
                                    const std::vector<Shape>& shapes,
                                    size_t k_world_size, bool add_open) {
  std::vector<NdArrayRef> ret;

  auto reduce = [&](NdArrayRef& r, yacl::Buffer& b) {
    if (b.size() == 0) {
      return;
    }
    EXPECT_EQ(b.size(), r.shape().numel() * SizeOf(k_field));
    NdArrayRef a(std::make_shared<yacl::Buffer>(std::move(b)), ret[0].eltype(),
                 r.shape());
    if (add_open) {
      ring_add_(r, a);
    } else {
      ring_xor_(r, a);
    }
  };
  if constexpr (std::is_same_v<T, Beaver::Triple>) {
    ret.resize(3);
    SPU_ENFORCE(shapes.size() == 3);
    for (size_t i = 0; i < shapes.size(); i++) {
      ret[i] = ring_zeros(k_field, shapes[i]);
    }
    for (Rank r = 0; r < k_world_size; r++) {
      auto& [a_buf, b_buf, c_buf] = in_buffers[r];
      reduce(ret[0], a_buf);
      reduce(ret[1], b_buf);
      reduce(ret[2], c_buf);
    }
  } else if constexpr (std::is_same_v<T, Beaver::Pair>) {
    ret.resize(2);
    SPU_ENFORCE(shapes.size() == 2);
    for (size_t i = 0; i < shapes.size(); i++) {
      ret[i] = ring_zeros(k_field, shapes[i]);
    }
    for (Rank r = 0; r < k_world_size; r++) {
      auto& [a_buf, b_buf] = in_buffers[r];
      reduce(ret[0], a_buf);
      reduce(ret[1], b_buf);
    }
  } else if constexpr (std::is_same_v<T, Beaver::Array>) {
    ret.resize(1);
    SPU_ENFORCE(shapes.size() == 1);
    for (size_t i = 0; i < shapes.size(); i++) {
      ret[i] = ring_zeros(k_field, shapes[i]);
    }
    for (Rank r = 0; r < k_world_size; r++) {
      auto& a_buf = in_buffers[r];
      reduce(ret[0], a_buf);
    }
  }
  return ret;
}

template <class T>
std::vector<NdArrayRef> open_buffer_gfmp(std::vector<T>& in_buffers,
                                         FieldType k_field,
                                         const std::vector<Shape>& shapes,
                                         size_t k_world_size, bool add_open) {
  std::vector<NdArrayRef> ret;

  auto reduce = [&](NdArrayRef& r, yacl::Buffer& b) {
    if (b.size() == 0) {
      return;
    }
    EXPECT_EQ(b.size(), r.shape().numel() * SizeOf(k_field));
    NdArrayRef a(std::make_shared<yacl::Buffer>(std::move(b)), ret[0].eltype(),
                 r.shape());
    auto Ta = r.eltype();
    gfmp_add_mod_(r, a.as(Ta));
  };
  if constexpr (std::is_same_v<T, Beaver::Triple>) {
    ret.resize(3);
    SPU_ENFORCE(shapes.size() == 3);
    for (size_t i = 0; i < shapes.size(); i++) {
      ret[i] = gfmp_zeros(k_field, shapes[i]);
    }
    for (Rank r = 0; r < k_world_size; r++) {
      auto& [a_buf, b_buf, c_buf] = in_buffers[r];
      reduce(ret[0], a_buf);
      reduce(ret[1], b_buf);
      reduce(ret[2], c_buf);
    }
  } else if constexpr (std::is_same_v<T, Beaver::Pair>) {
    ret.resize(2);
    SPU_ENFORCE(shapes.size() == 2);
    for (size_t i = 0; i < shapes.size(); i++) {
      ret[i] = gfmp_zeros(k_field, shapes[i]);
    }
    for (Rank r = 0; r < k_world_size; r++) {
      auto& [a_buf, b_buf] = in_buffers[r];
      reduce(ret[0], a_buf);
      reduce(ret[1], b_buf);
    }
  } else if constexpr (std::is_same_v<T, Beaver::Array>) {
    ret.resize(1);
    SPU_ENFORCE(shapes.size() == 1);
    for (size_t i = 0; i < shapes.size(); i++) {
      ret[i] = gfmp_zeros(k_field, shapes[i]);
    }
    for (Rank r = 0; r < k_world_size; r++) {
      auto& a_buf = in_buffers[r];
      reduce(ret[0], a_buf);
    }
  }
  return ret;
}
}  // namespace

TEST_P(BeaverTest, Mul_large) {
  const auto factory = std::get<0>(GetParam()).first;
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const int64_t kMaxDiff = std::get<3>(GetParam());
  const size_t adjust_rank = std::get<4>(GetParam());
  const int64_t kNumel = 10000;

  std::vector<Triple> triples(kWorldSize);

  std::vector<Beaver::ReplayDesc> x_desc(kWorldSize);
  std::vector<Beaver::ReplayDesc> y_desc(kWorldSize);
  NdArrayRef x_cache;
  NdArrayRef y_cache;
  {
    utils::simulate(
        kWorldSize, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
          auto beaver = factory(lctx, ttp_options_, adjust_rank);
          triples[lctx->Rank()] = beaver->Mul(
              kField, kNumel, &x_desc[lctx->Rank()], &y_desc[lctx->Rank()]);
          yacl::link::Barrier(lctx, "BeaverUT");
        });

    auto open = open_buffer(triples, kField, std::vector<Shape>(3, {kNumel}),
                            kWorldSize, true);

    DISPATCH_ALL_FIELDS(kField, [&]() {
      NdArrayView<ring2k_t> _a(open[0]);
      NdArrayView<ring2k_t> _b(open[1]);
      NdArrayView<ring2k_t> _c(open[2]);
      for (auto idx = 0; idx < _a.numel(); idx++) {
        auto t = _a[idx] * _b[idx];
        auto err = t > _c[idx] ? t - _c[idx] : _c[idx] - t;
        EXPECT_LE(err, kMaxDiff);
      }
    });

    x_cache = open[0];
    y_cache = open[1];
  }
  {
    utils::simulate(
        kWorldSize, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
          auto beaver = factory(lctx, ttp_options_, adjust_rank);
          x_desc[lctx->Rank()].status = Beaver::Replay;
          triples[lctx->Rank()] =
              beaver->Mul(kField, kNumel, &x_desc[lctx->Rank()], nullptr);
          yacl::link::Barrier(lctx, "BeaverUT");
        });

    auto open = open_buffer(triples, kField, std::vector<Shape>(3, {kNumel}),
                            kWorldSize, true);

    DISPATCH_ALL_FIELDS(kField, [&]() {
      NdArrayView<ring2k_t> _a(open[0]);
      NdArrayView<ring2k_t> _a_cache(x_cache);
      NdArrayView<ring2k_t> _b(open[1]);
      NdArrayView<ring2k_t> _c(open[2]);
      for (auto idx = 0; idx < _a.numel(); idx++) {
        EXPECT_EQ(_a_cache[idx], _a[idx]);
        auto t = _a[idx] * _b[idx];
        auto err = t > _c[idx] ? t - _c[idx] : _c[idx] - t;
        EXPECT_LE(err, kMaxDiff);
      }
    });
  }
  {
    utils::simulate(
        kWorldSize, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
          auto beaver = factory(lctx, ttp_options_, adjust_rank);
          y_desc[lctx->Rank()].status = Beaver::Replay;
          triples[lctx->Rank()] =
              beaver->Mul(kField, kNumel, nullptr, &y_desc[lctx->Rank()]);
          yacl::link::Barrier(lctx, "BeaverUT");
        });

    auto open = open_buffer(triples, kField, std::vector<Shape>(3, {kNumel}),
                            kWorldSize, true);

    DISPATCH_ALL_FIELDS(kField, [&]() {
      NdArrayView<ring2k_t> _a(open[0]);
      NdArrayView<ring2k_t> _b(open[1]);
      NdArrayView<ring2k_t> _b_cache(y_cache);
      NdArrayView<ring2k_t> _c(open[2]);
      for (auto idx = 0; idx < _a.numel(); idx++) {
        EXPECT_EQ(_b_cache[idx], _b[idx]);
        auto t = _a[idx] * _b[idx];
        auto err = t > _c[idx] ? t - _c[idx] : _c[idx] - t;
        EXPECT_LE(err, kMaxDiff);
      }
    });
  }
  {
    utils::simulate(
        kWorldSize, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
          auto beaver = factory(lctx, ttp_options_, adjust_rank);
          x_desc[lctx->Rank()].status = Beaver::Replay;
          y_desc[lctx->Rank()].status = Beaver::Replay;
          triples[lctx->Rank()] = beaver->Mul(
              kField, kNumel, &x_desc[lctx->Rank()], &y_desc[lctx->Rank()]);
          yacl::link::Barrier(lctx, "BeaverUT");
        });

    auto open = open_buffer(triples, kField, std::vector<Shape>(3, {kNumel}),
                            kWorldSize, true);

    DISPATCH_ALL_FIELDS(kField, [&]() {
      NdArrayView<ring2k_t> _a(open[0]);
      NdArrayView<ring2k_t> _b(open[1]);
      NdArrayView<ring2k_t> _a_cache(x_cache);
      NdArrayView<ring2k_t> _b_cache(y_cache);
      NdArrayView<ring2k_t> _c(open[2]);
      for (auto idx = 0; idx < _a.numel(); idx++) {
        EXPECT_EQ(_a_cache[idx], _a[idx]);
        EXPECT_EQ(_b_cache[idx], _b[idx]);
        auto t = _a[idx] * _b[idx];
        auto err = t > _c[idx] ? t - _c[idx] : _c[idx] - t;
        EXPECT_LE(err, kMaxDiff);
      }
    });
  }
  {
    utils::simulate(
        kWorldSize, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
          auto beaver = factory(lctx, ttp_options_, adjust_rank);
          x_desc[lctx->Rank()].status = Beaver::TransposeReplay;
          y_desc[lctx->Rank()].status = Beaver::TransposeReplay;
          // mul not support transpose.
          triples[lctx->Rank()] = beaver->Mul(
              kField, kNumel, &x_desc[lctx->Rank()], &y_desc[lctx->Rank()]);
          yacl::link::Barrier(lctx, "BeaverUT");
        });

    auto open = open_buffer(triples, kField, std::vector<Shape>(3, {kNumel}),
                            kWorldSize, true);

    DISPATCH_ALL_FIELDS(kField, [&]() {
      NdArrayView<ring2k_t> _a(open[0]);
      NdArrayView<ring2k_t> _b(open[1]);
      NdArrayView<ring2k_t> _a_cache(x_cache);
      NdArrayView<ring2k_t> _b_cache(y_cache);
      NdArrayView<ring2k_t> _c(open[2]);
      for (auto idx = 0; idx < _a.numel(); idx++) {
        // mul not support transpose.
        // enforce ne
        EXPECT_NE(_a_cache[idx], _a[idx]);
        EXPECT_NE(_b_cache[idx], _b[idx]);
        auto t = _a[idx] * _b[idx];
        auto err = t > _c[idx] ? t - _c[idx] : _c[idx] - t;
        EXPECT_LE(err, kMaxDiff);
      }
    });
  }
}

TEST_P(BeaverTest, Mul) {
  const auto factory = std::get<0>(GetParam()).first;
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const int64_t kMaxDiff = std::get<3>(GetParam());
  const size_t adjust_rank = std::get<4>(GetParam());
  const int64_t kNumel = 7;

  std::vector<Triple> triples(kWorldSize);

  std::vector<Beaver::ReplayDesc> x_desc(kWorldSize);
  std::vector<Beaver::ReplayDesc> y_desc(kWorldSize);
  NdArrayRef x_cache;
  NdArrayRef y_cache;
  {
    utils::simulate(
        kWorldSize, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
          auto beaver = factory(lctx, ttp_options_, adjust_rank);
          triples[lctx->Rank()] = beaver->Mul(
              kField, kNumel, &x_desc[lctx->Rank()], &y_desc[lctx->Rank()]);
          yacl::link::Barrier(lctx, "BeaverUT");
        });

    auto open = open_buffer(triples, kField, std::vector<Shape>(3, {kNumel}),
                            kWorldSize, true);

    DISPATCH_ALL_FIELDS(kField, [&]() {
      NdArrayView<ring2k_t> _a(open[0]);
      NdArrayView<ring2k_t> _b(open[1]);
      NdArrayView<ring2k_t> _c(open[2]);
      for (auto idx = 0; idx < _a.numel(); idx++) {
        auto t = _a[idx] * _b[idx];
        auto err = t > _c[idx] ? t - _c[idx] : _c[idx] - t;
        EXPECT_LE(err, kMaxDiff);
      }
    });

    x_cache = open[0];
    y_cache = open[1];
  }
  {
    utils::simulate(
        kWorldSize, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
          auto beaver = factory(lctx, ttp_options_, adjust_rank);
          x_desc[lctx->Rank()].status = Beaver::Replay;
          triples[lctx->Rank()] =
              beaver->Mul(kField, kNumel, &x_desc[lctx->Rank()], nullptr);
          yacl::link::Barrier(lctx, "BeaverUT");
        });

    auto open = open_buffer(triples, kField, std::vector<Shape>(3, {kNumel}),
                            kWorldSize, true);

    DISPATCH_ALL_FIELDS(kField, [&]() {
      NdArrayView<ring2k_t> _a(open[0]);
      NdArrayView<ring2k_t> _a_cache(x_cache);
      NdArrayView<ring2k_t> _b(open[1]);
      NdArrayView<ring2k_t> _c(open[2]);
      for (auto idx = 0; idx < _a.numel(); idx++) {
        EXPECT_EQ(_a_cache[idx], _a[idx]);
        auto t = _a[idx] * _b[idx];
        auto err = t > _c[idx] ? t - _c[idx] : _c[idx] - t;
        EXPECT_LE(err, kMaxDiff);
      }
    });
  }
  {
    utils::simulate(
        kWorldSize, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
          auto beaver = factory(lctx, ttp_options_, adjust_rank);
          y_desc[lctx->Rank()].status = Beaver::Replay;
          triples[lctx->Rank()] =
              beaver->Mul(kField, kNumel, nullptr, &y_desc[lctx->Rank()]);
          yacl::link::Barrier(lctx, "BeaverUT");
        });

    auto open = open_buffer(triples, kField, std::vector<Shape>(3, {kNumel}),
                            kWorldSize, true);

    DISPATCH_ALL_FIELDS(kField, [&]() {
      NdArrayView<ring2k_t> _a(open[0]);
      NdArrayView<ring2k_t> _b(open[1]);
      NdArrayView<ring2k_t> _b_cache(y_cache);
      NdArrayView<ring2k_t> _c(open[2]);
      for (auto idx = 0; idx < _a.numel(); idx++) {
        EXPECT_EQ(_b_cache[idx], _b[idx]);
        auto t = _a[idx] * _b[idx];
        auto err = t > _c[idx] ? t - _c[idx] : _c[idx] - t;
        EXPECT_LE(err, kMaxDiff);
      }
    });
  }
  {
    utils::simulate(
        kWorldSize, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
          auto beaver = factory(lctx, ttp_options_, adjust_rank);
          x_desc[lctx->Rank()].status = Beaver::Replay;
          y_desc[lctx->Rank()].status = Beaver::Replay;
          triples[lctx->Rank()] = beaver->Mul(
              kField, kNumel, &x_desc[lctx->Rank()], &y_desc[lctx->Rank()]);
          yacl::link::Barrier(lctx, "BeaverUT");
        });

    auto open = open_buffer(triples, kField, std::vector<Shape>(3, {kNumel}),
                            kWorldSize, true);

    DISPATCH_ALL_FIELDS(kField, [&]() {
      NdArrayView<ring2k_t> _a(open[0]);
      NdArrayView<ring2k_t> _b(open[1]);
      NdArrayView<ring2k_t> _a_cache(x_cache);
      NdArrayView<ring2k_t> _b_cache(y_cache);
      NdArrayView<ring2k_t> _c(open[2]);
      for (auto idx = 0; idx < _a.numel(); idx++) {
        EXPECT_EQ(_a_cache[idx], _a[idx]);
        EXPECT_EQ(_b_cache[idx], _b[idx]);
        auto t = _a[idx] * _b[idx];
        auto err = t > _c[idx] ? t - _c[idx] : _c[idx] - t;
        EXPECT_LE(err, kMaxDiff);
      }
    });
  }
  {
    utils::simulate(
        kWorldSize, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
          auto beaver = factory(lctx, ttp_options_, adjust_rank);
          x_desc[lctx->Rank()].status = Beaver::TransposeReplay;
          y_desc[lctx->Rank()].status = Beaver::TransposeReplay;
          // mul not support transpose.
          triples[lctx->Rank()] = beaver->Mul(
              kField, kNumel, &x_desc[lctx->Rank()], &y_desc[lctx->Rank()]);
          yacl::link::Barrier(lctx, "BeaverUT");
        });

    auto open = open_buffer(triples, kField, std::vector<Shape>(3, {kNumel}),
                            kWorldSize, true);

    DISPATCH_ALL_FIELDS(kField, [&]() {
      NdArrayView<ring2k_t> _a(open[0]);
      NdArrayView<ring2k_t> _b(open[1]);
      NdArrayView<ring2k_t> _a_cache(x_cache);
      NdArrayView<ring2k_t> _b_cache(y_cache);
      NdArrayView<ring2k_t> _c(open[2]);
      for (auto idx = 0; idx < _a.numel(); idx++) {
        // mul not support transpose.
        // enforce ne
        EXPECT_NE(_a_cache[idx], _a[idx]);
        EXPECT_NE(_b_cache[idx], _b[idx]);
        auto t = _a[idx] * _b[idx];
        auto err = t > _c[idx] ? t - _c[idx] : _c[idx] - t;
        EXPECT_LE(err, kMaxDiff);
      }
    });
  }
}

TEST_P(BeaverTest, MulGfmp) {
  const auto factory = std::get<0>(GetParam()).first;
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const int64_t kMaxDiff = std::get<3>(GetParam());
  const size_t adjust_rank = std::get<4>(GetParam());
  const int64_t kNumel = 7;

  std::vector<Triple> triples(kWorldSize);

  std::vector<Beaver::ReplayDesc> x_desc(kWorldSize);
  std::vector<Beaver::ReplayDesc> y_desc(kWorldSize);
  NdArrayRef x_cache;
  NdArrayRef y_cache;
  {
    utils::simulate(
        kWorldSize, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
          auto beaver = factory(lctx, ttp_options_, adjust_rank);
          triples[lctx->Rank()] =
              beaver->Mul(kField, kNumel, &x_desc[lctx->Rank()],
                          &y_desc[lctx->Rank()], ElementType::kGfmp);
          yacl::link::Barrier(lctx, "BeaverUT");
        });

    auto open = open_buffer_gfmp(
        triples, kField, std::vector<Shape>(3, {kNumel}), kWorldSize, true);

    DISPATCH_ALL_FIELDS(kField, [&]() {
      NdArrayView<ring2k_t> _a(open[0]);
      NdArrayView<ring2k_t> _b(open[1]);
      NdArrayView<ring2k_t> _c(open[2]);
      for (auto idx = 0; idx < _a.numel(); idx++) {
        auto prime = ScalarTypeToPrime<ring2k_t>::prime;
        auto t = mul_mod(_a[idx], _b[idx]);
        auto err = t > _c[idx] ? t - _c[idx] : _c[idx] - t;
        auto error_mod_p = static_cast<ring2k_t>(err) % prime;
        EXPECT_LE(error_mod_p, kMaxDiff);
      }
    });

    x_cache = open[0];
    y_cache = open[1];
  }
  {
    utils::simulate(kWorldSize,
                    [&](const std::shared_ptr<yacl::link::Context>& lctx) {
                      auto beaver = factory(lctx, ttp_options_, adjust_rank);
                      x_desc[lctx->Rank()].status = Beaver::Replay;
                      triples[lctx->Rank()] =
                          beaver->Mul(kField, kNumel, &x_desc[lctx->Rank()],
                                      nullptr, ElementType::kGfmp);
                      yacl::link::Barrier(lctx, "BeaverUT");
                    });

    auto open = open_buffer_gfmp(
        triples, kField, std::vector<Shape>(3, {kNumel}), kWorldSize, true);

    DISPATCH_ALL_FIELDS(kField, [&]() {
      NdArrayView<ring2k_t> _a(open[0]);
      NdArrayView<ring2k_t> _a_cache(x_cache);
      NdArrayView<ring2k_t> _b(open[1]);
      NdArrayView<ring2k_t> _c(open[2]);
      for (auto idx = 0; idx < _a.numel(); idx++) {
        auto prime = ScalarTypeToPrime<ring2k_t>::prime;
        EXPECT_EQ(_a_cache[idx], _a[idx]);
        auto t = mul_mod(_a[idx], _b[idx]) % prime;
        auto err = t > _c[idx] ? t - _c[idx] : _c[idx] - t;
        auto error_mod_p = static_cast<ring2k_t>(err) % prime;
        EXPECT_LE(error_mod_p, kMaxDiff);
      }
    });
  }
  {
    utils::simulate(
        kWorldSize, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
          auto beaver = factory(lctx, ttp_options_, adjust_rank);
          y_desc[lctx->Rank()].status = Beaver::Replay;
          triples[lctx->Rank()] =
              beaver->Mul(kField, kNumel, nullptr, &y_desc[lctx->Rank()],
                          ElementType::kGfmp);
          yacl::link::Barrier(lctx, "BeaverUT");
        });

    auto open = open_buffer_gfmp(
        triples, kField, std::vector<Shape>(3, {kNumel}), kWorldSize, true);

    DISPATCH_ALL_FIELDS(kField, [&]() {
      NdArrayView<ring2k_t> _a(open[0]);
      NdArrayView<ring2k_t> _b(open[1]);
      NdArrayView<ring2k_t> _b_cache(y_cache);
      NdArrayView<ring2k_t> _c(open[2]);
      for (auto idx = 0; idx < _a.numel(); idx++) {
        EXPECT_EQ(_b_cache[idx], _b[idx]);
        auto t = mul_mod(_a[idx], _b[idx]);
        auto err = t > _c[idx] ? t - _c[idx] : _c[idx] - t;
        auto prime = ScalarTypeToPrime<ring2k_t>::prime;
        auto error_mod_p = static_cast<ring2k_t>(err) % prime;
        EXPECT_LE(error_mod_p, kMaxDiff);
      }
    });
  }
  {
    utils::simulate(
        kWorldSize, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
          auto beaver = factory(lctx, ttp_options_, adjust_rank);
          x_desc[lctx->Rank()].status = Beaver::Replay;
          y_desc[lctx->Rank()].status = Beaver::Replay;
          triples[lctx->Rank()] =
              beaver->Mul(kField, kNumel, &x_desc[lctx->Rank()],
                          &y_desc[lctx->Rank()], ElementType::kGfmp);
          yacl::link::Barrier(lctx, "BeaverUT");
        });

    auto open = open_buffer_gfmp(
        triples, kField, std::vector<Shape>(3, {kNumel}), kWorldSize, true);

    DISPATCH_ALL_FIELDS(kField, [&]() {
      NdArrayView<ring2k_t> _a(open[0]);
      NdArrayView<ring2k_t> _b(open[1]);
      NdArrayView<ring2k_t> _a_cache(x_cache);
      NdArrayView<ring2k_t> _b_cache(y_cache);
      NdArrayView<ring2k_t> _c(open[2]);
      for (auto idx = 0; idx < _a.numel(); idx++) {
        EXPECT_EQ(_a_cache[idx], _a[idx]);
        EXPECT_EQ(_b_cache[idx], _b[idx]);
        auto t = mul_mod(_a[idx], _b[idx]);
        auto err = t > _c[idx] ? t - _c[idx] : _c[idx] - t;
        auto prime = ScalarTypeToPrime<ring2k_t>::prime;
        auto error_mod_p = static_cast<ring2k_t>(err) % prime;
        EXPECT_LE(error_mod_p, kMaxDiff);
      }
    });
  }
  {
    utils::simulate(
        kWorldSize, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
          auto beaver = factory(lctx, ttp_options_, adjust_rank);
          x_desc[lctx->Rank()].status = Beaver::TransposeReplay;
          y_desc[lctx->Rank()].status = Beaver::TransposeReplay;
          // mul not support transpose.
          triples[lctx->Rank()] =
              beaver->Mul(kField, kNumel, &x_desc[lctx->Rank()],
                          &y_desc[lctx->Rank()], ElementType::kGfmp);
          yacl::link::Barrier(lctx, "BeaverUT");
        });

    auto open = open_buffer_gfmp(
        triples, kField, std::vector<Shape>(3, {kNumel}), kWorldSize, true);

    DISPATCH_ALL_FIELDS(kField, [&]() {
      NdArrayView<ring2k_t> _a(open[0]);
      NdArrayView<ring2k_t> _b(open[1]);
      NdArrayView<ring2k_t> _a_cache(x_cache);
      NdArrayView<ring2k_t> _b_cache(y_cache);
      NdArrayView<ring2k_t> _c(open[2]);
      for (auto idx = 0; idx < _a.numel(); idx++) {
        // mul not support transpose.
        // enforce ne
        EXPECT_NE(_a_cache[idx], _a[idx]);
        EXPECT_NE(_b_cache[idx], _b[idx]);
        auto t = mul_mod(_a[idx], _b[idx]);
        auto err = t > _c[idx] ? t - _c[idx] : _c[idx] - t;
        auto prime = ScalarTypeToPrime<ring2k_t>::prime;
        auto error_mod_p = static_cast<ring2k_t>(err) % prime;
        EXPECT_LE(error_mod_p, kMaxDiff);
      }
    });
  }
}

TEST_P(BeaverTest, And) {
  const auto factory = std::get<0>(GetParam()).first;
  const size_t kWorldSize = std::get<1>(GetParam());
  // const FieldType kField = std::get<2>(GetParam());
  const size_t adjust_rank = std::get<4>(GetParam());
  const int64_t kBytes = 42 + rand() % 31;

  std::vector<Triple> triples(kWorldSize);

  utils::simulate(kWorldSize,
                  [&](const std::shared_ptr<yacl::link::Context>& lctx) {
                    auto beaver = factory(lctx, ttp_options_, adjust_rank);
                    triples[lctx->Rank()] = beaver->And(kBytes);
                    yacl::link::Barrier(lctx, "BeaverUT");
                  });

  std::vector<uint8_t> open_a(kBytes);
  std::vector<uint8_t> open_b(kBytes);
  std::vector<uint8_t> open_c(kBytes);

  for (size_t r = 0; r < kWorldSize; r++) {
    const auto& [a, b, c] = triples[r];
    EXPECT_EQ(kBytes, a.size());
    EXPECT_EQ(kBytes, b.size());
    EXPECT_EQ(kBytes, c.size());
    for (int64_t i = 0; i < kBytes; i++) {
      open_a[i] ^= a.data<uint8_t>()[i];
      open_b[i] ^= b.data<uint8_t>()[i];
      open_c[i] ^= c.data<uint8_t>()[i];
    }
  }
  for (int64_t i = 0; i < kBytes; i++) {
    EXPECT_EQ(open_c[i], open_a[i] & open_b[i]);
  }
}

TEST_P(BeaverTest, Dot) {
  const auto factory = std::get<0>(GetParam()).first;
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const int64_t kMaxDiff = std::get<3>(GetParam());
  const size_t adjust_rank = std::get<4>(GetParam());
  // M > N
  const int64_t M = 17;
  const int64_t N = 8;
  const int64_t K = 1024;

  std::vector<Triple> triples(kWorldSize);

  std::vector<Beaver::ReplayDesc> x_desc(kWorldSize);
  std::vector<Beaver::ReplayDesc> y_desc(kWorldSize);
  NdArrayRef x_cache;
  NdArrayRef y_cache;

  {
    utils::simulate(
        kWorldSize, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
          auto beaver = factory(lctx, ttp_options_, adjust_rank);
          triples[lctx->Rank()] = beaver->Dot(
              kField, M, N, K, &x_desc[lctx->Rank()], &y_desc[lctx->Rank()]);
          yacl::link::Barrier(lctx, "BeaverUT");
        });

    EXPECT_EQ(triples.size(), kWorldSize);
    auto open = open_buffer(triples, kField, {{M, K}, {K, N}, {M, N}},
                            kWorldSize, true);

    auto res = ring_mmul(open[0], open[1]);
    DISPATCH_ALL_FIELDS(kField, [&]() {
      NdArrayView<ring2k_t> _r(res);
      NdArrayView<ring2k_t> _c(open[2]);
      for (auto idx = 0; idx < res.numel(); idx++) {
        auto err = _r[idx] > _c[idx] ? _r[idx] - _c[idx] : _c[idx] - _r[idx];
        EXPECT_LE(err, kMaxDiff);
      }
    });
    x_cache = open[0];
    y_cache = open[1];
  }
  {
    utils::simulate(
        kWorldSize, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
          auto beaver = factory(lctx, ttp_options_, adjust_rank);
          x_desc[lctx->Rank()].status = Beaver::Replay;
          triples[lctx->Rank()] =
              beaver->Dot(kField, M, N, K, &x_desc[lctx->Rank()], nullptr);
          yacl::link::Barrier(lctx, "BeaverUT");
        });

    EXPECT_EQ(triples.size(), kWorldSize);
    auto open = open_buffer(triples, kField, {{M, K}, {K, N}, {M, N}},
                            kWorldSize, true);

    auto res = ring_mmul(x_cache, open[1]);
    DISPATCH_ALL_FIELDS(kField, [&]() {
      NdArrayView<ring2k_t> _a(open[0]);
      NdArrayView<ring2k_t> _a_cache(x_cache);
      NdArrayView<ring2k_t> _r(res);
      NdArrayView<ring2k_t> _c(open[2]);
      for (auto idx = 0; idx < res.numel(); idx++) {
        EXPECT_EQ(_a_cache[idx], _a[idx]);
        auto err = _r[idx] > _c[idx] ? _r[idx] - _c[idx] : _c[idx] - _r[idx];
        EXPECT_LE(err, kMaxDiff);
      }
    });
  }
  {
    utils::simulate(
        kWorldSize, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
          auto beaver = factory(lctx, ttp_options_, adjust_rank);
          y_desc[lctx->Rank()].status = Beaver::Replay;
          triples[lctx->Rank()] =
              beaver->Dot(kField, M, N, K, nullptr, &y_desc[lctx->Rank()]);
          yacl::link::Barrier(lctx, "BeaverUT");
        });

    EXPECT_EQ(triples.size(), kWorldSize);
    auto open = open_buffer(triples, kField, {{M, K}, {K, N}, {M, N}},
                            kWorldSize, true);

    auto res = ring_mmul(open[0], y_cache);
    DISPATCH_ALL_FIELDS(kField, [&]() {
      NdArrayView<ring2k_t> _b(open[1]);
      NdArrayView<ring2k_t> _b_cache(y_cache);
      NdArrayView<ring2k_t> _r(res);
      NdArrayView<ring2k_t> _c(open[2]);
      for (auto idx = 0; idx < res.numel(); idx++) {
        EXPECT_EQ(_b_cache[idx], _b[idx]);
        auto err = _r[idx] > _c[idx] ? _r[idx] - _c[idx] : _c[idx] - _r[idx];
        EXPECT_LE(err, kMaxDiff);
      }
    });
  }
  {
    utils::simulate(
        kWorldSize, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
          auto beaver = factory(lctx, ttp_options_, adjust_rank);
          x_desc[lctx->Rank()].status = Beaver::Replay;
          y_desc[lctx->Rank()].status = Beaver::Replay;
          triples[lctx->Rank()] = beaver->Dot(
              kField, M, N, K, &x_desc[lctx->Rank()], &y_desc[lctx->Rank()]);
          yacl::link::Barrier(lctx, "BeaverUT");
        });

    EXPECT_EQ(triples.size(), kWorldSize);
    auto open = open_buffer(triples, kField, {{M, K}, {K, N}, {M, N}},
                            kWorldSize, true);

    auto res = ring_mmul(x_cache, y_cache);
    DISPATCH_ALL_FIELDS(kField, [&]() {
      NdArrayView<ring2k_t> _a(open[0]);
      NdArrayView<ring2k_t> _a_cache(x_cache);
      NdArrayView<ring2k_t> _b(open[1]);
      NdArrayView<ring2k_t> _b_cache(y_cache);
      NdArrayView<ring2k_t> _r(res);
      NdArrayView<ring2k_t> _c(open[2]);
      for (auto idx = 0; idx < res.numel(); idx++) {
        EXPECT_EQ(_a_cache[idx], _a[idx]);
        EXPECT_EQ(_b_cache[idx], _b[idx]);
        auto err = _r[idx] > _c[idx] ? _r[idx] - _c[idx] : _c[idx] - _r[idx];
        EXPECT_LE(err, kMaxDiff);
      }
    });
  }
  {
    utils::simulate(
        kWorldSize, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
          auto beaver = factory(lctx, ttp_options_, adjust_rank);
          x_desc[lctx->Rank()].status = Beaver::TransposeReplay;
          y_desc[lctx->Rank()].status = Beaver::TransposeReplay;
          triples[lctx->Rank()] = beaver->Dot(
              kField, N, M, K, &y_desc[lctx->Rank()], &x_desc[lctx->Rank()]);
          yacl::link::Barrier(lctx, "BeaverUT");
        });

    EXPECT_EQ(triples.size(), kWorldSize);
    auto open = open_buffer(triples, kField, {{N, K}, {K, M}, {N, M}},
                            kWorldSize, true);

    auto res = ring_mmul(x_cache, y_cache);
    DISPATCH_ALL_FIELDS(kField, [&]() {
      auto transpose_a = open[0].transpose();
      NdArrayView<ring2k_t> _a(transpose_a);
      NdArrayView<ring2k_t> _a_cache(y_cache);
      auto transpose_b = open[1].transpose();
      NdArrayView<ring2k_t> _b(transpose_b);
      NdArrayView<ring2k_t> _b_cache(x_cache);
      auto transpose_r = res.transpose();
      NdArrayView<ring2k_t> _r(transpose_r);
      NdArrayView<ring2k_t> _c(open[2]);
      for (auto idx = 0; idx < res.numel(); idx++) {
        EXPECT_EQ(_a_cache[idx], _a[idx]);
        EXPECT_EQ(_b_cache[idx], _b[idx]);
        auto err = _r[idx] > _c[idx] ? _r[idx] - _c[idx] : _c[idx] - _r[idx];
        EXPECT_LE(err, kMaxDiff);
      }
    });
  }
  {
    utils::simulate(
        kWorldSize, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
          auto beaver = factory(lctx, ttp_options_, adjust_rank);
          x_desc[lctx->Rank()].status = Beaver::Replay;
          // replay by different func.
          triples[lctx->Rank()] =
              beaver->Mul(kField, M * K, &x_desc[lctx->Rank()], nullptr);
          yacl::link::Barrier(lctx, "BeaverUT");
        });

    EXPECT_EQ(triples.size(), kWorldSize);
    auto open = open_buffer(triples, kField, std::vector<Shape>(3, {M * K}),
                            kWorldSize, true);

    DISPATCH_ALL_FIELDS(kField, [&]() {
      NdArrayView<ring2k_t> _a(open[0]);
      NdArrayView<ring2k_t> _a_cache(x_cache);
      NdArrayView<ring2k_t> _b(open[1]);
      NdArrayView<ring2k_t> _c(open[2]);
      for (auto idx = 0; idx < _a.numel(); idx++) {
        EXPECT_EQ(_a_cache[idx], _a[idx]);
        auto t = _a[idx] * _b[idx];
        auto err = t > _c[idx] ? t - _c[idx] : _c[idx] - t;
        EXPECT_LE(err, kMaxDiff);
      }
    });
  }
}

TEST_P(BeaverTest, Dot_large) {
  const auto factory = std::get<0>(GetParam()).first;
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const int64_t kMaxDiff = std::get<3>(GetParam());
  const size_t adjust_rank = std::get<4>(GetParam());
  // M < N
  const int64_t M = 11;
  const int64_t N = 20;
  const int64_t K = 1023;

  std::vector<Triple> triples;
  triples.resize(kWorldSize);

  utils::simulate(kWorldSize,
                  [&](const std::shared_ptr<yacl::link::Context>& lctx) {
                    auto beaver = factory(lctx, ttp_options_, adjust_rank);
                    triples[lctx->Rank()] = beaver->Dot(kField, M, N, K);
                    yacl::link::Barrier(lctx, "BeaverUT");
                  });

  EXPECT_EQ(triples.size(), kWorldSize);
  auto open =
      open_buffer(triples, kField, {{M, K}, {K, N}, {M, N}}, kWorldSize, true);

  auto res = ring_mmul(open[0], open[1]);
  DISPATCH_ALL_FIELDS(kField, [&]() {
    NdArrayView<ring2k_t> _r(res);
    NdArrayView<ring2k_t> _c(open[2]);
    for (auto idx = 0; idx < res.numel(); idx++) {
      auto err = _r[idx] > _c[idx] ? _r[idx] - _c[idx] : _c[idx] - _r[idx];
      EXPECT_LE(err, kMaxDiff);
    }
  });
}

TEST_P(BeaverTest, Trunc) {
  const auto factory = std::get<0>(GetParam()).first;
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const size_t adjust_rank = std::get<4>(GetParam());
  const int64_t kNumel = 7;
  const int64_t kBits = 5;

  std::vector<Pair> pairs;
  pairs.resize(kWorldSize);

  utils::simulate(kWorldSize,
                  [&](const std::shared_ptr<yacl::link::Context>& lctx) {
                    auto beaver = factory(lctx, ttp_options_, adjust_rank);
                    pairs[lctx->Rank()] = beaver->Trunc(kField, kNumel, kBits);
                    yacl::link::Barrier(lctx, "BeaverUT");
                  });

  EXPECT_EQ(pairs.size(), kWorldSize);
  auto open =
      open_buffer(pairs, kField, {{kNumel}, {kNumel}}, kWorldSize, true);
  EXPECT_TRUE(ring_all_equal(ring_arshift(open[0], {kBits}), open[1], 0));
}

TEST_P(BeaverTest, TruncPr) {
  const auto factory = std::get<0>(GetParam()).first;
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const size_t adjust_rank = std::get<4>(GetParam());
  const int64_t kNumel = 7;
  const size_t kBits = 5;
  const size_t kRingSize = SizeOf(kField) * 8;

  std::vector<Triple> rets;
  rets.resize(kWorldSize);

  utils::simulate(kWorldSize,
                  [&](const std::shared_ptr<yacl::link::Context>& lctx) {
                    auto beaver = factory(lctx, ttp_options_, adjust_rank);
                    rets[lctx->Rank()] = beaver->TruncPr(kField, kNumel, kBits);
                    yacl::link::Barrier(lctx, "BeaverUT");
                  });

  EXPECT_EQ(rets.size(), kWorldSize);
  auto open = open_buffer(rets, kField, std::vector<Shape>(3, {kNumel}),
                          kWorldSize, true);

  DISPATCH_ALL_FIELDS(kField, [&]() {
    using T = ring2k_t;
    auto sum_r_iter = open[0].begin();
    auto sum_rc_iter = open[1].begin();
    auto sum_rb_iter = open[2].begin();

    for (int64_t i = 0; i < open[0].numel();
         ++i, ++sum_r_iter, ++sum_rc_iter, ++sum_rb_iter) {
      auto r = sum_r_iter.getScalarValue<T>();
      auto rc = sum_rc_iter.getScalarValue<T>();
      auto rb = sum_rb_iter.getScalarValue<T>();

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
  const size_t adjust_rank = std::get<4>(GetParam());
  const int64_t kNumel = 51;

  std::vector<Beaver::Array> shares(kWorldSize);

  utils::simulate(kWorldSize,
                  [&](const std::shared_ptr<yacl::link::Context>& lctx) {
                    auto beaver = factory(lctx, ttp_options_, adjust_rank);
                    shares[lctx->Rank()] = beaver->RandBit(kField, kNumel);
                    yacl::link::Barrier(lctx, "BeaverUT");
                  });

  EXPECT_EQ(shares.size(), kWorldSize);
  auto open = open_buffer(shares, kField, {{kNumel}}, kWorldSize, true);

  DISPATCH_ALL_FIELDS(kField, [&]() {
    using scalar_t = typename Ring2kTrait<_kField>::scalar_t;
    auto x = xt_adapt<scalar_t>(open[0]);
    EXPECT_TRUE(xt::all(x <= xt::ones_like(x)));
    EXPECT_TRUE(xt::all(x >= xt::zeros_like(x)));
    EXPECT_TRUE(x != xt::zeros_like(x));
    return;
  });
}

TEST_P(BeaverTest, Eqz) {
  const auto factory = std::get<0>(GetParam()).first;
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const size_t adjust_rank = std::get<4>(GetParam());
  const int64_t kNumel = 2;

  std::vector<Pair> pairs;
  pairs.resize(kWorldSize);
  utils::simulate(kWorldSize,
                  [&](const std::shared_ptr<yacl::link::Context>& lctx) {
                    auto beaver = factory(lctx, ttp_options_, adjust_rank);
                    pairs[lctx->Rank()] = beaver->Eqz(kField, kNumel);
                    yacl::link::Barrier(lctx, "BeaverUT");
                  });
  EXPECT_EQ(pairs.size(), kWorldSize);
  auto sum_a = ring_zeros(kField, {kNumel});
  auto sum_b = ring_zeros(kField, {kNumel});
  for (Rank r = 0; r < kWorldSize; r++) {
    auto [a_buf, b_buf] = pairs[r];
    NdArrayRef a(std::make_shared<yacl::Buffer>(std::move(a_buf)),
                 sum_a.eltype(), sum_a.shape());
    NdArrayRef b(std::make_shared<yacl::Buffer>(std::move(b_buf)),
                 sum_b.eltype(), sum_b.shape());
    EXPECT_EQ(a.numel(), kNumel);
    EXPECT_EQ(b.numel(), kNumel);

    ring_add_(sum_a, a);
    ring_xor_(sum_b, b);
  }
  EXPECT_TRUE(ring_all_equal(sum_a, sum_b));
}

TEST_P(BeaverTest, PermPair) {
  const auto factory = std::get<0>(GetParam()).first;
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const size_t adjust_rank = std::get<4>(GetParam());
  const int64_t kNumel = 10;
  std::random_device rd;
  uint128_t seed = rd();
  uint64_t ctr = rd();
  const auto r_perm = genRandomPerm(kNumel, seed, &ctr);

  for (size_t r = 0; r < kWorldSize; ++r) {
    std::vector<Pair> pairs(kWorldSize);
    utils::simulate(
        kWorldSize, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
          auto beaver = factory(lctx, ttp_options_, adjust_rank);
          auto rank = lctx->Rank();
          if (rank == r) {
            pairs[lctx->Rank()] = beaver->PermPair(kField, kNumel, r, r_perm);
          } else {
            pairs[lctx->Rank()] = beaver->PermPair(kField, kNumel, r, {});
          }
          yacl::link::Barrier(lctx, "BeaverUT");
        });

    EXPECT_EQ(pairs.size(), kWorldSize);
    auto open = open_buffer(pairs, kField, std::vector<Shape>(2, {kNumel}),
                            kWorldSize, true);
    EXPECT_TRUE(ring_all_equal(applyInvPerm(open[0], r_perm), open[1], 0));
  }
}

}  // namespace spu::mpc::semi2k
