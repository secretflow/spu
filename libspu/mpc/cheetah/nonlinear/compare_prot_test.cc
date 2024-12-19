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

#include "libspu/mpc/cheetah/nonlinear/compare_prot.h"

#include "gtest/gtest.h"

#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah {

class CompareProtTest
    : public ::testing::TestWithParam<std::tuple<FieldType, bool, size_t>> {
  void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, CompareProtTest,
    testing::Combine(testing::Values(FieldType::FM32, FieldType::FM64),
                     testing::Values(true, false),
                     testing::Values(1UL, 4UL, 8UL)),  // divide k
    [](const testing::TestParamInfo<CompareProtTest::ParamType> &p) {
      return fmt::format("{}Radix{}Greater{}", std::get<0>(p.param),
                         (int)std::get<2>(p.param), (int)std::get<1>(p.param));
    });

TEST_P(CompareProtTest, Compare) {
  size_t kWorldSize = 2;
  Shape shape = {13, 2, 3};
  FieldType field = std::get<0>(GetParam());
  size_t radix = std::get<2>(GetParam());
  bool greater_than = std::get<1>(GetParam());

  NdArrayRef inp[2];
  inp[0] = ring_rand(field, shape);
  inp[1] = ring_rand(field, shape);

  DISPATCH_ALL_FIELDS(field, [&]() {
    auto xinp = NdArrayView<ring2k_t>(inp[0]);
    xinp[0] = 1;
    xinp[1] = 10;
    xinp[2] = 100;

    xinp = NdArrayView<ring2k_t>(inp[1]);
    xinp[0] = 1;
    xinp[1] = 9;
    if constexpr (std::is_same_v<ring2k_t, uint8_t>) {
      xinp[2] = 100;
    } else {
      xinp[2] = 1000;
    }
  });

  NdArrayRef cmp_oup[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    auto conn = std::make_shared<Communicator>(ctx);
    int rank = ctx->Rank();
    auto base = std::make_shared<BasicOTProtocols>(
        conn, CheetahOtKind::YACL_Softspoken);
    CompareProtocol comp_prot(base, radix);
    auto _c = comp_prot.Compute(inp[rank], greater_than);
    cmp_oup[rank] = _c;
  });

  DISPATCH_ALL_FIELDS(field, [&]() {
    auto xout0 = NdArrayView<ring2k_t>(cmp_oup[0]);
    auto xout1 = NdArrayView<ring2k_t>(cmp_oup[1]);
    auto xinp0 = NdArrayView<ring2k_t>(inp[0]);
    auto xinp1 = NdArrayView<ring2k_t>(inp[1]);

    for (int64_t i = 0; i < shape.numel(); ++i) {
      bool expected = greater_than ? xinp0[i] > xinp1[i] : xinp0[i] < xinp1[i];
      bool got_cmp = xout0[i] ^ xout1[i];
      ASSERT_EQ(expected, got_cmp);
    }
  });
}

TEST_P(CompareProtTest, CompareBitWidth) {
  size_t kWorldSize = 2;
  FieldType field = std::get<0>(GetParam());
  size_t radix = std::get<2>(GetParam());
  bool greater_than = std::get<1>(GetParam());
  int64_t bw = std::min<int>(32, SizeOf(field) * 8);

  NdArrayRef inp[2];
  int64_t n = 100;
  inp[0] = ring_rand(field, {n, 2});
  inp[1] = ring_rand(field, {n, 2});

  DISPATCH_ALL_FIELDS(field, [&]() {
    ring2k_t mask = (static_cast<ring2k_t>(1) << bw) - 1;
    auto xinp = NdArrayView<ring2k_t>(inp[0]);
    xinp[0] = 1;
    xinp[1] = 10;
    xinp[2] = 100;
    pforeach(0, inp[0].numel(), [&](int64_t i) { xinp[i] &= mask; });

    xinp = NdArrayView<ring2k_t>(inp[1]);
    xinp[0] = 1;
    xinp[1] = 9;
    if constexpr (std::is_same_v<ring2k_t, uint8_t>) {
      xinp[2] = 100;
    } else {
      xinp[2] = 1000;
    }
    pforeach(0, inp[0].numel(), [&](int64_t i) { xinp[i] &= mask; });
  });

  inp[0] = inp[0].reshape({n, 2});
  inp[1] = inp[1].reshape({n, 2});

  NdArrayRef cmp_oup[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    auto conn = std::make_shared<Communicator>(ctx);
    int rank = ctx->Rank();
    auto base = std::make_shared<BasicOTProtocols>(
        conn, CheetahOtKind::YACL_Softspoken);

    CompareProtocol comp_prot(base, radix);

    [[maybe_unused]] auto b0 = ctx->GetStats()->sent_bytes.load();
    [[maybe_unused]] auto s0 = ctx->GetStats()->sent_actions.load();

    auto _c = comp_prot.Compute(inp[rank], greater_than, bw);

    [[maybe_unused]] auto b1 = ctx->GetStats()->sent_bytes.load();
    [[maybe_unused]] auto s1 = ctx->GetStats()->sent_actions.load();

    SPDLOG_DEBUG(
        "Compare {} bits {} elements sent {} bytes, {} bits each #sent {}", bw,
        inp[0].numel(), (b1 - b0), (b1 - b0) * 8. / inp[0].numel(), (s1 - s0));

    cmp_oup[rank] = _c;
  });

  DISPATCH_ALL_FIELDS(field, [&]() {
    auto xout0 = NdArrayView<ring2k_t>(cmp_oup[0]);
    auto xout1 = NdArrayView<ring2k_t>(cmp_oup[1]);
    auto xinp0 = NdArrayView<ring2k_t>(inp[0]);
    auto xinp1 = NdArrayView<ring2k_t>(inp[1]);

    for (int64_t i = 0; i < inp[0].numel(); ++i) {
      bool expected = greater_than ? xinp0[i] > xinp1[i] : xinp0[i] < xinp1[i];
      bool got_cmp = xout0[i] ^ xout1[i];
      EXPECT_EQ(expected, got_cmp);
    }
  });
}

TEST_P(CompareProtTest, WithEq) {
  size_t kWorldSize = 2;
  Shape shape = {10, 10, 10};
  FieldType field = std::get<0>(GetParam());
  size_t radix = std::get<2>(GetParam());
  bool greater_than = std::get<1>(GetParam());

  NdArrayRef _inp[2];
  _inp[0] = ring_rand(field, shape);
  _inp[1] = ring_rand(field, shape);

  NdArrayRef inp[2];
  inp[0] = _inp[0].slice({0, 0, 0}, {10, 10, 10}, {2, 3, 2});
  inp[1] = _inp[0].slice({0, 0, 0}, {10, 10, 10}, {2, 3, 2});
  shape = inp[0].shape();

  DISPATCH_ALL_FIELDS(field, [&]() {
    auto xinp = NdArrayView<ring2k_t>(inp[0]);
    xinp[0] = 1;
    xinp[1] = 10;
    xinp[2] = 100;

    xinp = NdArrayView<ring2k_t>(inp[1]);
    xinp[0] = 1;
    xinp[1] = 9;
    if constexpr (std::is_same_v<ring2k_t, uint8_t>) {
      xinp[2] = 100;
    } else {
      xinp[2] = 1000;
    }
  });

  NdArrayRef cmp_oup[2];
  NdArrayRef eq_oup[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    auto conn = std::make_shared<Communicator>(ctx);
    int rank = ctx->Rank();
    auto base = std::make_shared<BasicOTProtocols>(
        conn, CheetahOtKind::YACL_Softspoken);
    CompareProtocol comp_prot(base, radix);
    auto [_c, _e] = comp_prot.ComputeWithEq(inp[rank], greater_than);
    cmp_oup[rank] = _c;
    eq_oup[rank] = _e;
  });

  DISPATCH_ALL_FIELDS(field, [&]() {
    auto xout0 = NdArrayView<ring2k_t>(cmp_oup[0]);
    auto xout1 = NdArrayView<ring2k_t>(cmp_oup[1]);
    auto xeq0 = NdArrayView<ring2k_t>(eq_oup[0]);
    auto xeq1 = NdArrayView<ring2k_t>(eq_oup[1]);
    auto xinp0 = NdArrayView<ring2k_t>(inp[0]);
    auto xinp1 = NdArrayView<ring2k_t>(inp[1]);

    for (int64_t i = 0; i < shape.numel(); ++i) {
      bool expected = greater_than ? xinp0[i] > xinp1[i] : xinp0[i] < xinp1[i];
      bool got_cmp = xout0[i] ^ xout1[i];
      bool got_eq = xeq0[i] ^ xeq1[i];
      EXPECT_EQ(expected, got_cmp);
      EXPECT_EQ((xinp0[i] == xinp1[i]), got_eq);
    }
  });
}

TEST_P(CompareProtTest, WithEqBitWidth) {
  size_t kWorldSize = 2;
  Shape shape = {10, 10, 10};
  FieldType field = std::get<0>(GetParam());
  size_t radix = std::get<2>(GetParam());
  bool greater_than = std::get<1>(GetParam());

  int64_t bw = std::min<int>(32, SizeOf(field) * 8);

  NdArrayRef inp[2];
  int64_t n = 1 << 10;
  inp[0] = ring_rand(field, {n, 2});
  inp[1] = ring_rand(field, {n, 2});

  DISPATCH_ALL_FIELDS(field, [&]() {
    ring2k_t mask = (static_cast<ring2k_t>(1) << bw) - 1;
    auto xinp = NdArrayView<ring2k_t>(inp[0]);
    xinp[0] = 1;
    xinp[1] = 10;
    xinp[2] = 100;
    pforeach(0, inp[0].numel(), [&](int64_t i) { xinp[i] &= mask; });

    xinp = NdArrayView<ring2k_t>(inp[1]);
    xinp[0] = 1;
    xinp[1] = 9;
    if constexpr (std::is_same_v<ring2k_t, uint8_t>) {
      xinp[2] = 100;
    } else {
      xinp[2] = 1000;
    }
    pforeach(0, inp[0].numel(), [&](int64_t i) { xinp[i] &= mask; });
  });

  NdArrayRef cmp_oup[2];
  NdArrayRef eq_oup[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    auto conn = std::make_shared<Communicator>(ctx);
    int rank = ctx->Rank();
    auto base = std::make_shared<BasicOTProtocols>(
        conn, CheetahOtKind::YACL_Softspoken);

    [[maybe_unused]] auto b0 = ctx->GetStats()->sent_bytes.load();
    [[maybe_unused]] auto s0 = ctx->GetStats()->sent_actions.load();

    CompareProtocol comp_prot(base, radix);
    auto [_c, _e] = comp_prot.ComputeWithEq(inp[rank], greater_than, bw);

    [[maybe_unused]] auto b1 = ctx->GetStats()->sent_bytes.load();
    [[maybe_unused]] auto s1 = ctx->GetStats()->sent_actions.load();

    SPDLOG_DEBUG(
        "CompareWithEq {} bits {} elements sent {} bytes, {} bits each #sent "
        "{}",
        bw, inp[0].numel(), (b1 - b0), (b1 - b0) * 8. / inp[0].numel(),
        (s1 - s0));

    cmp_oup[rank] = _c;
    eq_oup[rank] = _e;
  });

  DISPATCH_ALL_FIELDS(field, [&]() {
    auto xout0 = NdArrayView<ring2k_t>(cmp_oup[0]);
    auto xout1 = NdArrayView<ring2k_t>(cmp_oup[1]);
    auto xeq0 = NdArrayView<ring2k_t>(eq_oup[0]);
    auto xeq1 = NdArrayView<ring2k_t>(eq_oup[1]);
    auto xinp0 = NdArrayView<ring2k_t>(inp[0]);
    auto xinp1 = NdArrayView<ring2k_t>(inp[1]);

    for (int64_t i = 0; i < shape.numel(); ++i) {
      bool expected = greater_than ? xinp0[i] > xinp1[i] : xinp0[i] < xinp1[i];
      bool got_cmp = xout0[i] ^ xout1[i];
      bool got_eq = xeq0[i] ^ xeq1[i];
      EXPECT_EQ(expected, got_cmp);
      EXPECT_EQ((xinp0[i] == xinp1[i]), got_eq);
    }
  });
}
}  // namespace spu::mpc::cheetah
