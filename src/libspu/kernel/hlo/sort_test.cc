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

#include "libspu/kernel/hlo/sort.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <tuple>
#include <vector>
#include <xtensor/xadapt.hpp>
#include <xtensor/xsort.hpp>

#include "gtest/gtest.h"
#include "magic_enum.hpp"
#include "xtensor/xio.hpp"

#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/hlo/casting.h"
#include "libspu/kernel/hlo/const.h"
#include "libspu/kernel/test_util.h"
#include "libspu/mpc/utils/simulate.h"
// to print method name
std::ostream &operator<<(std::ostream &os,
                         spu::RuntimeConfig::SortMethod method) {
  os << magic_enum::enum_name(method);
  return os;
}
namespace spu::kernel::hlo {

TEST(SortTest, Simple) {
  SPUContext ctx = test::makeSPUContext();
  xt::xarray<float> x = {{0.05, 0.5, 0.24}, {5, 50, 2}};
  xt::xarray<float> sorted_x = {{0.05, 0.24, 0.5}, {2, 5, 50}};

  Value x_v = test::makeValue(&ctx, x, VIS_SECRET);

  std::vector<spu::Value> rets = Sort(
      &ctx, {x_v}, 1, false,
      [&](absl::Span<const spu::Value> inputs) {
        return hal::less(&ctx, inputs[0], inputs[1]);
      },
      Visibility::VIS_SECRET);

  EXPECT_EQ(rets.size(), 1);

  auto sorted_x_hat =
      hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, rets[0]));

  EXPECT_TRUE(xt::allclose(sorted_x, sorted_x_hat, 0.01, 0.001))
      << sorted_x << std::endl
      << sorted_x_hat << std::endl;
}

TEST(SortTest, SimpleWithNoPadding) {
  SPUContext ctx = test::makeSPUContext();
  xt::xarray<float> x = {{0.05, 0.5, 0.24, 0.5}, {2, 5, 50, 2}};
  xt::xarray<float> sorted_x = {{0.05, 0.24, 0.5, 0.5}, {2, 2, 5, 50}};

  Value x_v = test::makeValue(&ctx, x, VIS_SECRET);

  std::vector<spu::Value> rets = Sort(
      &ctx, {x_v}, 1, false,
      [&](absl::Span<const spu::Value> inputs) {
        return hal::less(&ctx, inputs[0], inputs[1]);
      },
      Visibility::VIS_SECRET);

  EXPECT_EQ(rets.size(), 1);

  auto sorted_x_hat =
      hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, rets[0]));

  EXPECT_TRUE(xt::allclose(sorted_x, sorted_x_hat, 0.01, 0.001))
      << sorted_x << std::endl
      << sorted_x_hat << std::endl;
}

TEST(SortTest, MultiInputs) {
  SPUContext ctx = test::makeSPUContext();
  xt::xarray<float> x1 = {{0.5, 0.05, 0.5, 0.24, 0.5, 0.5, 0.5}};
  xt::xarray<float> x2 = {{5, 1, 2, 1, 2, 3, 4}};
  xt::xarray<float> sorted_x1 = {{0.05, 0.24, 0.5, 0.5, 0.5, 0.5, 0.5}};
  xt::xarray<float> sorted_x2 = {{1, 1, 2, 2, 3, 4, 5}};

  Value x1_v = test::makeValue(&ctx, x1, VIS_SECRET);
  Value x2_v = test::makeValue(&ctx, x2, VIS_SECRET);

  std::vector<spu::Value> rets = Sort(
      &ctx, {x1_v, x2_v}, 1, false,
      [&](absl::Span<const spu::Value> inputs) {
        return hal::less(&ctx, inputs[0], inputs[1]);
      },
      Visibility::VIS_SECRET);

  EXPECT_EQ(rets.size(), 2);

  auto sorted_x1_hat =
      hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, rets[0]));

  EXPECT_TRUE(xt::allclose(sorted_x1, sorted_x1_hat, 0.01, 0.001))
      << sorted_x1 << std::endl
      << sorted_x1_hat << std::endl;

  // NOTE: Secret sort is unstable, so rets[1] need to be sort before check.
  auto sorted_x2_hat =
      xt::sort(hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, rets[1])));

  EXPECT_TRUE(xt::allclose(sorted_x2, sorted_x2_hat, 0.01, 0.001))
      << sorted_x2 << std::endl
      << sorted_x2_hat << std::endl;
}

TEST(SortTest, MultiOperands) {
  SPUContext ctx = test::makeSPUContext();
  xt::xarray<float> k1 = {6, 6, 3, 4, 4, 5, 4};
  xt::xarray<float> k2 = {0.5, 0.1, 3.1, 6.5, 4.1, 6.7, 2.5};

  xt::xarray<float> sorted_k1 = {3, 4, 4, 4, 5, 6, 6};
  xt::xarray<float> sorted_k2 = {3.1, 2.5, 4.1, 6.5, 6.7, 0.1, 0.5};

  Value k1_v = test::makeValue(&ctx, k1, VIS_SECRET);
  Value k2_v = test::makeValue(&ctx, k2, VIS_SECRET);

  std::vector<spu::Value> rets = Sort(
      &ctx, {k1_v, k2_v}, 0, false,
      [&](absl::Span<const spu::Value> inputs) {
        auto pred_0 = hal::equal(&ctx, inputs[0], inputs[1]);
        auto pred_1 = hal::less(&ctx, inputs[0], inputs[1]);
        auto pred_2 = hal::less(&ctx, inputs[2], inputs[3]);

        return hal::select(&ctx, pred_0, pred_2, pred_1);
      },
      Visibility::VIS_SECRET);

  EXPECT_EQ(rets.size(), 2);

  auto sorted_k1_hat =
      hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, rets[0]));
  auto sorted_k2_hat =
      hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, rets[1]));

  EXPECT_TRUE(xt::allclose(sorted_k1, sorted_k1_hat, 0.01, 0.001))
      << sorted_k1 << std::endl
      << sorted_k1_hat << std::endl;

  EXPECT_TRUE(xt::allclose(sorted_k2, sorted_k2_hat, 0.01, 0.001))
      << sorted_k2 << std::endl
      << sorted_k2_hat << std::endl;
}

TEST(SortTest, EmptyOperands) {
  SPUContext ctx = test::makeSPUContext();
  auto empty_x = Seal(&ctx, Constant(&ctx, 1, {0}));

  std::vector<spu::Value> rets = Sort(
      &ctx, {empty_x}, 0, false,
      [&](absl::Span<const spu::Value> inputs) {
        return hal::less(&ctx, inputs[0], inputs[1]);
      },
      Visibility::VIS_SECRET);

  EXPECT_EQ(rets.size(), 1);
  EXPECT_EQ(rets[0].numel(), 0);
  EXPECT_EQ(rets[0].shape().size(), 1);
  EXPECT_EQ(rets[0].shape()[0], 0);
}

TEST(SortTest, LargeNumel) {
  SPUContext ctx = test::makeSPUContext();

  std::vector<std::size_t> numels = {63, 64, 65, 149, 170, 255, 256, 257, 500};

  for (auto numel : numels) {
    std::vector<int64_t> asc_arr(numel);
    std::vector<std::size_t> shape = {numel};
    std::iota(asc_arr.begin(), asc_arr.end(), 0);

    auto shuffled_arr = asc_arr;
    std::random_device rd;
    std::mt19937 rng(rd());
    std::shuffle(shuffled_arr.begin(), shuffled_arr.end(), rng);

    auto des_arr = asc_arr;
    std::reverse(des_arr.begin(), des_arr.end());

    auto x = xt::adapt(shuffled_arr, shape);
    auto asc_x = xt::adapt(asc_arr, shape);
    auto des_x = xt::adapt(des_arr, shape);

    Value x_v = test::makeValue(&ctx, x, VIS_SECRET);

    // ascending sort
    std::vector<spu::Value> rets = Sort(
        &ctx, {x_v}, 0, false,
        [&](absl::Span<const spu::Value> inputs) {
          return hal::less(&ctx, inputs[0], inputs[1]);
        },
        Visibility::VIS_SECRET);

    EXPECT_EQ(rets.size(), 1);

    auto asc_x_hat =
        hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, rets[0]));

    EXPECT_TRUE(xt::allclose(asc_x, asc_x_hat, 0.01, 0.001))
        << asc_x << std::endl
        << asc_x_hat << std::endl;

    // descending sort
    rets = Sort(
        &ctx, {x_v}, 0, false,
        [&](absl::Span<const spu::Value> inputs) {
          return hal::greater(&ctx, inputs[0], inputs[1]);
        },
        Visibility::VIS_SECRET);

    EXPECT_EQ(rets.size(), 1);

    auto des_x_hat =
        hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, rets[0]));

    EXPECT_TRUE(xt::allclose(des_x, des_x_hat, 0.01, 0.001))
        << des_x << std::endl
        << des_x_hat << std::endl;
  }
}

class SimpleSortTest
    : public ::testing::TestWithParam<std::tuple<
          size_t, FieldType, ProtocolKind, RuntimeConfig::SortMethod>> {};

TEST_P(SimpleSortTest, MultiOperands) {
  size_t npc = std::get<0>(GetParam());
  FieldType field = std::get<1>(GetParam());
  ProtocolKind prot = std::get<2>(GetParam());
  RuntimeConfig::SortMethod method = std::get<3>(GetParam());

  mpc::utils::simulate(
      npc, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        RuntimeConfig cfg;
        cfg.protocol = prot;
        cfg.field = field;
        cfg.enable_action_trace = false;
        cfg.sort_method = method;

        SPUContext ctx = test::makeSPUContext(cfg, lctx);

        xt::xarray<float> k1 = {7, 6, 5, 5, 4, 4, 4, 1, 3, 3};
        xt::xarray<float> k2 = {1, 2, 3, 6, 7, 6, 5, 2, 1, 2};

        xt::xarray<float> sorted_k1 = {1, 3, 3, 4, 4, 4, 5, 5, 6, 7};
        xt::xarray<float> sorted_k2 = {2, 1, 2, 5, 6, 7, 3, 6, 2, 1};

        Value k1_v = test::makeValue(&ctx, k1, VIS_SECRET);
        Value k2_v = test::makeValue(&ctx, k2, VIS_SECRET);

        std::vector<spu::Value> rets =
            SimpleSort(&ctx, {k1_v, k2_v}, 0, hal::SortDirection::Ascending, 2);

        EXPECT_EQ(rets.size(), 2);

        auto sorted_k1_hat =
            hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, rets[0]));
        auto sorted_k2_hat =
            hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, rets[1]));

        EXPECT_TRUE(xt::allclose(sorted_k1, sorted_k1_hat, 0.01, 0.001))
            << sorted_k1 << std::endl
            << sorted_k1_hat << std::endl;

        EXPECT_TRUE(xt::allclose(sorted_k2, sorted_k2_hat, 0.01, 0.001))
            << sorted_k2 << std::endl
            << sorted_k2_hat << std::endl;
      });
}

TEST_P(SimpleSortTest, SingleKeyWithPayload) {
  size_t npc = std::get<0>(GetParam());
  FieldType field = std::get<1>(GetParam());
  ProtocolKind prot = std::get<2>(GetParam());
  RuntimeConfig::SortMethod method = std::get<3>(GetParam());

  mpc::utils::simulate(
      npc, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        RuntimeConfig cfg;
        cfg.protocol = prot;
        cfg.field = field;
        cfg.enable_action_trace = false;
        cfg.sort_method = method;
        SPUContext ctx = test::makeSPUContext(cfg, lctx);

        xt::xarray<float> k1 = {7, 6, 5, 4, 1, 3, 2};
        xt::xarray<float> k2 = {1, 2, 3, 6, 7, 6, 5};

        xt::xarray<float> sorted_k1 = {1, 2, 3, 4, 5, 6, 7};
        xt::xarray<float> sorted_k2 = {7, 5, 6, 6, 3, 2, 1};

        Value k1_v = test::makeValue(&ctx, k1, VIS_SECRET);
        Value k2_v = test::makeValue(&ctx, k2, VIS_SECRET);

        std::vector<spu::Value> rets =
            SimpleSort(&ctx, {k1_v, k2_v}, 0, hal::SortDirection::Ascending, 1);

        EXPECT_EQ(rets.size(), 2);

        auto sorted_k1_hat =
            hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, rets[0]));
        auto sorted_k2_hat =
            hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, rets[1]));

        EXPECT_TRUE(xt::allclose(sorted_k1, sorted_k1_hat, 0.01, 0.001))
            << sorted_k1 << std::endl
            << sorted_k1_hat << std::endl;

        EXPECT_TRUE(xt::allclose(sorted_k2, sorted_k2_hat, 0.01, 0.001))
            << sorted_k2 << std::endl
            << sorted_k2_hat << std::endl;
      });
}

TEST_P(SimpleSortTest, PrivateKeyOnly) {
  size_t npc = std::get<0>(GetParam());
  FieldType field = std::get<1>(GetParam());
  ProtocolKind prot = std::get<2>(GetParam());
  RuntimeConfig::SortMethod method = std::get<3>(GetParam());

  mpc::utils::simulate(
      npc, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        RuntimeConfig cfg;
        cfg.protocol = prot;
        cfg.field = field;
        cfg.enable_action_trace = false;
        cfg.sort_method = method;
        SPUContext ctx = test::makeSPUContext(cfg, lctx);

        xt::xarray<float> k1 = {7, 6, 5, 4, 1, 3, 2};
        xt::xarray<float> k2 = {1, 2, 3, 6, 7, 6, 5};

        xt::xarray<float> sorted_k1 = {1, 2, 3, 4, 5, 6, 7};
        xt::xarray<float> sorted_k2 = {7, 5, 6, 6, 3, 2, 1};

        Value k1_v = test::makeValue(&ctx, k1, VIS_SECRET);
        Value k2_v = test::makeValue(&ctx, k2, VIS_SECRET);

        // make k1 private
        k1_v = RevealTo(&ctx, k1_v, 0);
        Value k2_val;

        for (Visibility vis_k2 : {VIS_PUBLIC, VIS_PRIVATE, VIS_SECRET}) {
          if (vis_k2 == VIS_PUBLIC) {
            k2_val = Reveal(&ctx, k2_v);
          } else if (vis_k2 == VIS_PRIVATE) {
            k2_val = RevealTo(&ctx, k2_v, 1);
          }

          std::vector<spu::Value> rets = SimpleSort(
              &ctx, {k1_v, k2_val}, 0, hal::SortDirection::Ascending, 1);

          EXPECT_EQ(rets.size(), 2);

          auto sorted_k1_hat =
              hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, rets[0]));
          auto sorted_k2_hat =
              hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, rets[1]));

          EXPECT_TRUE(xt::allclose(sorted_k1, sorted_k1_hat, 0.01, 0.001))
              << sorted_k1 << std::endl
              << sorted_k1_hat << std::endl;

          EXPECT_TRUE(xt::allclose(sorted_k2, sorted_k2_hat, 0.01, 0.001))
              << sorted_k2 << std::endl
              << sorted_k2_hat << std::endl;
        }
      });
}

TEST_P(SimpleSortTest, MixVisibilityKey) {
  size_t npc = std::get<0>(GetParam());
  FieldType field = std::get<1>(GetParam());
  ProtocolKind prot = std::get<2>(GetParam());
  RuntimeConfig::SortMethod method = std::get<3>(GetParam());

  mpc::utils::simulate(
      npc, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        RuntimeConfig cfg;
        cfg.protocol = prot;
        cfg.field = field;
        cfg.enable_action_trace = false;
        cfg.sort_method = method;
        SPUContext ctx = test::makeSPUContext(cfg, lctx);

        xt::xarray<float> k1 = {7, 3, 5, 4, 3, 3, 2};
        xt::xarray<float> k2 = {1, 2, 3, 6, 7, 6, 5};
        xt::xarray<float> k3 = {-1, 2, -3, 6, 7, 2, -3};
        xt::xarray<float> k4 = {-1, -2, 3, -6, -7, -6, 3};

        xt::xarray<float> sorted_k1 = {2, 3, 3, 3, 4, 5, 7};
        xt::xarray<float> sorted_k2 = {5, 2, 6, 7, 6, 3, 1};
        xt::xarray<float> sorted_k3 = {-3, 2, 2, 7, 6, -3, -1};
        xt::xarray<float> sorted_k4 = {3, -2, -6, -7, -6, 3, -1};

        Value k1_v = test::makeValue(&ctx, k1, VIS_SECRET);
        Value k2_v = test::makeValue(&ctx, k2, VIS_SECRET);
        Value k3_v = test::makeValue(&ctx, k3, VIS_SECRET);

        Value k4_v = test::makeValue(&ctx, k4, VIS_SECRET);

        k1_v = RevealTo(&ctx, k1_v, 1);
        k3_v = Reveal(&ctx, k3_v);

        Value k4_val;

        for (Visibility vis_k4 : {VIS_PUBLIC, VIS_PRIVATE, VIS_SECRET}) {
          if (vis_k4 == VIS_PUBLIC) {
            k4_val = Reveal(&ctx, k4_v);
          } else if (vis_k4 == VIS_PRIVATE) {
            k4_val = RevealTo(&ctx, k4_v, 1);
          }

          std::vector<spu::Value> rets =
              SimpleSort(&ctx, {k1_v, k2_v, k3_v, k4_val}, 0,
                         hal::SortDirection::Ascending, /*num_keys*/ 3);

          EXPECT_EQ(rets.size(), 4);

          auto sorted_k1_hat =
              hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, rets[0]));
          auto sorted_k2_hat =
              hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, rets[1]));
          auto sorted_k3_hat =
              hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, rets[2]));
          auto sorted_k4_hat =
              hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, rets[3]));

          EXPECT_TRUE(xt::allclose(sorted_k1, sorted_k1_hat, 0.01, 0.001))
              << sorted_k1 << std::endl
              << sorted_k1_hat << std::endl;

          EXPECT_TRUE(xt::allclose(sorted_k2, sorted_k2_hat, 0.01, 0.001))
              << sorted_k2 << std::endl
              << sorted_k2_hat << std::endl;

          EXPECT_TRUE(xt::allclose(sorted_k3, sorted_k3_hat, 0.01, 0.001))
              << sorted_k3 << std::endl
              << sorted_k3_hat << std::endl;

          EXPECT_TRUE(xt::allclose(sorted_k4, sorted_k4_hat, 0.01, 0.001))
              << sorted_k4 << std::endl
              << sorted_k4_hat << std::endl;
        }
      });
}

// Helper template to run unsigned type sort test
template <typename T>
void RunUnsignedSortTest(SPUContext *ctx) {
  xt::xarray<T> k1 = {7, 6, 5, 4, 1, 3, 2};
  xt::xarray<T> sorted_k1 = {1, 2, 3, 4, 5, 6, 7};
  xt::xarray<float> payload = {1, 2, 3, 6, 7, 6, 5};
  xt::xarray<float> sorted_payload = {7, 5, 6, 6, 3, 2, 1};

  Value k1_v = test::makeValue(ctx, k1, VIS_SECRET);
  Value payload_v = test::makeValue(ctx, payload, VIS_SECRET);

  std::vector<spu::Value> rets =
      SimpleSort(ctx, {k1_v, payload_v}, 0, hal::SortDirection::Ascending, 1);

  EXPECT_EQ(rets.size(), 2);

  auto sorted_k1_hat = hal::dump_public_as<T>(ctx, hal::reveal(ctx, rets[0]));
  auto sorted_payload_hat =
      hal::dump_public_as<float>(ctx, hal::reveal(ctx, rets[1]));

  EXPECT_TRUE(xt::allclose(sorted_k1, sorted_k1_hat, 0.01, 0.001))
      << "sort failed: " << sorted_k1 << std::endl
      << sorted_k1_hat << std::endl;

  EXPECT_TRUE(xt::allclose(sorted_payload, sorted_payload_hat, 0.01, 0.001))
      << "payload failed: " << sorted_payload << std::endl
      << sorted_payload_hat << std::endl;
}

TEST_P(SimpleSortTest, UnsignedTypeSort) {
  size_t npc = std::get<0>(GetParam());
  FieldType field = std::get<1>(GetParam());
  ProtocolKind prot = std::get<2>(GetParam());
  RuntimeConfig::SortMethod method = std::get<3>(GetParam());

  mpc::utils::simulate(npc,
                       [&](const std::shared_ptr<yacl::link::Context> &lctx) {
                         RuntimeConfig cfg;
                         cfg.protocol = prot;
                         cfg.field = field;
                         cfg.enable_action_trace = false;
                         cfg.sort_method = method;
                         SPUContext ctx = test::makeSPUContext(cfg, lctx);

                         RunUnsignedSortTest<uint8_t>(&ctx);
                         RunUnsignedSortTest<uint16_t>(&ctx);
                         RunUnsignedSortTest<uint32_t>(&ctx);
                         if (field >= FieldType::FM64) {
                           RunUnsignedSortTest<uint64_t>(&ctx);
                         }
                       });
}

// Helper template to test signed interpretation of unsigned-range values
// When values like 255 (for int8_t) are interpreted as signed, they become -1
// So sorting should treat them as negative numbers
template <typename SignedT, typename UnsignedT>
void RunSignedInterpretationSortTest(SPUContext *ctx) {
  // Use max value of unsigned type which becomes -1 when interpreted as signed
  constexpr UnsignedT max_val = std::numeric_limits<UnsignedT>::max();
  // Key: {0, max_val} where max_val is interpreted as -1 in signed
  // For ascending sort with signed interpretation: -1 < 0, so max_val comes
  // first
  xt::xarray<SignedT> k1 = {0, static_cast<SignedT>(max_val)};
  // Expected: max_val (-1) < 0, so sorted order is {max_val, 0}
  xt::xarray<SignedT> sorted_k1 = {static_cast<SignedT>(max_val), 0};
  xt::xarray<float> payload = {1.0, 2.0};
  xt::xarray<float> sorted_payload = {2.0, 1.0};

  Value k1_v = test::makeValue(ctx, k1, VIS_SECRET);
  Value payload_v = test::makeValue(ctx, payload, VIS_SECRET);

  std::vector<spu::Value> rets =
      SimpleSort(ctx, {k1_v, payload_v}, 0, hal::SortDirection::Ascending, 1);

  EXPECT_EQ(rets.size(), 2);

  auto sorted_k1_hat =
      hal::dump_public_as<SignedT>(ctx, hal::reveal(ctx, rets[0]));
  auto sorted_payload_hat =
      hal::dump_public_as<float>(ctx, hal::reveal(ctx, rets[1]));

  EXPECT_TRUE(xt::allclose(sorted_k1, sorted_k1_hat, 0.01, 0.001))
      << "sort failed: expected " << sorted_k1 << ", got " << sorted_k1_hat
      << std::endl;

  EXPECT_TRUE(xt::allclose(sorted_payload, sorted_payload_hat, 0.01, 0.001))
      << "payload failed: expected " << sorted_payload << ", got "
      << sorted_payload_hat << std::endl;
}

// IMPORTANT: the user should ensure that the data has the correct signed or
// unsigned type. Incorrect type interpretation will result in incorrect sort
// order (for example, treating signed values as unsigned may place negative
// numbers at the end instead of the beginning).
TEST_P(SimpleSortTest, SignedInterpretationSort) {
  size_t npc = std::get<0>(GetParam());
  FieldType field = std::get<1>(GetParam());
  ProtocolKind prot = std::get<2>(GetParam());
  RuntimeConfig::SortMethod method = std::get<3>(GetParam());

  mpc::utils::simulate(
      npc, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        RuntimeConfig cfg;
        cfg.protocol = prot;
        cfg.field = field;
        cfg.enable_action_trace = false;
        cfg.sort_method = method;
        SPUContext ctx = test::makeSPUContext(cfg, lctx);

        // Test: data is uint8_t range but treated as int8_t
        // 255 (uint8_t) -> -1 (int8_t), so -1 < 0
        RunSignedInterpretationSortTest<int8_t, uint8_t>(&ctx);

        // Test: data is uint16_t range but treated as int16_t
        // 65535 (uint16_t) -> -1 (int16_t), so -1 < 0
        RunSignedInterpretationSortTest<int16_t, uint16_t>(&ctx);

        // Test: data is uint32_t range but treated as int32_t
        // 4294967295 (uint32_t) -> -1 (int32_t), so -1 < 0
        RunSignedInterpretationSortTest<int32_t, uint32_t>(&ctx);

        if (field >= FieldType::FM64) {
          // Test: data is uint64_t range but treated as int64_t
          RunSignedInterpretationSortTest<int64_t, uint64_t>(&ctx);
        }
      });
}

TEST_P(SimpleSortTest, BoolKeyWithPayloads) {
  size_t npc = std::get<0>(GetParam());
  FieldType field = std::get<1>(GetParam());
  ProtocolKind prot = std::get<2>(GetParam());
  RuntimeConfig::SortMethod method = std::get<3>(GetParam());

  mpc::utils::simulate(
      npc, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        RuntimeConfig cfg;
        cfg.protocol = prot;
        cfg.field = field;
        cfg.enable_action_trace = false;
        cfg.sort_method = method;

        SPUContext ctx = test::makeSPUContext(cfg, lctx);

        // Bool key with two payloads
        xt::xarray<bool> k1 = {true, false, true, false, true};
        xt::xarray<float> p1 = {1.0, 2.0, 3.0, 4.0, 5.0};
        xt::xarray<int32_t> p2 = {10, 20, 30, 40, 50};

        // Expected sorted keys
        xt::xarray<bool> sorted_k1_desc = {true, true, true, false, false};
        xt::xarray<bool> sorted_k1_asc = {false, false, true, true, true};

        // Expected payloads (sorted within each group since sort is unstable)
        // Descending: true keys first {1,3,5}, then false keys {2,4}
        xt::xarray<float> sorted_p1_desc = {1.0, 3.0, 5.0, 2.0, 4.0};
        xt::xarray<int32_t> sorted_p2_desc = {10, 30, 50, 20, 40};
        // Ascending: false keys first {2,4}, then true keys {1,3,5}
        xt::xarray<float> sorted_p1_asc = {2.0, 4.0, 1.0, 3.0, 5.0};
        xt::xarray<int32_t> sorted_p2_asc = {20, 40, 10, 30, 50};

        Value k1_v = test::makeValue(&ctx, k1, VIS_SECRET);
        Value p1_v = test::makeValue(&ctx, p1, VIS_SECRET);
        Value p2_v = test::makeValue(&ctx, p2, VIS_SECRET);

        // Test descending sort (true before false)
        {
          std::vector<spu::Value> rets = SimpleSort(
              &ctx, {k1_v, p1_v, p2_v}, 0, hal::SortDirection::Descending, 1);

          EXPECT_EQ(rets.size(), 3);

          auto sorted_k1_hat =
              hal::dump_public_as<bool>(&ctx, hal::reveal(&ctx, rets[0]));
          auto sorted_p1_hat =
              hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, rets[1]));
          auto sorted_p2_hat =
              hal::dump_public_as<int32_t>(&ctx, hal::reveal(&ctx, rets[2]));

          // Check bool key is sorted correctly
          EXPECT_TRUE(xt::allclose(sorted_k1_desc, sorted_k1_hat, 0.01, 0.001))
              << "Bool descending sort failed: " << sorted_k1_desc << std::endl
              << sorted_k1_hat << std::endl;

          // Sort each part and compare (since sort is unstable within same key)
          auto p1_hat_sorted = xt::concatenate(
              xt::xtuple(xt::sort(xt::view(sorted_p1_hat, xt::range(0, 3))),
                         xt::sort(xt::view(sorted_p1_hat, xt::range(3, 5)))));
          auto p2_hat_sorted = xt::concatenate(
              xt::xtuple(xt::sort(xt::view(sorted_p2_hat, xt::range(0, 3))),
                         xt::sort(xt::view(sorted_p2_hat, xt::range(3, 5)))));

          EXPECT_TRUE(xt::allclose(sorted_p1_desc, p1_hat_sorted, 0.01, 0.001))
              << "Descending p1 failed: " << sorted_p1_desc << std::endl
              << p1_hat_sorted << std::endl;
          EXPECT_TRUE(xt::allclose(sorted_p2_desc, p2_hat_sorted, 0.01, 0.001))
              << "Descending p2 failed: " << sorted_p2_desc << std::endl
              << p2_hat_sorted << std::endl;
        }

        // Test ascending sort (false before true)
        {
          std::vector<spu::Value> rets = SimpleSort(
              &ctx, {k1_v, p1_v, p2_v}, 0, hal::SortDirection::Ascending, 1);

          EXPECT_EQ(rets.size(), 3);

          auto sorted_k1_hat =
              hal::dump_public_as<bool>(&ctx, hal::reveal(&ctx, rets[0]));
          auto sorted_p1_hat =
              hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, rets[1]));
          auto sorted_p2_hat =
              hal::dump_public_as<int32_t>(&ctx, hal::reveal(&ctx, rets[2]));

          // Check bool key is sorted correctly
          EXPECT_TRUE(xt::allclose(sorted_k1_asc, sorted_k1_hat, 0.01, 0.001))
              << "Bool ascending sort failed: " << sorted_k1_asc << std::endl
              << sorted_k1_hat << std::endl;

          // Sort each part and compare (since sort is unstable within same key)
          auto p1_hat_sorted = xt::concatenate(
              xt::xtuple(xt::sort(xt::view(sorted_p1_hat, xt::range(0, 2))),
                         xt::sort(xt::view(sorted_p1_hat, xt::range(2, 5)))));
          auto p2_hat_sorted = xt::concatenate(
              xt::xtuple(xt::sort(xt::view(sorted_p2_hat, xt::range(0, 2))),
                         xt::sort(xt::view(sorted_p2_hat, xt::range(2, 5)))));

          EXPECT_TRUE(xt::allclose(sorted_p1_asc, p1_hat_sorted, 0.01, 0.001))
              << "Ascending p1 failed: " << sorted_p1_asc << std::endl
              << p1_hat_sorted << std::endl;
          EXPECT_TRUE(xt::allclose(sorted_p2_asc, p2_hat_sorted, 0.01, 0.001))
              << "Ascending p2 failed: " << sorted_p2_asc << std::endl
              << p2_hat_sorted << std::endl;
        }
      });
}

// Test stable sort with Cheetah protocol
// Stable sort guarantees that elements with equal keys maintain their original
// relative order after sorting.
class CheetahStableSortTest
    : public ::testing::TestWithParam<std::tuple<FieldType>> {};

TEST_P(CheetahStableSortTest, StableSortPreservesOrder) {
  FieldType field = std::get<0>(GetParam());

  // Cheetah is a 2PC protocol
  mpc::utils::simulate(
      2, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        RuntimeConfig cfg;
        cfg.protocol = ProtocolKind::CHEETAH;
        cfg.field = field;
        cfg.enable_action_trace = false;
        SPUContext ctx = test::makeSPUContext(cfg, lctx);

        // Test data: keys have duplicates, payloads are unique identifiers
        // Key:     {3, 1, 2, 1, 3, 2}
        // Payload: {0, 1, 2, 3, 4, 5} (original indices)
        //
        // After stable ascending sort by key:
        // Key:     {1, 1, 2, 2, 3, 3}
        // Payload: {1, 3, 2, 5, 0, 4}
        //
        // For equal keys, the original relative order is preserved:
        // - Two 1s: indices 1 and 3, should remain in order (1, 3)
        // - Two 2s: indices 2 and 5, should remain in order (2, 5)
        // - Two 3s: indices 0 and 4, should remain in order (0, 4)
        xt::xarray<float> keys = {3, 1, 2, 1, 3, 2};
        xt::xarray<float> payloads = {0, 1, 2, 3, 4, 5};

        // Expected results for stable sort
        xt::xarray<float> stable_sorted_keys = {1, 1, 2, 2, 3, 3};
        xt::xarray<float> stable_sorted_payloads = {1, 3, 2, 5, 0, 4};

        Value keys_v = test::makeValue(&ctx, keys, VIS_SECRET);
        Value payloads_v = test::makeValue(&ctx, payloads, VIS_SECRET);

        // Test with is_stable = true
        std::vector<spu::Value> stable_rets =
            SimpleSort(&ctx, {keys_v, payloads_v}, 0,
                       hal::SortDirection::Ascending, 1, -1, true);

        EXPECT_EQ(stable_rets.size(), 2);

        auto sorted_keys_hat =
            hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, stable_rets[0]));
        auto sorted_payloads_hat =
            hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, stable_rets[1]));

        // Keys should be correctly sorted
        EXPECT_TRUE(
            xt::allclose(stable_sorted_keys, sorted_keys_hat, 0.01, 0.001))
            << "Keys mismatch: expected " << stable_sorted_keys << ", got "
            << sorted_keys_hat << std::endl;

        // Payloads should maintain stable order for equal keys
        EXPECT_TRUE(xt::allclose(stable_sorted_payloads, sorted_payloads_hat,
                                 0.01, 0.001))
            << "Stable sort failed: expected " << stable_sorted_payloads
            << ", got " << sorted_payloads_hat << std::endl;
      });
}

TEST_P(CheetahStableSortTest, StableSortDescending) {
  FieldType field = std::get<0>(GetParam());

  mpc::utils::simulate(2, [&](const std::shared_ptr<yacl::link::Context>
                                  &lctx) {
    RuntimeConfig cfg;
    cfg.protocol = ProtocolKind::CHEETAH;
    cfg.field = field;
    cfg.enable_action_trace = false;
    SPUContext ctx = test::makeSPUContext(cfg, lctx);

    // Test descending stable sort
    // Key:     {3, 1, 2, 1, 3, 2}
    // Payload: {0, 1, 2, 3, 4, 5}
    //
    // After stable descending sort:
    // Key:     {3, 3, 2, 2, 1, 1}
    // Payload: {0, 4, 2, 5, 1, 3}
    xt::xarray<float> keys = {3, 1, 2, 1, 3, 2};
    xt::xarray<float> payloads = {0, 1, 2, 3, 4, 5};

    xt::xarray<float> stable_sorted_keys = {3, 3, 2, 2, 1, 1};
    xt::xarray<float> stable_sorted_payloads = {0, 4, 2, 5, 1, 3};

    Value keys_v = test::makeValue(&ctx, keys, VIS_SECRET);
    Value payloads_v = test::makeValue(&ctx, payloads, VIS_SECRET);

    std::vector<spu::Value> stable_rets =
        SimpleSort(&ctx, {keys_v, payloads_v}, 0,
                   hal::SortDirection::Descending, 1, -1, true);

    EXPECT_EQ(stable_rets.size(), 2);

    auto sorted_keys_hat =
        hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, stable_rets[0]));
    auto sorted_payloads_hat =
        hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, stable_rets[1]));

    EXPECT_TRUE(xt::allclose(stable_sorted_keys, sorted_keys_hat, 0.01, 0.001))
        << "Keys mismatch: expected " << stable_sorted_keys << ", got "
        << sorted_keys_hat << std::endl;

    EXPECT_TRUE(
        xt::allclose(stable_sorted_payloads, sorted_payloads_hat, 0.01, 0.001))
        << "Stable descending sort failed: expected " << stable_sorted_payloads
        << ", got " << sorted_payloads_hat << std::endl;
  });
}

TEST_P(CheetahStableSortTest, UnstableSortMayNotPreserveOrder) {
  FieldType field = std::get<0>(GetParam());

  mpc::utils::simulate(
      2, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        RuntimeConfig cfg;
        cfg.protocol = ProtocolKind::CHEETAH;
        cfg.field = field;
        cfg.enable_action_trace = false;
        SPUContext ctx = test::makeSPUContext(cfg, lctx);

        xt::xarray<float> keys = {3, 1, 2, 1, 3, 2};
        xt::xarray<float> payloads = {0, 1, 2, 3, 4, 5};

        // Expected sorted keys (same for both stable and unstable)
        xt::xarray<float> sorted_keys = {1, 1, 2, 2, 3, 3};

        Value keys_v = test::makeValue(&ctx, keys, VIS_SECRET);
        Value payloads_v = test::makeValue(&ctx, payloads, VIS_SECRET);

        // Test with is_stable = false (default)
        std::vector<spu::Value> unstable_rets =
            SimpleSort(&ctx, {keys_v, payloads_v}, 0,
                       hal::SortDirection::Ascending, 1, -1, false);

        EXPECT_EQ(unstable_rets.size(), 2);

        auto sorted_keys_hat = hal::dump_public_as<float>(
            &ctx, hal::reveal(&ctx, unstable_rets[0]));
        auto sorted_payloads_hat = hal::dump_public_as<float>(
            &ctx, hal::reveal(&ctx, unstable_rets[1]));

        // Keys should still be correctly sorted
        EXPECT_TRUE(xt::allclose(sorted_keys, sorted_keys_hat, 0.01, 0.001))
            << "Keys mismatch: expected " << sorted_keys << ", got "
            << sorted_keys_hat << std::endl;

        // For unstable sort, we only verify that:
        // 1. Payloads are permuted consistently with keys
        // 2. Payloads for equal keys form the correct set
        // We check that payloads {1,3} appear for keys=1, {2,5} for keys=2,
        // {0,4} for keys=3
        auto p1 = xt::view(sorted_payloads_hat, xt::range(0, 2));
        auto p2 = xt::view(sorted_payloads_hat, xt::range(2, 4));
        auto p3 = xt::view(sorted_payloads_hat, xt::range(4, 6));

        auto p1_sorted = xt::sort(p1);
        auto p2_sorted = xt::sort(p2);
        auto p3_sorted = xt::sort(p3);

        xt::xarray<float> expected_p1 = {1, 3};
        xt::xarray<float> expected_p2 = {2, 5};
        xt::xarray<float> expected_p3 = {0, 4};

        EXPECT_TRUE(xt::allclose(expected_p1, p1_sorted, 0.01, 0.001))
            << "Payloads for key=1: expected {1,3}, got " << p1_sorted
            << std::endl;
        EXPECT_TRUE(xt::allclose(expected_p2, p2_sorted, 0.01, 0.001))
            << "Payloads for key=2: expected {2,5}, got " << p2_sorted
            << std::endl;
        EXPECT_TRUE(xt::allclose(expected_p3, p3_sorted, 0.01, 0.001))
            << "Payloads for key=3: expected {0,4}, got " << p3_sorted
            << std::endl;
      });
}

INSTANTIATE_TEST_SUITE_P(
    CheetahStableSortTestInstances, CheetahStableSortTest,
    // Note: Stable sort is only supported by radix sort method
    testing::Combine(testing::Values(FieldType::FM64, FieldType::FM128)),
    [](const testing::TestParamInfo<CheetahStableSortTest::ParamType> &p) {
      return fmt::format("{}", std::get<0>(p.param));
    });

INSTANTIATE_TEST_SUITE_P(
    SimpleSort2PCTestInstances, SimpleSortTest,
    testing::Combine(testing::Values(2),
                     testing::Values(FieldType::FM32, FieldType::FM64),
                     testing::Values(ProtocolKind::SEMI2K),
                     testing::Values(RuntimeConfig::SORT_DEFAULT,
                                     RuntimeConfig::SORT_RADIX,
                                     RuntimeConfig::SORT_QUICK,
                                     RuntimeConfig::SORT_NETWORK)),
    [](const testing::TestParamInfo<SimpleSortTest::ParamType> &p) {
      return fmt::format("{}x{}x{}x{}", std::get<0>(p.param),
                         std::get<1>(p.param), std::get<2>(p.param),
                         std::get<3>(p.param));
    });

INSTANTIATE_TEST_SUITE_P(
    SimpleSort3PCTestInstances, SimpleSortTest,
    testing::Combine(testing::Values(3),
                     testing::Values(FieldType::FM32, FieldType::FM64),
                     testing::Values(ProtocolKind::SEMI2K, ProtocolKind::ABY3),
                     testing::Values(RuntimeConfig::SORT_DEFAULT,
                                     RuntimeConfig::SORT_RADIX,
                                     RuntimeConfig::SORT_QUICK,
                                     RuntimeConfig::SORT_NETWORK)),
    [](const testing::TestParamInfo<SimpleSortTest::ParamType> &p) {
      return fmt::format("{}x{}x{}x{}", std::get<0>(p.param),
                         std::get<1>(p.param), std::get<2>(p.param),
                         std::get<3>(p.param));
    });

}  // namespace spu::kernel::hlo
