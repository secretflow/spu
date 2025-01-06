// Copyright 2024 Ant Group Co., Ltd.
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

#include "libspu/kernel/hlo/permute.h"

#include "gtest/gtest.h"
#include "xtensor/xio.hpp"

#include "libspu/core/encoding.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hlo/casting.h"
#include "libspu/kernel/hlo/const.h"
#include "libspu/kernel/test_util.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::kernel::hlo {

namespace {

using PermuteFunc = std::function<std::vector<Value>(
    SPUContext*, absl::Span<const spu::Value> inputs, const spu::Value& perm,
    int64_t perm_dim)>;

const FieldType kField = FM64;

enum class VisType {
  VisPriv0 = 0,  // private, own by party 0
  VisPriv1 = 1,  // private, own by party 1
  VisPub = 2,
  VisSec = 3,
};

const std::vector<VisType> kVisTypes = {VisType::VisPub, VisType::VisSec,
                                        VisType::VisPriv0, VisType::VisPriv1};

inline std::string get_vis_str(VisType type) {
  switch (type) {
    case VisType::VisPub:
      return "VisPub";
    case VisType::VisSec:
      return "VisSec";
    case VisType::VisPriv0:
      return "VisPriv0";
    case VisType::VisPriv1:
      return "VisPriv1";
    default:
      return "Unknown";
  }
}

bool checkCommFree(VisType x_vis, VisType perm_vis) {
  // Permutation is comm. free if:
  //  1. perm is Public
  //  2. perm is Private, x is Public or Private with same owner
  if (perm_vis == VisType::VisPub) {
    return true;
  } else if (perm_vis == VisType::VisPriv0 &&
             (x_vis == VisType::VisPriv0 || x_vis == VisType::VisPub)) {
    return true;
  } else if (perm_vis == VisType::VisPriv1 &&
             (x_vis == VisType::VisPriv1 || x_vis == VisType::VisPub)) {
    return true;
  }

  return false;
}

bool checkSpPass(VisType x_vis, VisType perm_vis) {
  // `inv_perm_av` will hit, when:
  //   1. perm is Private and x is Secret
  //   2. perm is Private and x is Private with different owner
  if (perm_vis == VisType::VisPriv0) {
    if (x_vis == VisType::VisSec || x_vis == VisType::VisPriv1) {
      return true;
    }
  } else if (perm_vis == VisType::VisPriv1) {
    if (x_vis == VisType::VisSec || x_vis == VisType::VisPriv0) {
      return true;
    }
  }

  return false;
}

Value makeTestValue(SPUContext* ctx, PtBufferView init, VisType vis) {
  DataType dtype = getEncodeType(init.pt_type);

  auto res = hal::constant(ctx, init, dtype, {});

  switch (vis) {
    case VisType::VisPub:
      return res;
    case VisType::VisSec: {
      return Seal(ctx, res);
    }
    case VisType::VisPriv0: {
      res = Seal(ctx, res);
      return RevealTo(ctx, res, 0);
    }
    case VisType::VisPriv1: {
      res = Seal(ctx, res);
      return RevealTo(ctx, res, 1);
    }
    default:
      SPU_THROW("Unknown vis type");
  }
}

template <typename T>
xt::xarray<T> evalSinglePermuteOp(SPUContext* ctx, VisType x_vis,
                                  VisType perm_vis, PtBufferView x,
                                  PtBufferView perm,
                                  const PermuteFunc& perm_func,
                                  int64_t perm_dim = 0) {
  auto x_v = makeTestValue(ctx, x, x_vis);
  auto perm_v = makeTestValue(ctx, perm, perm_vis);

  size_t send_round = ctx->lctx()->GetStats()->sent_actions;
  size_t recv_round = ctx->lctx()->GetStats()->recv_actions;
  auto perm_ret = perm_func(ctx, {x_v}, perm_v, perm_dim);
  send_round = ctx->lctx()->GetStats()->sent_actions - send_round;
  recv_round = ctx->lctx()->GetStats()->recv_actions - recv_round;

  // test whether hit the proper kernel.
  if (checkCommFree(x_vis, perm_vis)) {
    EXPECT_EQ(send_round, 0);
  }
  if (ctx->hasKernel("inv_perm_av") && checkSpPass(x_vis, perm_vis)) {
    auto n_repeat = x_v.shape().numel() / x_v.shape().dim(perm_dim);
    // For ss version, at least 3 rounds.
    EXPECT_LE(std::min(send_round, recv_round), 2 * n_repeat);
  }
  EXPECT_EQ(perm_ret.size(), 1);

  auto ret = perm_ret[0];
  if (!ret.isPublic()) {
    ret = Reveal(ctx, ret);
  }
  EXPECT_TRUE(ret.isPublic());

  return hal::dump_public_as<T>(ctx, ret);
}

template <typename T>
std::vector<xt::xarray<T>> evalMultiplePermuteOp(
    SPUContext* ctx, VisType x_vis, VisType perm_vis, PtBufferView x,
    PtBufferView perm, const PermuteFunc& perm_func, int64_t perm_dim = 0) {
  std::vector<Value> x_vec;
  x_vec.reserve(4);
  x_vec.push_back(makeTestValue(ctx, x, x_vis));
  x_vec.push_back(makeTestValue(ctx, x, x_vis));
  x_vec.push_back(makeTestValue(ctx, x, x_vis));
  x_vec.push_back(makeTestValue(ctx, x, x_vis));

  auto perm_v = makeTestValue(ctx, perm, perm_vis);

  auto perm_ret = perm_func(ctx, x_vec, perm_v, perm_dim);
  EXPECT_EQ(perm_ret.size(), 4);

  std::vector<xt::xarray<T>> ret_vec;
  for (auto ret : perm_ret) {
    if (!ret.isPublic()) {
      ret = Reveal(ctx, ret);
    }
    EXPECT_TRUE(ret.isPublic());
    ret_vec.push_back(hal::dump_public_as<T>(ctx, ret));
  }

  return ret_vec;
}

}  // namespace

class PermuteTest : public ::testing::TestWithParam<
                        std::tuple<VisType, VisType, ProtocolKind, size_t>> {};

INSTANTIATE_TEST_SUITE_P(
    GeneralPermute, PermuteTest,
    testing::Combine(testing::ValuesIn(kVisTypes),   // vis of x
                     testing::ValuesIn(kVisTypes),   // vis of perm
                     testing::Values(SEMI2K, ABY3),  // underlying protocol
                     testing::Values(2, 3)  // npc=2 is not valid in ABY3
                     ),
    [](const testing::TestParamInfo<PermuteTest::ParamType>& p) {
      return fmt::format("{}x{}x{}x{}", get_vis_str(std::get<0>(p.param)),
                         get_vis_str(std::get<1>(p.param)),
                         std::get<2>(p.param), std::get<3>(p.param));
    });

TEST_P(PermuteTest, SinglePermuteWork) {
  const VisType x_vis = std::get<0>(GetParam());
  const VisType perm_vis = std::get<1>(GetParam());
  const ProtocolKind protocol = std::get<2>(GetParam());
  const size_t npc = std::get<3>(GetParam());

  if (protocol == ABY3 && npc == 2) {
    return;
  }

  xt::xarray<int64_t> x = {10, 0, 2, 3, 9, 1, 5, 6};
  xt::xarray<int64_t> perm = {2, 7, 1, 6, 0, 4, 3, 5};

  xt::xarray<int64_t> expected_inv_perm = {9, 2, 10, 5, 1, 6, 3, 0};
  xt::xarray<int64_t> expected_perm = {2, 6, 0, 5, 10, 9, 3, 1};

  mpc::utils::simulate(
      npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
        SPUContext sctx = test::makeSPUContext(protocol, kField, lctx);

        // test of inv_permute
        auto inv_perm_ret = evalSinglePermuteOp<int64_t>(&sctx, x_vis, perm_vis,
                                                         x, perm, InvPermute);
        EXPECT_TRUE(xt::allclose(expected_inv_perm, inv_perm_ret, 0.001, 0.001))
            << expected_inv_perm << std::endl
            << inv_perm_ret << std::endl;

        // test of permute
        auto perm_ret = evalSinglePermuteOp<int64_t>(&sctx, x_vis, perm_vis, x,
                                                     perm, Permute);
        EXPECT_TRUE(xt::allclose(expected_perm, perm_ret, 0.001, 0.001))
            << expected_perm << std::endl
            << perm_ret << std::endl;
      });
}

TEST_P(PermuteTest, PermDimWork) {
  const VisType x_vis = std::get<0>(GetParam());
  const VisType perm_vis = std::get<1>(GetParam());
  const ProtocolKind protocol = std::get<2>(GetParam());
  const size_t npc = std::get<3>(GetParam());

  if (protocol == ABY3 && npc == 2) {
    return;
  }

  xt::xarray<int64_t> x = {{10, 0, 2, 3, 9, 1, 5, 6},
                           {-10, 0, -2, -3, -9, -1, -5, -6}};
  xt::xarray<int64_t> perm = {2, 7, 1, 6, 0, 4, 3, 5};

  xt::xarray<int64_t> expected_inv_perm = {{9, 2, 10, 5, 1, 6, 3, 0},
                                           {-9, -2, -10, -5, -1, -6, -3, -0}};
  xt::xarray<int64_t> expected_perm = {{2, 6, 0, 5, 10, 9, 3, 1},
                                       {-2, -6, -0, -5, -10, -9, -3, -1}};

  mpc::utils::simulate(
      npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
        SPUContext sctx = test::makeSPUContext(protocol, kField, lctx);

        // test of inv_permute
        auto inv_perm_ret = evalSinglePermuteOp<int64_t>(
            &sctx, x_vis, perm_vis, x, perm, InvPermute, /*perm_dim*/ 1);
        EXPECT_TRUE(xt::allclose(expected_inv_perm, inv_perm_ret, 0.001, 0.001))
            << expected_inv_perm << std::endl
            << inv_perm_ret << std::endl;

        // test of permute
        auto perm_ret = evalSinglePermuteOp<int64_t>(
            &sctx, x_vis, perm_vis, x, perm, Permute, /*perm_dim*/ 1);
        EXPECT_TRUE(xt::allclose(expected_perm, perm_ret, 0.001, 0.001))
            << expected_perm << std::endl
            << perm_ret << std::endl;
      });
}

TEST_P(PermuteTest, MultiplePermuteWork) {
  const VisType x_vis = std::get<0>(GetParam());
  const VisType perm_vis = std::get<1>(GetParam());
  const ProtocolKind protocol = std::get<2>(GetParam());
  const size_t npc = std::get<3>(GetParam());

  if (protocol == ABY3 && npc == 2) {
    return;
  }

  xt::xarray<int64_t> x = {10, 0, 2, 3, 9, 1, 5, 6};
  xt::xarray<int64_t> perm = {2, 7, 1, 6, 0, 4, 3, 5};

  xt::xarray<int64_t> expected_inv_perm = {9, 2, 10, 5, 1, 6, 3, 0};
  xt::xarray<int64_t> expected_perm = {2, 6, 0, 5, 10, 9, 3, 1};

  mpc::utils::simulate(
      npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
        SPUContext sctx = test::makeSPUContext(protocol, kField, lctx);

        // test of inv_permute
        auto inv_perm_ret_vec = evalMultiplePermuteOp<int64_t>(
            &sctx, x_vis, perm_vis, x, perm, InvPermute);
        for (const auto& inv_perm_ret : inv_perm_ret_vec) {
          EXPECT_TRUE(
              xt::allclose(expected_inv_perm, inv_perm_ret, 0.001, 0.001))
              << expected_inv_perm << std::endl
              << inv_perm_ret << std::endl;
        }

        // test of permute
        auto perm_ret_vec = evalMultiplePermuteOp<int64_t>(
            &sctx, x_vis, perm_vis, x, perm, Permute);
        for (const auto& perm_ret : perm_ret_vec) {
          EXPECT_TRUE(xt::allclose(expected_perm, perm_ret, 0.001, 0.001))
              << expected_perm << std::endl
              << perm_ret << std::endl;
        }
      });
}

class PermuteEmptyTest : public ::testing::TestWithParam<ProtocolKind> {};

INSTANTIATE_TEST_SUITE_P(
    PermuteEmpty, PermuteEmptyTest,
    testing::Values(ProtocolKind::SEMI2K, ProtocolKind::ABY3),
    [](const testing::TestParamInfo<PermuteEmptyTest::ParamType>& p) {
      return fmt::format("{}", p.param);
    });

TEST_P(PermuteEmptyTest, Empty) {
  ProtocolKind prot = GetParam();

  mpc::utils::simulate(
      3, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
        SPUContext sctx = test::makeSPUContext(prot, kField, lctx);

        auto empty_x =
            Seal(&sctx, Constant(&sctx, static_cast<int64_t>(1), {0}));
        auto empty_perm =
            Seal(&sctx, Constant(&sctx, static_cast<int64_t>(0), {0}));

        auto empty_inv_perm_x = InvPermute(&sctx, {empty_x}, empty_perm, 0);
        EXPECT_EQ(empty_inv_perm_x.size(), 1);
        EXPECT_EQ(empty_inv_perm_x[0].numel(), 0);
        EXPECT_EQ(empty_inv_perm_x[0].shape().size(), 1);
        EXPECT_EQ(empty_inv_perm_x[0].shape()[0], 0);

        auto empty_perm_x = Permute(&sctx, {empty_x}, empty_perm, 0);
        EXPECT_EQ(empty_perm_x.size(), 1);
        EXPECT_EQ(empty_perm_x[0].numel(), 0);
        EXPECT_EQ(empty_perm_x[0].shape().size(), 1);
        EXPECT_EQ(empty_perm_x[0].shape()[0], 0);
      });
}

}  // namespace spu::kernel::hlo
