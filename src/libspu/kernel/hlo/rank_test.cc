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
#include "libspu/kernel/hlo/rank.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xtensor/xsort.hpp"

#include "libspu/core/context.h"
#include "libspu/core/value.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/shape_ops.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/test_util.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::kernel::hlo {
class TopkTest : public ::testing::TestWithParam<
                     std::tuple<size_t, FieldType, ProtocolKind>> {};

TEST_P(TopkTest, FlattenArrayTest) {
  size_t npc = std::get<0>(GetParam());
  FieldType field = std::get<1>(GetParam());
  ProtocolKind prot = std::get<2>(GetParam());

  mpc::utils::simulate(
      npc, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        SPUContext sctx = test::makeSPUContext(prot, field, lctx);
        xt::xarray<double> a = {3.2, 0.2, 3.1, 0.2, -0.2, 0, 1};
        int64_t k = 2;

        // flatten 1-d array test
        {
          auto inp = test::makeValue(&sctx, a, VIS_SECRET);
          auto out = TopK(&sctx, inp, k);

          auto val = out[0];
          auto ind = out[1];

          auto val_pub =
              hal::dump_public_as<float>(&sctx, hal::reveal(&sctx, val));
          auto ind_pub =
              hal::dump_public_as<int64_t>(&sctx, hal::reveal(&sctx, ind));

          xt::xarray<float> gt_val = {3.2F, 3.1F};
          xt::xarray<int64_t> gt_ind = {0, 2};

          ASSERT_THAT(val_pub.shape(), testing::ElementsAre(k));
          ASSERT_THAT(ind_pub.shape(), testing::ElementsAre(k));

          EXPECT_TRUE(
              xt::allclose(xt::sort(val_pub), xt::sort(gt_val), 0.001, 0.001));
          EXPECT_TRUE(xt::all(xt::equal(xt::sort(ind_pub), xt::sort(gt_ind))));
        }

        // public test
        {
          auto inp = test::makeValue(&sctx, a, VIS_PUBLIC);
          auto out = TopK(&sctx, inp, k);

          auto val = out[0];
          auto ind = out[1];

          auto val_pub = hal::dump_public_as<float>(&sctx, val);
          auto ind_pub = hal::dump_public_as<int64_t>(&sctx, ind);

          xt::xarray<float> gt_val = {3.2F, 3.1F};
          xt::xarray<int64_t> gt_ind = {0, 2};

          ASSERT_THAT(val_pub.shape(), testing::ElementsAre(k));
          ASSERT_THAT(ind_pub.shape(), testing::ElementsAre(k));

          EXPECT_TRUE(
              xt::allclose(xt::sort(val_pub), xt::sort(gt_val), 0.001, 0.001));
          EXPECT_TRUE(xt::all(xt::equal(xt::sort(ind_pub), xt::sort(gt_ind))));
        }
      });
}

TEST_P(TopkTest, MultiDimArrayTest) {
  size_t npc = std::get<0>(GetParam());
  FieldType field = std::get<1>(GetParam());
  ProtocolKind prot = std::get<2>(GetParam());

  mpc::utils::simulate(
      npc, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        SPUContext sctx = test::makeSPUContext(prot, field, lctx);

        // 2-d array
        xt::xarray<double> a = {{3.2, 0.2, 3.1, 0.2, -0.2},
                                {0.2, -0.2, 0, 1, 0.12}};
        int64_t k = 2;
        {
          auto inp = test::makeValue(&sctx, a, VIS_SECRET);
          auto out = TopK(&sctx, inp, k);

          auto val = out[0];
          auto ind = out[1];

          auto val_pub =
              hal::dump_public_as<float>(&sctx, hal::reveal(&sctx, val));
          auto ind_pub =
              hal::dump_public_as<int64_t>(&sctx, hal::reveal(&sctx, ind));

          xt::xarray<float> gt_val = {{3.2F, 3.1F}, {1.0F, 0.2F}};
          xt::xarray<int64_t> gt_ind = {{0, 2}, {3, 0}};

          ASSERT_THAT(val_pub.shape(),
                      testing::ElementsAre(inp.shape().front(), k));
          ASSERT_THAT(ind_pub.shape(),
                      testing::ElementsAre(inp.shape().front(), k));

          EXPECT_TRUE(
              xt::allclose(xt::sort(val_pub), xt::sort(gt_val), 0.001, 0.001));
          EXPECT_TRUE(xt::all(xt::equal(xt::sort(ind_pub), xt::sort(gt_ind))));
        }
      });
}

TEST_P(TopkTest, MultiKTest) {
  size_t npc = std::get<0>(GetParam());
  FieldType field = std::get<1>(GetParam());
  ProtocolKind prot = std::get<2>(GetParam());

  mpc::utils::simulate(
      npc, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        SPUContext sctx = test::makeSPUContext(prot, field, lctx);
        xt::xarray<double> a = {3.2, 0.2, 3.1, 0.2, -0.2, 0, 1};
        int64_t k1 = 2;
        int64_t k2 = 5;

        // flatten 1-d array test
        {
          auto inp = test::makeValue(&sctx, a, VIS_SECRET);
          auto out = TopK(&sctx, inp, k1, k2);

          auto val = out[0];
          auto ind = out[1];

          auto val_pub =
              hal::dump_public_as<float>(&sctx, hal::reveal(&sctx, val));
          auto ind_pub =
              hal::dump_public_as<int64_t>(&sctx, hal::reveal(&sctx, ind));

          xt::xarray<float> gt_val = {3.2F, 3.1F, 1.0F, 0.2F, 0.2F};
          xt::xarray<int64_t> gt_ind = {0, 2, 6, 1, 3};

          ASSERT_THAT(val_pub.shape(), testing::ElementsAre(k2));
          ASSERT_THAT(ind_pub.shape(), testing::ElementsAre(k2));

          EXPECT_TRUE(xt::allclose(val_pub[k1 - 1], 3.1F, 0.001, 0.001));
          EXPECT_TRUE(xt::allclose(val_pub[k2 - 1], 0.2F, 0.001, 0.001));

          EXPECT_TRUE(
              xt::allclose(xt::sort(val_pub), xt::sort(gt_val), 0.001, 0.001));
          EXPECT_TRUE(xt::all(xt::equal(xt::sort(ind_pub), xt::sort(gt_ind))));
        }

        // public test
        {
          auto inp = test::makeValue(&sctx, a, VIS_PUBLIC);
          auto out = TopK(&sctx, inp, k1, k2);

          auto val = out[0];
          auto ind = out[1];

          auto val_pub = hal::dump_public_as<float>(&sctx, val);
          auto ind_pub = hal::dump_public_as<int64_t>(&sctx, ind);

          xt::xarray<float> gt_val = {3.2F, 3.1F, 1.0F, 0.2F, 0.2F};
          xt::xarray<int64_t> gt_ind = {0, 2, 6, 1, 3};

          ASSERT_THAT(val_pub.shape(), testing::ElementsAre(k2));
          ASSERT_THAT(ind_pub.shape(), testing::ElementsAre(k2));

          EXPECT_TRUE(xt::allclose(val_pub[k1 - 1], 3.1F, 0.001, 0.001));
          EXPECT_TRUE(xt::allclose(val_pub[k2 - 1], 0.2F, 0.001, 0.001));

          EXPECT_TRUE(
              xt::allclose(xt::sort(val_pub), xt::sort(gt_val), 0.001, 0.001));
          EXPECT_TRUE(xt::all(xt::equal(xt::sort(ind_pub), xt::sort(gt_ind))));
        }
      });
}

TEST_P(TopkTest, SmallestTest) {
  size_t npc = std::get<0>(GetParam());
  FieldType field = std::get<1>(GetParam());
  ProtocolKind prot = std::get<2>(GetParam());

  mpc::utils::simulate(
      npc, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        SPUContext sctx = test::makeSPUContext(prot, field, lctx);
        xt::xarray<double> a = {3.2, 0.2, 3.1, 0.2, -0.2, 0, 1};
        int64_t k1 = 2;
        int64_t k2 = 5;

        // flatten 1-d array test
        {
          auto inp = test::makeValue(&sctx, a, VIS_SECRET);
          auto out = TopK(&sctx, inp, k1, k2, false);

          auto val = out[0];
          auto ind = out[1];

          auto val_pub =
              hal::dump_public_as<float>(&sctx, hal::reveal(&sctx, val));
          auto ind_pub =
              hal::dump_public_as<int64_t>(&sctx, hal::reveal(&sctx, ind));

          xt::xarray<float> gt_val = {-0.2F, 0.F, 0.2F, 0.2F, 1.F};
          xt::xarray<int64_t> gt_ind = {4, 5, 1, 3, 6};

          ASSERT_THAT(val_pub.shape(), testing::ElementsAre(k2));
          ASSERT_THAT(ind_pub.shape(), testing::ElementsAre(k2));

          EXPECT_TRUE(xt::allclose(val_pub[k1 - 1], 0.0F, 0.001, 0.001));
          EXPECT_TRUE(xt::allclose(val_pub[k2 - 1], 1.0F, 0.001, 0.001));

          EXPECT_TRUE(
              xt::allclose(xt::sort(val_pub), xt::sort(gt_val), 0.001, 0.001));
          EXPECT_TRUE(xt::all(xt::equal(xt::sort(ind_pub), xt::sort(gt_ind))));
        }

        // public test
        {
          auto inp = test::makeValue(&sctx, a, VIS_PUBLIC);
          auto out = TopK(&sctx, inp, k1, k2, false);

          auto val = out[0];
          auto ind = out[1];

          auto val_pub = hal::dump_public_as<float>(&sctx, val);
          auto ind_pub = hal::dump_public_as<int64_t>(&sctx, ind);

          xt::xarray<float> gt_val = {-0.2F, 0.F, 0.2F, 0.2F, 1.F};
          xt::xarray<int64_t> gt_ind = {4, 5, 1, 3, 6};

          ASSERT_THAT(val_pub.shape(), testing::ElementsAre(k2));
          ASSERT_THAT(ind_pub.shape(), testing::ElementsAre(k2));

          EXPECT_TRUE(xt::allclose(val_pub[k1 - 1], 0.0F, 0.001, 0.001));
          EXPECT_TRUE(xt::allclose(val_pub[k2 - 1], 1.0F, 0.001, 0.001));

          EXPECT_TRUE(
              xt::allclose(xt::sort(val_pub), xt::sort(gt_val), 0.001, 0.001));
          EXPECT_TRUE(xt::all(xt::equal(xt::sort(ind_pub), xt::sort(gt_ind))));
        }
      });
}

TEST_P(TopkTest, PrivateTest) {
  size_t npc = std::get<0>(GetParam());
  FieldType field = std::get<1>(GetParam());
  ProtocolKind prot = std::get<2>(GetParam());

  mpc::utils::simulate(
      npc, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        SPUContext sctx = test::makeSPUContext(prot, field, lctx);
        xt::xarray<double> a = {3.2, 0.2, 3.1};
        xt::xarray<double> b = {0.2, -0.2, 0, 1};
        int64_t k = 2;

        // flatten 1-d array test
        {
          // auto inp = test::makeValue(&sctx, a, VIS_SECRET);
          auto a_inp = hal::constant(&sctx, a, DT_F64);
          auto b_inp = hal::constant(&sctx, b, DT_F64);

          a_inp = hal::_p2v(&sctx, a_inp, 0).setDtype(a_inp.dtype());
          b_inp = hal::_p2v(&sctx, b_inp, 1).setDtype(b_inp.dtype());

          auto inp = hal::concatenate(&sctx, {a_inp, b_inp}, 0);

          auto out = TopK(&sctx, inp, k);

          auto val = out[0];
          auto ind = out[1];

          auto val_pub =
              hal::dump_public_as<float>(&sctx, hal::reveal(&sctx, val));
          auto ind_pub =
              hal::dump_public_as<int64_t>(&sctx, hal::reveal(&sctx, ind));

          xt::xarray<float> gt_val = {3.2F, 3.1F};
          xt::xarray<int64_t> gt_ind = {0, 2};

          ASSERT_THAT(val_pub.shape(), testing::ElementsAre(k));
          ASSERT_THAT(ind_pub.shape(), testing::ElementsAre(k));

          EXPECT_TRUE(
              xt::allclose(xt::sort(val_pub), xt::sort(gt_val), 0.001, 0.001));
          EXPECT_TRUE(xt::all(xt::equal(xt::sort(ind_pub), xt::sort(gt_ind))));
        }
      });
}

TEST_P(TopkTest, ValueOnlyTest) {
  size_t npc = std::get<0>(GetParam());
  FieldType field = std::get<1>(GetParam());
  ProtocolKind prot = std::get<2>(GetParam());

  mpc::utils::simulate(
      npc, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        SPUContext sctx = test::makeSPUContext(prot, field, lctx);
        xt::xarray<double> a = {3.2, 0.2, 3.1, 0.2, -0.2, 0, 1};
        int64_t k1 = 2;
        int64_t k2 = 5;

        // flatten 1-d array test
        {
          auto inp = test::makeValue(&sctx, a, VIS_SECRET);
          auto out = TopK(&sctx, inp, k1, k2, true, true);

          EXPECT_EQ(out.size(), 1);

          auto val = out[0];
          auto val_pub =
              hal::dump_public_as<float>(&sctx, hal::reveal(&sctx, val));
          ASSERT_THAT(val_pub.shape(), testing::ElementsAre(k2));
          EXPECT_TRUE(xt::allclose(val_pub[k1 - 1], 3.1F, 0.001, 0.001));
          EXPECT_TRUE(xt::allclose(val_pub[k2 - 1], 0.2F, 0.001, 0.001));

          xt::xarray<float> gt_val = {3.2F, 3.1F, 1.0F, 0.2F, 0.2F};
          EXPECT_TRUE(
              xt::allclose(xt::sort(val_pub), xt::sort(gt_val), 0.001, 0.001));
        }
        // 1-d public test
        {
          auto inp = test::makeValue(&sctx, a, VIS_PUBLIC);
          auto out = TopK(&sctx, inp, k1, k2, true, true);
          EXPECT_EQ(out.size(), 1);

          auto val = out[0];
          auto val_pub = hal::dump_public_as<float>(&sctx, val);
          ASSERT_THAT(val_pub.shape(), testing::ElementsAre(k2));
          EXPECT_TRUE(xt::allclose(val_pub[k1 - 1], 3.1F, 0.001, 0.001));
          EXPECT_TRUE(xt::allclose(val_pub[k2 - 1], 0.2F, 0.001, 0.001));

          xt::xarray<float> gt_val = {3.2F, 3.1F, 1.0F, 0.2F, 0.2F};
          EXPECT_TRUE(
              xt::allclose(xt::sort(val_pub), xt::sort(gt_val), 0.001, 0.001));
        }
        // multi-dim test
        xt::xarray<double> b = {{3.2, 0.2, 3.1, 0.2, -0.2},
                                {0.2, -0.2, 0, 1, 0.12}};
        int64_t k = 2;
        {
          auto inp = test::makeValue(&sctx, b, VIS_SECRET);
          auto out = TopK(&sctx, inp, k, k, true, true);
          EXPECT_EQ(out.size(), 1);

          auto val = out[0];
          auto val_pub =
              hal::dump_public_as<float>(&sctx, hal::reveal(&sctx, val));
          xt::xarray<float> gt_val = {{3.2F, 3.1F}, {1.0F, 0.2F}};

          ASSERT_THAT(val_pub.shape(),
                      testing::ElementsAre(inp.shape().front(), k));
          EXPECT_TRUE(
              xt::allclose(xt::sort(val_pub), xt::sort(gt_val), 0.001, 0.001));
        }
      });
}

INSTANTIATE_TEST_SUITE_P(
    Topk2PCTestInstances, TopkTest,
    testing::Combine(testing::Values(2),
                     testing::Values(FieldType::FM64, FieldType::FM128),
                     testing::Values(ProtocolKind::SEMI2K,
                                     ProtocolKind::CHEETAH)),
    [](const testing::TestParamInfo<TopkTest::ParamType> &p) {
      return fmt::format("{}x{}x{}", std::get<0>(p.param), std::get<1>(p.param),
                         std::get<2>(p.param));
    });

INSTANTIATE_TEST_SUITE_P(
    Topk3PCTestInstances, TopkTest,
    testing::Combine(testing::Values(3),
                     testing::Values(FieldType::FM64, FieldType::FM128),
                     testing::Values(ProtocolKind::SEMI2K, ProtocolKind::ABY3)),
    [](const testing::TestParamInfo<TopkTest::ParamType> &p) {
      return fmt::format("{}x{}x{}", std::get<0>(p.param), std::get<1>(p.param),
                         std::get<2>(p.param));
    });

}  // namespace spu::kernel::hlo