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

#include "libspu/kernel/hal/join.h"

#include <memory>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"
#include "xtensor/xio.hpp"
#include "xtensor/xsort.hpp"

#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/hlo/casting.h"
#include "libspu/kernel/hlo/const.h"
#include "libspu/kernel/hlo/geometrical.h"
#include "libspu/kernel/test_util.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::kernel::hal {

namespace {
SPUContext makeSPUContextWithProfile(
    ProtocolKind prot_kind, FieldType field,
    const std::shared_ptr<yacl::link::Context>& lctx) {
  RuntimeConfig cfg;
  cfg.protocol = prot_kind;
  cfg.field = field;
  cfg.enable_action_trace = false;

  if (lctx->Rank() == 0) {
    cfg.enable_hal_profile = true;
    cfg.enable_pphlo_profile = true;
  }
  return test::makeSPUContext(cfg, lctx);
}
}  // namespace

class JoinTest
    : public ::testing::TestWithParam<std::tuple<FieldType, ProtocolKind>> {};

INSTANTIATE_TEST_SUITE_P(
    JoinTestInstances, JoinTest,
    testing::Combine(testing::Values(FieldType::FM64, FieldType::FM128),
                     testing::Values(ProtocolKind::SEMI2K)),
    [](const testing::TestParamInfo<JoinTest::ParamType>& p) {
      return fmt::format("{}x{}", std::get<0>(p.param), std::get<1>(p.param));
    });

TEST_P(JoinTest, Work) {
  FieldType field = std::get<0>(GetParam());
  ProtocolKind prot = std::get<1>(GetParam());
  size_t num_join_keys = 1;
  const size_t num_hash = 3;
  const double scale_factor = 1.5;

  const Shape shape_1 = {2, 8};
  const Shape shape_2 = {2, 7};

  xt::xarray<uint64_t> data_1 = {{1, 4, 8, 5, 2, 6, 7, 0},
                                 {11, 44, 88, 55, 22, 66, 77, 00}};
  xt::xarray<uint64_t> data_2 = {{3, 5, 7, 9, 1, 4, 0},
                                 {333, 555, 777, 999, 111, 444, 000}};
  xt::xarray<uint64_t> data_out_expected = {
      {1, 4, 5, 7, 0}, {11, 44, 55, 77, 0}, {111, 444, 555, 777, 0}};

  mpc::utils::simulate(
      2, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
        SPUContext sctx = makeSPUContextWithProfile(prot, field, lctx);

        std::vector<Value> table1_columns;
        for (int64_t i = 0; i < shape_1[0]; ++i) {
          xt::xarray<uint64_t> col_data = xt::row(data_1, i);
          Value col = test::makeValue(&sctx, col_data, VIS_SECRET);
          table1_columns.push_back(col);
        }

        std::vector<Value> table2_columns;
        for (int64_t i = 0; i < shape_2[0]; ++i) {
          xt::xarray<uint64_t> col_data = xt::row(data_2, i);
          Value col = test::makeValue(&sctx, col_data, VIS_SECRET);
          table2_columns.push_back(col);
        }

        setupTrace(&sctx, sctx.config());

        auto ret = join_uu(&sctx, table1_columns, table2_columns, num_join_keys,
                           num_hash, scale_factor);

        test::printProfileData(&sctx);

        EXPECT_EQ(ret.size(), 1 + shape_1[0] + shape_2[0] - num_join_keys);

        auto valid_flag =
            hal::dump_public_as<uint64_t>(&sctx, hal::reveal(&sctx, ret[0]));
        xt::xarray<uint64_t> valid_ret;
        for (size_t i = 1; i < ret.size(); ++i) {
          auto ret_hat =
              hal::dump_public_as<uint64_t>(&sctx, hal::reveal(&sctx, ret[i]));

          valid_ret = xt::filter(ret_hat, xt::equal(valid_flag, 1));
          EXPECT_EQ(valid_ret, xt::row(data_out_expected, i - 1))
              << "Mismatch in output column " << i - 1 << std::endl;
        }
      });
}

class MultiKeyJoinTest
    : public ::testing::TestWithParam<
          std::tuple<FieldType, ProtocolKind, size_t, size_t>> {};

INSTANTIATE_TEST_SUITE_P(
    MultiKeyJoinTestInstances, MultiKeyJoinTest,
    testing::Combine(testing::Values(FieldType::FM64),
                     testing::Values(ProtocolKind::SEMI2K),
                     testing::Values(2, 3), testing::Values(2, 3)),
    [](const testing::TestParamInfo<MultiKeyJoinTest::ParamType>& p) {
      return fmt::format("{}x{}x{}x{}", std::get<0>(p.param),
                         std::get<1>(p.param), std::get<2>(p.param),
                         std::get<3>(p.param));
    });

TEST_P(MultiKeyJoinTest, Work) {
  FieldType field = std::get<0>(GetParam());
  ProtocolKind prot = std::get<1>(GetParam());
  size_t num_join_keys = std::get<2>(GetParam());
  size_t num_hash = std::get<3>(GetParam());
  double scale_factor = 1.5;

  const Shape shape_1 = {4, 9};
  const Shape shape_2 = {5, 12};

  xt::xarray<uint64_t> data_1 = {{1, 4, 8, 5, 2, 6, 7, 0, 10},
                                 {11, 14, 18, 15, 12, 16, 17, 10, 110},
                                 {21, 24, 28, 25, 22, 26, 27, 20, 210},
                                 {31, 34, 38, 35, 32, 36, 37, 30, 310}};
  xt::xarray<uint64_t> data_2 = {
      {3, 5, 7, 9, 1, 4, 0, 11, 10, 6, 13, 14},
      {13, 15, 17, 19, 11, 14, 10, 111, 110, 16, 113, 114},
      {23, 25, 27, 29, 21, 24, 20, 211, 210, 26, 213, 214},
      {33, 35, 37, 39, 31, 34, 30, 311, 310, 36, 313, 314},
      {43, 45, 47, 49, 41, 44, 40, 411, 410, 46, 413, 414}};

  xt::xarray<uint64_t> data_out_expected_for_2_keys = {
      {5, 7, 1, 4, 0, 10, 6},        {15, 17, 11, 14, 10, 110, 16},
      {25, 27, 21, 24, 20, 210, 26}, {35, 37, 31, 34, 30, 310, 36},
      {45, 47, 41, 44, 40, 410, 46}, {25, 27, 21, 24, 20, 210, 26},
      {35, 37, 31, 34, 30, 310, 36}};

  xt::xarray<uint64_t> data_out_expected_for_3_keys = {
      {5, 7, 1, 4, 0, 10, 6},        {15, 17, 11, 14, 10, 110, 16},
      {25, 27, 21, 24, 20, 210, 26}, {35, 37, 31, 34, 30, 310, 36},
      {45, 47, 41, 44, 40, 410, 46}, {35, 37, 31, 34, 30, 310, 36}};

  mpc::utils::simulate(
      2, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
        SPUContext sctx = makeSPUContextWithProfile(prot, field, lctx);

        std::vector<Value> table1_columns;
        for (int64_t i = 0; i < shape_1[0]; ++i) {
          xt::xarray<uint64_t> col_data = xt::row(data_1, i);
          Value col = test::makeValue(&sctx, col_data, VIS_SECRET);
          table1_columns.push_back(col);
        }

        std::vector<Value> table2_columns;
        for (int64_t i = 0; i < shape_2[0]; ++i) {
          xt::xarray<uint64_t> col_data = xt::row(data_2, i);
          Value col = test::makeValue(&sctx, col_data, VIS_SECRET);
          table2_columns.push_back(col);
        }

        setupTrace(&sctx, sctx.config());

        auto ret = join_uu(&sctx, table1_columns, table2_columns, num_join_keys,
                           num_hash, scale_factor);

        test::printProfileData(&sctx);

        EXPECT_EQ(ret.size(), 1 + shape_1[0] + shape_2[0] - num_join_keys);

        auto valid_flag =
            hal::dump_public_as<uint64_t>(&sctx, hal::reveal(&sctx, ret[0]));
        xt::xarray<uint64_t> valid_ret;
        xt::xarray<uint64_t> data_out_expected =
            (num_join_keys == 2) ? data_out_expected_for_2_keys
                                 : data_out_expected_for_3_keys;
        for (size_t i = 1; i < ret.size(); ++i) {
          auto ret_hat =
              hal::dump_public_as<uint64_t>(&sctx, hal::reveal(&sctx, ret[i]));
          valid_ret = xt::filter(ret_hat, xt::equal(valid_flag, 1));
          EXPECT_EQ(valid_ret, xt::row(data_out_expected, i - 1))
              << "Mismatch in output column " << i - 1 << std::endl;
        }
      });
}

TEST(BigDataJoinTest, Work) {
  FieldType field = FieldType::FM64;
  ProtocolKind prot = ProtocolKind::SEMI2K;
  size_t num_join_keys = 1;
  const size_t num_hash = 3;
  const double scale_factor = 1.2;

  int64_t n = 1000000;
  const Shape shape_1 = {2, n};
  const Shape shape_2 = {2, n};
  xt::xarray<uint64_t> data_1 = xt::random::randint<uint64_t>(shape_1, 0);
  xt::xarray<uint64_t> data_2 = xt::random::randint<uint64_t>(shape_2, 0);
  for (auto i = 0; i < shape_1[1]; ++i) {
    data_1(0, i) = i;
    data_1(1, i) = i + 100;
  }
  for (auto i = shape_2[1] - 1; i >= 0; --i) {
    data_2(0, i) = i;
    data_2(1, i) = i + 200;
  }

  mpc::utils::simulate(
      2, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
        SPUContext sctx = makeSPUContextWithProfile(prot, field, lctx);

        std::vector<Value> table1_columns;
        for (int64_t i = 0; i < shape_1[0]; ++i) {
          xt::xarray<uint64_t> col_data = xt::row(data_1, i);
          Value col = test::makeValue(&sctx, col_data, VIS_SECRET);
          table1_columns.push_back(col);
        }

        std::vector<Value> table2_columns;
        for (int64_t i = 0; i < shape_2[0]; ++i) {
          xt::xarray<uint64_t> col_data = xt::row(data_2, i);
          Value col = test::makeValue(&sctx, col_data, VIS_SECRET);
          table2_columns.push_back(col);
        }

        setupTrace(&sctx, sctx.config());

        auto send_bytes_start_ = lctx->GetStats()->sent_bytes.load();
        auto recv_bytes_start_ = lctx->GetStats()->recv_bytes.load();
        auto send_actions_start_ = lctx->GetStats()->sent_actions.load();
        auto recv_actions_start_ = lctx->GetStats()->recv_actions.load();

        auto ret = join_uu(&sctx, table1_columns, table2_columns, num_join_keys,
                           num_hash, scale_factor);

        auto send_bytes_end_ = lctx->GetStats()->sent_bytes.load();
        auto recv_bytes_end_ = lctx->GetStats()->recv_bytes.load();
        auto send_actions_end_ = lctx->GetStats()->sent_actions.load();
        auto recv_actions_end_ = lctx->GetStats()->recv_actions.load();

        test::printProfileData(&sctx);

        if (lctx->Rank() == 0) {
          std::cout << "Join send bytes: "
                    << send_bytes_end_ - send_bytes_start_ << std::endl;
          std::cout << "Join recv bytes: "
                    << recv_bytes_end_ - recv_bytes_start_ << std::endl;
          std::cout << "Join send actions: "
                    << send_actions_end_ - send_actions_start_ << std::endl;
          std::cout << "Join recv actions: "
                    << recv_actions_end_ - recv_actions_start_ << std::endl;
        }
      });
}

class JoinunTest
    : public ::testing::TestWithParam<std::tuple<FieldType, ProtocolKind>> {};

INSTANTIATE_TEST_SUITE_P(
    MultiKeyJoinunTestInstances, JoinunTest,
    testing::Combine(testing::Values(FieldType::FM64),
                     testing::Values(ProtocolKind::SEMI2K)),
    [](const testing::TestParamInfo<JoinunTest::ParamType>& p) {
      return fmt::format("{}x{}", std::get<0>(p.param), std::get<1>(p.param));
    });

TEST_P(JoinunTest, Work) {
  FieldType field = std::get<0>(GetParam());
  ProtocolKind prot = std::get<1>(GetParam());
  size_t num_join_keys = 1;

  const Shape shape_1 = {2, 9};
  const Shape shape_2 = {2, 12};

  xt::xarray<uint64_t> data_1 = {{1, 4, 8, 5, 2, 6, 7, 0, 10},
                                 {11, 14, 18, 15, 12, 16, 17, 10, 110}};
  xt::xarray<uint64_t> data_2 = {
      {7, 5, 7, 9, 1, 4, 0, 4, 7, 6, 13, 14},
      {17, 15, 16, 19, 11, 14, 10, 14, 17, 16, 113, 114}};

  xt::xarray<uint64_t> data_out_expected = {
      {7, 5, 7, 1, 4, 0, 4, 7, 6},
      {17, 15, 16, 11, 14, 10, 14, 17, 16},
      {17, 15, 17, 11, 14, 10, 14, 17, 16}};

  mpc::utils::simulate(
      2, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
        SPUContext sctx = makeSPUContextWithProfile(prot, field, lctx);

        std::vector<Value> table1_columns;
        for (int64_t i = 0; i < shape_1[0]; ++i) {
          xt::xarray<uint64_t> col_data = xt::row(data_1, i);
          Value col = test::makeValue(&sctx, col_data, VIS_SECRET);
          table1_columns.push_back(col);
        }

        std::vector<Value> table2_columns;
        for (int64_t i = 0; i < shape_2[0]; ++i) {
          xt::xarray<uint64_t> col_data = xt::row(data_2, i);
          Value col = test::makeValue(&sctx, col_data, VIS_SECRET);
          table2_columns.push_back(col);
        }

        // setupTrace(&sctx, sctx.config());

        auto ret =
            join_un(&sctx, table1_columns, table2_columns, num_join_keys);

        // test::printProfileData(&sctx);

        EXPECT_EQ(ret.size(), 1 + shape_2[0] + shape_1[0] - num_join_keys);

        auto valid_flag =
            hal::dump_public_as<uint64_t>(&sctx, hal::reveal(&sctx, ret[0]));
        xt::xarray<uint64_t> valid_ret;
        for (size_t i = 1; i < ret.size(); ++i) {
          auto ret_hat =
              hal::dump_public_as<uint64_t>(&sctx, hal::reveal(&sctx, ret[i]));
          valid_ret = xt::filter(ret_hat, xt::equal(valid_flag, 1));
          EXPECT_EQ(valid_ret, xt::row(data_out_expected, i - 1))
              << "Mismatch in output column " << i - 1 << std::endl;
        }
      });
}

class MultiKeyJoinunTest : public ::testing::TestWithParam<
                               std::tuple<FieldType, ProtocolKind, size_t>> {};

INSTANTIATE_TEST_SUITE_P(
    MultiKeyJoinunTestInstances, MultiKeyJoinunTest,
    testing::Combine(testing::Values(FieldType::FM64),
                     testing::Values(ProtocolKind::SEMI2K),
                     testing::Values(2, 3)),
    [](const testing::TestParamInfo<MultiKeyJoinunTest::ParamType>& p) {
      return fmt::format("{}x{}x{}", std::get<0>(p.param), std::get<1>(p.param),
                         std::get<2>(p.param));
    });

TEST_P(MultiKeyJoinunTest, Work) {
  FieldType field = std::get<0>(GetParam());
  ProtocolKind prot = std::get<1>(GetParam());
  size_t num_join_keys = std::get<2>(GetParam());

  const Shape shape_1 = {4, 9};
  const Shape shape_2 = {5, 12};

  xt::xarray<uint64_t> data_1 = {{1, 4, 8, 5, 2, 6, 7, 0, 10},
                                 {11, 14, 18, 15, 12, 16, 17, 10, 110},
                                 {21, 24, 28, 25, 22, 26, 27, 20, 210},
                                 {31, 34, 38, 35, 32, 36, 37, 30, 310}};
  xt::xarray<uint64_t> data_2 = {
      {7, 5, 7, 9, 1, 4, 0, 4, 7, 6, 13, 14},
      {17, 15, 16, 19, 11, 14, 10, 14, 17, 16, 113, 114},
      {27, 25, 26, 29, 21, 24, 20, 24, 27, 26, 213, 214},
      {37, 35, 37, 39, 31, 34, 30, 34, 37, 36, 313, 314},
      {47, 45, 47, 49, 41, 44, 40, 44, 47, 46, 413, 414}};

  xt::xarray<uint64_t> data_out_expected_for_2_keys = {
      {7, 5, 1, 4, 0, 4, 7, 6},         {17, 15, 11, 14, 10, 14, 17, 16},
      {27, 25, 21, 24, 20, 24, 27, 26}, {37, 35, 31, 34, 30, 34, 37, 36},
      {47, 45, 41, 44, 40, 44, 47, 46}, {27, 25, 21, 24, 20, 24, 27, 26},
      {37, 35, 31, 34, 30, 34, 37, 36}};

  xt::xarray<uint64_t> data_out_expected_for_3_keys = {
      {7, 5, 1, 4, 0, 4, 7, 6},         {17, 15, 11, 14, 10, 14, 17, 16},
      {27, 25, 21, 24, 20, 24, 27, 26}, {37, 35, 31, 34, 30, 34, 37, 36},
      {47, 45, 41, 44, 40, 44, 47, 46}, {37, 35, 31, 34, 30, 34, 37, 36}};

  mpc::utils::simulate(
      2, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
        SPUContext sctx = makeSPUContextWithProfile(prot, field, lctx);

        std::vector<Value> table1_columns;
        for (int64_t i = 0; i < shape_1[0]; ++i) {
          xt::xarray<uint64_t> col_data = xt::row(data_1, i);
          Value col = test::makeValue(&sctx, col_data, VIS_SECRET);
          table1_columns.push_back(col);
        }

        std::vector<Value> table2_columns;
        for (int64_t i = 0; i < shape_2[0]; ++i) {
          xt::xarray<uint64_t> col_data = xt::row(data_2, i);
          Value col = test::makeValue(&sctx, col_data, VIS_SECRET);
          table2_columns.push_back(col);
        }

        // setupTrace(&sctx, sctx.config());

        auto ret =
            join_un(&sctx, table1_columns, table2_columns, num_join_keys);

        // test::printProfileData(&sctx);

        EXPECT_EQ(ret.size(), 1 + shape_2[0] + shape_1[0] - num_join_keys);

        auto valid_flag =
            hal::dump_public_as<uint64_t>(&sctx, hal::reveal(&sctx, ret[0]));
        xt::xarray<uint64_t> valid_ret;
        xt::xarray<uint64_t> data_out_expected =
            (num_join_keys == 2) ? data_out_expected_for_2_keys
                                 : data_out_expected_for_3_keys;
        for (size_t i = 1; i < ret.size(); ++i) {
          auto ret_hat =
              hal::dump_public_as<uint64_t>(&sctx, hal::reveal(&sctx, ret[i]));
          valid_ret = xt::filter(ret_hat, xt::equal(valid_flag, 1));
          EXPECT_EQ(valid_ret, xt::row(data_out_expected, i - 1))
              << "Mismatch in output column " << i - 1 << std::endl;
        }
      });
}

}  // namespace spu::kernel::hal
