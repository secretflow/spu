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

#include "experimental/squirrel/objectives.h"

#include <random>

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xview.hpp"

#include "libspu/core/xt_helper.h"
#include "libspu/device/io.h"
#include "libspu/kernel/hlo/casting.h"
#include "libspu/mpc/factory.h"
#include "libspu/mpc/utils/simulate.h"

namespace squirrel::test {

class ObjectivesTest
    : public ::testing::TestWithParam<
          std::tuple<spu::FieldType, std::tuple<size_t, size_t>>> {};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, ObjectivesTest,
    testing::Combine(testing::Values(spu::FM64, spu::FM128),
                     testing::Values(std::make_tuple<size_t>(8, 51),
                                     std::make_tuple<size_t>(11, 46),
                                     std::make_tuple<size_t>(42, 14))),
    [](const testing::TestParamInfo<ObjectivesTest::ParamType>& p) {
      return fmt::format("{}x{}x{}", std::get<0>(p.param),
                         std::get<0>(std::get<1>(p.param)),
                         std::get<1>(std::get<1>(p.param)));
    });

template <typename T>
spu::Value infeed(spu::SPUContext* hctx, const xt::xarray<T>& ds) {
  spu::device::ColocatedIo cio(hctx);
  if (hctx->lctx()->Rank() == 0) {
    cio.hostSetVar(fmt::format("x-{}", hctx->lctx()->Rank()), ds);
  }
  cio.sync();
  auto x = cio.deviceGetVar("x-0");
  return x;
}

TEST_P(ObjectivesTest, MaxGain) {
  using namespace spu;
  using namespace spu::kernel;
  using namespace spu::mpc;

  auto field = std::get<0>(GetParam());
  auto shape = std::get<1>(GetParam());

  std::default_random_engine rdv(std::time(0));
  std::uniform_real_distribution<double> uniform_n(1, 10.);
  // accumulated Gs, Hs is linearly with the samples, and thus could be huge.
  std::uniform_real_distribution<double> uniform_N(10000., 100000.);
  std::uniform_real_distribution<double> uniform_p(0., 1.);

  std::vector<size_t> _shape = {std::get<0>(shape), std::get<1>(shape)};
  xt::xarray<double> _G(_shape);
  xt::xarray<double> _H(_shape);

  for (size_t i = 0; i < _shape[0]; ++i) {
    for (size_t j = 0; j < _shape[1]; ++j) {
      // simulate sigmoid output
      double p = uniform_p(rdv);
      // simulate a label
      int y = p >= 0.5 ? 1 : 0;
      // simulate #samples ie accumulation
      double n = j & 1 ? uniform_n(rdv) : uniform_N(rdv);

      _G(i, j) = n * (y - p);
      _H(i, j) = n * (p * (1. - p));
      if (j > 0) {
        _G(i, j) += _G(i, j - 1);
        _H(i, j) += _H(i, j - 1);
      }
    }
  }

  // The last column is the sum of all
  auto _GA = xt::view(xt::col(_G, _shape[1] - 1), xt::all(), xt::newaxis());
  auto _HA = xt::view(xt::col(_H, _shape[1] - 1), xt::all(), xt::newaxis());
  auto _GL = _G;
  auto _HL = _H;
  auto _GR = _GA - _GL;
  auto _HR = _HA - _HL;

  const double lambda = 1e-4;
  auto objL = xt::abs(_GL) / xt::sqrt(_HL + lambda);
  auto objR = xt::abs(_GR) / xt::sqrt(_HR + lambda);
  auto objA = xt::view(xt::col(objL, _shape[1] - 1), xt::all(), xt::newaxis());
  xt::xarray<double> gain = objL + objR - objA;
  auto expected_best_splits = xt::argsort(gain, 1);

  spu::mpc::utils::simulate(2, [&](std::shared_ptr<yacl::link::Context> lctx) {
    spu::RuntimeConfig rt_config;
    rt_config.set_protocol(ProtocolKind::REF2K);
    rt_config.set_field(field);
    rt_config.set_fxp_fraction_bits(16);

    auto _ctx = std::make_unique<spu::SPUContext>(rt_config, lctx);
    auto ctx = _ctx.get();
    spu::mpc::Factory::RegisterProtocol(ctx, lctx);

    auto G = infeed<double>(ctx, _G);
    auto H = infeed<double>(ctx, _H);
    auto _gain = MaxGainOnLevel(ctx, G, H, lambda);
    _gain = hlo::Reveal(ctx, _gain);
    if (lctx->Rank() == 0) return;

    double max_gain_rel = 0.0;
    double max_gain_rel_2 = 0.0;
    for (size_t i = 0; i < _shape[0]; ++i) {
      int32_t expected = expected_best_splits(i, _shape[1] - 1);
      int32_t expected_second = expected_best_splits(i, _shape[1] - 2);
      int32_t got = _gain.data().at<int32_t>(i);

      if (expected != got) {
        auto diff = std::abs(gain(i, expected) - gain(i, got));
        max_gain_rel =
            std::max(max_gain_rel, diff / std::abs(gain(i, expected)));
      }

      if (expected_second != got) {
        auto diff = std::abs(gain(i, expected_second) - gain(i, got));
        max_gain_rel_2 =
            std::max(max_gain_rel_2, diff / std::abs(gain(i, expected_second)));
      }
    }

    // relative error within 6%
    ASSERT_LT(max_gain_rel, 6e-2);
    ASSERT_LT(max_gain_rel_2, 6e-2);
  });
}

TEST_P(ObjectivesTest, Logistic) {
  using namespace spu;
  using namespace spu::kernel;
  using namespace spu::mpc;
  auto field = std::get<0>(GetParam());
  auto shape = std::get<1>(GetParam());

  std::default_random_engine rdv(std::time(0));
  std::uniform_real_distribution<double> uniform(-7.0, 7.0);
  std::uniform_real_distribution<double> uniform_large(-16.0, 16.0);

  std::vector<size_t> _shape = {std::get<0>(shape) * std::get<1>(shape)};
  xt::xarray<double> _x(_shape);
  size_t numel = _x.size();
  std::generate_n(_x.data(), numel / 2, [&]() { return uniform(rdv); });
  std::generate_n(_x.data() + numel / 2, numel - numel / 2,
                  [&]() { return uniform_large(rdv); });

  spu::mpc::utils::simulate(2, [&](std::shared_ptr<yacl::link::Context> lctx) {
    spu::RuntimeConfig rt_config;
    rt_config.set_protocol(ProtocolKind::CHEETAH);
    rt_config.mutable_cheetah_2pc_config()->set_ot_kind(
        CheetahOtKind::YACL_Softspoken);
    rt_config.set_field(field);
    rt_config.set_fxp_fraction_bits(17);
    rt_config.set_enable_hal_profile(true);
    rt_config.set_enable_pphlo_profile(true);

    auto _ctx = std::make_unique<spu::SPUContext>(rt_config, lctx);
    auto ctx = _ctx.get();
    spu::mpc::Factory::RegisterProtocol(ctx, lctx);

    auto x = infeed<double>(ctx, _x);

    size_t bytes_sent = lctx->GetStats()->sent_bytes;
    auto logistic = Logistic(ctx, x);
    bytes_sent = lctx->GetStats()->sent_bytes - bytes_sent;

    SPDLOG_INFO("Logistic {} elements Rank {} send {} bytes per", numel,
                lctx->Rank(), bytes_sent * 1. / numel);

    SPU_ENFORCE_EQ(logistic.numel(), x.numel());

    logistic = hlo::Reveal(ctx, logistic);
    if (lctx->Rank() == 0) {
      return;
    }

    double fxp = std::pow(2., rt_config.fxp_fraction_bits());
    double max_err = 0.;

    for (int64_t i = 0; i < logistic.numel(); ++i) {
      double expected = 1. / (1. + std::exp(-_x[i]));
      double got = logistic.data().at<int64_t>(i) / fxp;

      // NOTE(lwj): we assume the approximated value is within (0, 1)
      // so that we can lighten the computation for Gain.
      EXPECT_LT(got, 1.);
      EXPECT_GT(got, 0.);
      double e = std::abs(got - expected);
      if (e > max_err) {
        max_err = e;
      }
    }

    // printf("\nmax error %f\n", max_err);
    EXPECT_LT(max_err, 0.03);
  });
}

TEST_P(ObjectivesTest, Sigmoid) {
  using namespace spu;
  using namespace spu::kernel;
  using namespace spu::mpc;
  auto field = std::get<0>(GetParam());
  auto shape = std::get<1>(GetParam());
  // shape = {1, 10000};

  std::default_random_engine rdv(std::time(0));
  std::uniform_real_distribution<double> uniform(-7.0, 7.0);
  std::uniform_real_distribution<double> uniform_large(-16.0, 16.0);

  std::vector<size_t> _shape = {std::get<0>(shape) * std::get<1>(shape)};
  xt::xarray<double> _x(_shape);
  size_t numel = _x.size();
  std::generate_n(_x.data(), numel / 2, [&]() { return uniform(rdv); });
  std::generate_n(_x.data() + numel / 2, numel - numel / 2,
                  [&]() { return uniform_large(rdv); });

  spu::mpc::utils::simulate(2, [&](std::shared_ptr<yacl::link::Context> lctx) {
    spu::RuntimeConfig rt_config;
    rt_config.set_protocol(ProtocolKind::CHEETAH);
    rt_config.mutable_cheetah_2pc_config()->set_ot_kind(
        CheetahOtKind::YACL_Softspoken);
    rt_config.set_field(field);
    rt_config.set_fxp_fraction_bits(17);
    rt_config.set_enable_hal_profile(true);
    rt_config.set_enable_pphlo_profile(true);

    auto _ctx = std::make_unique<spu::SPUContext>(rt_config, lctx);
    auto ctx = _ctx.get();
    spu::mpc::Factory::RegisterProtocol(ctx, lctx);

    auto x = infeed<double>(ctx, _x);

    size_t bytes_sent = lctx->GetStats()->sent_bytes;
    auto logistic = Sigmoid(ctx, x);
    bytes_sent = lctx->GetStats()->sent_bytes - bytes_sent;

    SPDLOG_INFO("Sigmoid {} elements {} bytes per", numel,
                bytes_sent * 1. / numel);

    SPU_ENFORCE_EQ(logistic.numel(), x.numel());

    logistic = hlo::Reveal(ctx, logistic);
    if (lctx->Rank() == 0) {
      return;
    }

    double fxp = std::pow(2., rt_config.fxp_fraction_bits());
    double max_err = 0.;

    for (int64_t i = 0; i < logistic.numel(); ++i) {
      double expected = 0.5 + 0.5 * _x[i] / std::sqrt(1 + _x[i] * _x[i]);
      double got = logistic.data().at<int64_t>(i) / fxp;
      // NOTE(lwj): we assume the approximated value is within (0, 1)
      // so that we can lighten the computation for Gain.
      EXPECT_LT(got, 1.);
      EXPECT_GT(got, 0.);
      double e = std::abs(got - expected);
      if (e > max_err) {
        max_err = e;
      }
    }

    // printf("\nmax error %f\n", max_err);
    EXPECT_LT(max_err, 0.005);
  });
}

}  // namespace squirrel::test
