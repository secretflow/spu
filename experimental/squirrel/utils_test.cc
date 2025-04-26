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

#include "experimental/squirrel/utils.h"

#include <random>

#include "gtest/gtest.h"
#include "xtensor/xsort.hpp"

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/xt_helper.h"
#include "libspu/device/io.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/kernel/hlo/casting.h"
#include "libspu/mpc/factory.h"
#include "libspu/mpc/utils/simulate.h"

namespace squirrel::test {
class UtilsTest : public ::testing::Test {};

template <typename T>
spu::Value infeed(spu::SPUContext* ctx, const xt::xarray<T>& ds,
                  bool need_shared = false) {
  spu::device::ColocatedIo cio(ctx);
  if (ctx->lctx()->Rank() == 0) {
    cio.hostSetVar(fmt::format("x-{}", ctx->lctx()->Rank()), ds);
  }
  cio.sync();
  auto x = cio.deviceGetVar("x-0");

  if (need_shared && not x.isSecret()) {
    x = spu::kernel::hlo::Cast(ctx, x, spu::Visibility::VIS_SECRET, x.dtype());
  }
  return x;
}

TEST_F(UtilsTest, ReduceSum) {
  using namespace spu;
  using namespace spu::kernel;
  using namespace spu::mpc;
  spu::FieldType field = spu::FM64;
  spu::Shape shape = {3, 5, 4};

  std::default_random_engine rdv(std::time(0));
  std::uniform_real_distribution<double> uniform(-1024., 1024.);

  std::vector<size_t> _shape;
  for (int i = 0; i < shape.ndim(); ++i) {
    _shape.push_back((size_t)shape[i]);
  }
  xt::xarray<double> _x(_shape);
  std::generate_n(_x.data(), _x.size(), [&]() { return 0.1 + uniform(rdv); });

  spu::mpc::utils::simulate(2, [&](std::shared_ptr<yacl::link::Context> lctx) {
    spu::RuntimeConfig rt_config;
    rt_config.protocol = ProtocolKind::REF2K;
    rt_config.field = field;
    rt_config.fxp_fraction_bits = 16;

    auto _ctx = std::make_unique<spu::SPUContext>(rt_config, lctx);
    auto* ctx = _ctx.get();
    spu::mpc::Factory::RegisterProtocol(ctx, lctx);
    auto x = infeed(ctx, _x);

    for (int axis = 0; axis < shape.ndim(); ++axis) {
      auto expected = spu::xt_to_ndarray(xt::sum(_x, {axis}));
      auto got = ReduceSum(ctx, x, axis);
      got = hlo::Reveal(ctx, got);

      ASSERT_EQ(expected.numel(), got.numel());

      if (lctx->Rank() == 0) {
        const double fxp = std::pow(2., rt_config.fxp_fraction_bits);
        auto flatten = got.data().reshape({got.numel()});

        DISPATCH_ALL_FIELDS(field, [&]() {
          using s2k = std::make_signed<ring2k_t>::type;
          NdArrayView<s2k> got(flatten);
          for (int64_t i = 0; i < expected.numel(); ++i) {
            ASSERT_NEAR(expected.at<double>(i), got[i] / fxp, 1024. / fxp);
          }
        });
      }
    }
  });
}

TEST_F(UtilsTest, ArgMax) {
  using namespace spu;
  using namespace spu::kernel;
  using namespace spu::mpc;
  spu::FieldType field = spu::FM64;
  spu::Shape shape = {3, 5, 4};

  std::default_random_engine rdv(std::time(0));
  std::uniform_real_distribution<double> uniform(-1024., 1024.);

  std::vector<size_t> _shape;
  for (int i = 0; i < shape.ndim(); ++i) {
    _shape.push_back((size_t)shape[i]);
  }

  xt::xarray<double> _x(_shape);
  std::generate_n(_x.data(), _x.size(), [&]() { return 0.1 + uniform(rdv); });

  spu::mpc::utils::simulate(2, [&](std::shared_ptr<yacl::link::Context> lctx) {
    spu::RuntimeConfig rt_config;
    rt_config.protocol = ProtocolKind::REF2K;
    rt_config.field = field;
    rt_config.fxp_fraction_bits = 16;

    auto _ctx = std::make_unique<spu::SPUContext>(rt_config, lctx);
    auto* ctx = _ctx.get();
    spu::mpc::Factory::RegisterProtocol(ctx, lctx);
    auto x = infeed(ctx, _x);

    for (int axis = 0; axis < shape.ndim(); ++axis) {
      auto expected = xt::argmax(_x, axis);
      auto got = ArgMax(ctx, x, axis);
      got = hlo::Reveal(ctx, got);

      ASSERT_EQ(expected.size(), (size_t)got.numel());

      if (lctx->Rank() == 0) {
        auto flatten = got.data().reshape({got.numel()});

        DISPATCH_ALL_FIELDS(field, [&]() {
          NdArrayView<ring2k_t> got(flatten);
          for (size_t i = 0; i < expected.size(); ++i) {
            ASSERT_EQ(expected(i), got[i]);
          }
        });
      }
    }
  });
}

TEST_F(UtilsTest, MulA1BV) {
  using namespace spu;
  using namespace spu::kernel;
  using namespace spu::mpc;
  spu::FieldType field = spu::FM64;
  spu::Shape shape = {100, 200};

  std::default_random_engine rdv(std::time(0));
  std::uniform_real_distribution<double> uniform(-1024., 1024.);

  std::vector<size_t> _shape;
  for (int i = 0; i < shape.ndim(); ++i) {
    _shape.push_back((size_t)shape[i]);
  }

  xt::xarray<double> _x(_shape);
  std::vector<uint8_t> ind(_x.size());
  std::generate_n(_x.data(), _x.size(), [&]() { return 0.1 + uniform(rdv); });
  std::generate_n(ind.data(), ind.size(),
                  [&]() -> uint8_t { return uniform(rdv) > 0; });

  spu::mpc::utils::simulate(2, [&](std::shared_ptr<yacl::link::Context> lctx) {
    spu::RuntimeConfig rt_config;
    rt_config.protocol = ProtocolKind::CHEETAH;
    rt_config.field = field;
    rt_config.fxp_fraction_bits = 16;
    rt_config.experimental_enable_colocated_optimization = true;
    rt_config.enable_hal_profile = true;
    rt_config.cheetah_2pc_config.ot_kind = CheetahOtKind::YACL_Softspoken;

    auto _ctx = std::make_unique<spu::SPUContext>(rt_config, lctx);
    auto* ctx = _ctx.get();
    spu::mpc::Factory::RegisterProtocol(ctx, lctx);
    auto x = infeed(ctx, _x, /*shared*/ true);

    size_t sent = lctx->GetStats()->sent_bytes;
    spu::Value c = lctx->Rank() == 0
                       ? MulArithShareWithPrivateBoolean(ctx, x, ind)
                       : MulArithShareWithPrivateBoolean(ctx, x);
    sent = lctx->GetStats()->sent_bytes - sent;
    SPDLOG_INFO("MulA1BV {} bytes {} bits per", sent, sent * 8. / ind.size());

    c = hlo::Reveal(ctx, c);

    if (lctx->Rank() == 0) {
      double scale = std::pow(2., rt_config.fxp_fraction_bits);
      for (int64_t i = 0; i < c.numel(); ++i) {
        if (ind[i]) {
          ASSERT_NEAR(c.data().at<int64_t>(i) / scale, _x[i], 2. / scale);
        } else {
          ASSERT_EQ(c.data().at<uint64_t>(i), 0);
        }
      }
    }
  });
}

TEST_F(UtilsTest, MulA1B_AND_style) {
  using namespace spu;
  using namespace spu::kernel;
  using namespace spu::mpc;
  spu::FieldType field = spu::FM64;
  spu::Shape shape = {124, 212};

  std::default_random_engine rdv(std::time(0));
  std::uniform_real_distribution<double> uniform(-1024., 1024.);

  std::vector<size_t> _shape;
  for (int i = 0; i < shape.ndim(); ++i) {
    _shape.push_back((size_t)shape[i]);
  }

  xt::xarray<double> _x(_shape);
  std::vector<uint8_t> ind[2];
  std::generate_n(_x.data(), _x.size(), [&]() { return 0.1 + uniform(rdv); });
  ind[0].resize(_x.size());
  ind[1].resize(_x.size());
  std::generate_n(ind[0].data(), ind[0].size(),
                  [&]() -> uint8_t { return uniform(rdv) > 0; });
  std::generate_n(ind[1].data(), ind[1].size(),
                  [&]() -> uint8_t { return uniform(rdv) > 0; });

  spu::mpc::utils::simulate(2, [&](std::shared_ptr<yacl::link::Context> lctx) {
    spu::RuntimeConfig rt_config;
    rt_config.protocol = ProtocolKind::CHEETAH;
    rt_config.field = field;
    rt_config.fxp_fraction_bits = 16;
    rt_config.experimental_enable_colocated_optimization = true;
    rt_config.enable_hal_profile = true;
    rt_config.cheetah_2pc_config.ot_kind = CheetahOtKind::YACL_Softspoken;

    auto _ctx = std::make_unique<spu::SPUContext>(rt_config, lctx);
    auto* ctx = _ctx.get();
    spu::mpc::Factory::RegisterProtocol(ctx, lctx);
    auto x = infeed(ctx, _x, /*shared*/ true);

    size_t sent = lctx->GetStats()->sent_bytes;
    spu::Value c = MulArithShareWithANDBoolShare(ctx, x, ind[lctx->Rank()]);
    sent = lctx->GetStats()->sent_bytes - sent;
    SPDLOG_INFO("MulA1B {} bytes {} bits per", sent, sent * 8. / shape.numel());

    c = hlo::Reveal(ctx, c);

    if (lctx->Rank() == 0) {
      double scale = std::pow(2., rt_config.fxp_fraction_bits);
      for (int64_t i = 0; i < c.numel(); ++i) {
        if (ind[0][i] & ind[1][i]) {
          ASSERT_NEAR(c.data().at<int64_t>(i) / scale, _x[i], 2. / scale);
        } else {
          ASSERT_EQ(c.data().at<uint64_t>(i), 0);
        }
      }
    }
  });
}

TEST_F(UtilsTest, BatchMulA1B_AND_style) {
  using namespace spu;
  using namespace spu::kernel;
  using namespace spu::mpc;
  spu::FieldType field = spu::FM64;

  int64_t num_batch = 8;
  int64_t batch_size = 100;

  std::default_random_engine rdv(std::time(0));
  std::uniform_real_distribution<double> uniform(-1024., 1024.);

  std::vector<size_t> _shape;
  _shape.push_back(num_batch);
  xt::xarray<double> _x(_shape);
  std::vector<uint8_t> ind[2];
  std::generate_n(_x.data(), _x.size(), [&]() { return 0.1 + uniform(rdv); });
  ind[0].resize(num_batch * batch_size);
  ind[1].resize(num_batch * batch_size);
  std::generate_n(ind[0].data(), ind[0].size(),
                  [&]() -> uint8_t { return uniform(rdv) > 0; });
  std::generate_n(ind[1].data(), ind[1].size(),
                  [&]() -> uint8_t { return uniform(rdv) > 0; });

  spu::mpc::utils::simulate(2, [&](std::shared_ptr<yacl::link::Context> lctx) {
    spu::RuntimeConfig rt_config;
    rt_config.protocol = ProtocolKind::CHEETAH;
    rt_config.field = field;
    rt_config.fxp_fraction_bits = 16;
    rt_config.experimental_enable_colocated_optimization = true;
    rt_config.enable_hal_profile = true;
    rt_config.cheetah_2pc_config.ot_kind = CheetahOtKind::YACL_Softspoken;

    auto _ctx = std::make_unique<spu::SPUContext>(rt_config, lctx);
    auto* ctx = _ctx.get();
    spu::mpc::Factory::RegisterProtocol(ctx, lctx);
    auto x = infeed(ctx, _x, /*shared*/ true);

    spu::Value c = BatchMulArithShareWithANDBoolShare(ctx, x, batch_size,
                                                      ind[lctx->Rank()]);
    c = hlo::Reveal(ctx, c);

    if (lctx->Rank() == 0) {
      double scale = std::pow(2., rt_config.fxp_fraction_bits);

      for (int64_t k = 0; k < c.numel(); k += batch_size) {
        for (int64_t i = 0; i < batch_size; ++i) {
          if (ind[0][k + i] & ind[1][k + i]) {
            ASSERT_NEAR(c.data().at<int64_t>(k + i) / scale, _x[k / batch_size],
                        2. / scale);
          } else {
            ASSERT_EQ(c.data().at<uint64_t>(k + i), 0);
          }
        }
      }
    }
  });
}

}  // namespace squirrel::test
