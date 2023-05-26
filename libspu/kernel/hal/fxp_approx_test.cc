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

#include "libspu/kernel/hal/fxp_approx.h"

#include "gtest/gtest.h"
#include "xtensor/xio.hpp"

#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/test_util.h"

namespace spu::kernel::hal {

TEST(FxpTest, ExponentialPublic) {
  SPUContext ctx = test::makeSPUContext();

  xt::xarray<float> x = {
      -500.0, -100.0, -16.7, -14.3, -12.5, -11.0, -9.9, -6.7,
      -3.0,   -1.0,   -0.5,  0.5,   1.0,   1.5,   1.7,  2.1,
      6.7,    8.0,    10.5,  12.5,  14.3,  16.7,  18.0, 20.0,
  };

  Value a = constant(&ctx, x, DT_F32);
  Value c = f_exp(&ctx, a);
  EXPECT_EQ(c.dtype(), DT_F32);

  auto y = dump_public_as<float>(&ctx, c);
  EXPECT_TRUE(xt::allclose(xt::exp(x), y, 0.01, 0.001))
      << xt::exp(x) << std::endl
      << y;
}

TEST(FxpTest, ExponentialTaylorSeries) {
  SPUContext ctx = test::makeSPUContext();

  // GIVEN
  xt::xarray<float> x = {
      // -600.0 fail
      -500.0, -100.0, -16.7, -14.3, -12.5, -11.0, -9.9, -6.7,
      -3.0,   -1.0,   -0.5,  0.5,   1.0,   1.5,   1.7,  2.1,
      // 2.2 fail
  };

  Value a = test::makeValue(&ctx, x, VIS_SECRET);
  Value c = detail::exp_taylor_series(&ctx, a);
  EXPECT_EQ(c.dtype(), DT_F32);

  auto y = dump_public_as<float>(&ctx, reveal(&ctx, c));
  EXPECT_TRUE(xt::allclose(xt::exp(x), y, 0.01, 0.001))
      << xt::exp(x) << std::endl
      << y;
}

TEST(FxpTest, ExponentialPade) {
  SPUContext ctx = test::makeSPUContext();

  xt::xarray<float> x = xt::linspace<float>(-22., 22., 4000);

  Value a = test::makeValue(&ctx, x, VIS_SECRET);
  Value c = detail::exp_pade_approx(&ctx, a);
  EXPECT_EQ(c.dtype(), DT_F32);

  auto y = dump_public_as<float>(&ctx, reveal(&ctx, c));
  EXPECT_TRUE(xt::allclose(xt::exp(x), y, 0.01, 0.001))
      << xt::exp(x) << std::endl
      << y;
}

TEST(FxpTest, Log) {
  // GIVEN
  SPUContext ctx = test::makeSPUContext();

  xt::xarray<float> x = {{0.05, 0.5}, {5, 50}};
  // public log
  {
    Value a = constant(&ctx, x, DT_F32);
    Value c = f_log(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_F32);

    auto y = dump_public_as<float>(&ctx, c);
    EXPECT_TRUE(xt::allclose(xt::log(x), y, 0.01, 0.001))
        << xt::log(x) << std::endl
        << y;
  }

  // secret log
  {
    Value a = test::makeValue(&ctx, x, VIS_SECRET);
    Value c = f_log(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_F32);

    auto y = dump_public_as<float>(&ctx, reveal(&ctx, c));
    // low precision
    EXPECT_TRUE(xt::allclose(xt::log(x), y, 0.01, 0.001))
        << xt::log(x) << std::endl
        << y;
  }
}

TEST(FxpTest, Log2) {
  // GIVEN
  SPUContext ctx = test::makeSPUContext();

  xt::xarray<float> x = {{0.05, 0.5}, {5, 50}};
  // public log
  {
    Value a = constant(&ctx, x, DT_F32);
    Value c = f_log2(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_F32);

    auto y = dump_public_as<float>(&ctx, c);
    EXPECT_TRUE(xt::allclose(xt::log2(x), y, 0.01, 0.001))
        << xt::log2(x) << std::endl
        << y;
  }

  // secret log
  {
    Value a = test::makeValue(&ctx, x, VIS_SECRET);
    Value c = f_log2(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_F32);

    auto y = dump_public_as<float>(&ctx, reveal(&ctx, c));
    // low precision
    EXPECT_TRUE(xt::allclose(xt::log2(x), y, 0.01, 0.001))
        << xt::log2(x) << std::endl
        << y;
  }
}

TEST(FxpTest, Log1p) {
  // GIVEN
  SPUContext ctx = test::makeSPUContext();

  xt::xarray<float> x = {{0.5, 2.0}, {0.9, 1.8}};

  // public log1p
  {
    Value a = constant(&ctx, x, DT_F32);
    Value c = f_log1p(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_F32);

    auto y = dump_public_as<float>(&ctx, c);
    EXPECT_TRUE(xt::allclose(xt::log1p(x), y, 0.01, 0.001))
        << xt::log1p(x) << std::endl;
  }

  // secret log1p
  {
    Value a = test::makeValue(&ctx, x, VIS_SECRET);
    Value c = f_log1p(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_F32);

    auto y = dump_public_as<float>(&ctx, reveal(&ctx, c));
    // low precision
    EXPECT_TRUE(xt::allclose(xt::log1p(x), y, 0.01, 0.001))
        << xt::log1p(x) << std::endl
        << y;
  }
}

TEST(FxpTest, Exp2) {
  // GIVEN
  SPUContext ctx = test::makeSPUContext();

  xt::xarray<float> x = {{0.1, 0.2, 0.5, 0.7, 0.9},
                         {-0.1, -0.2, -0.5, -0.7, -0.9},
                         {10.1, 20.2, 25.5, 26.7, 26.9},
                         {1.1, 5.2, 7.5, 9.7, 15.9},
                         {-1.1, -5.2, -7.5, -9.7, -15.9}};

  // secret exp
  {
    Value a = test::makeValue(&ctx, x, VIS_SECRET);
    Value c = f_exp2(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_F32);

    auto y = dump_public_as<float>(&ctx, reveal(&ctx, c));
    EXPECT_TRUE(xt::allclose(xt::exp2(x), y, 0.01, 0.001))
        << xt::exp2(x) << std::endl
        << y;
  }
}

TEST(FxpTest, Tanh) {
  // GIVEN
  SPUContext ctx = test::makeSPUContext();

  xt::xarray<float> x = {{0.1, 0.2, 0.5, 0.7, 0.9, 2.1, 2.5, 4.0},
                         {-0.1, -0.2, -0.5, -0.7, -0.9, -2.1, -2.5, -4.0}};

  // secret exp
  {
    Value a = test::makeValue(&ctx, x, VIS_SECRET);
    Value c = f_tanh(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_F32);

    auto y = dump_public_as<float>(&ctx, reveal(&ctx, c));
    EXPECT_TRUE(xt::allclose(xt::tanh(x), y, 0.01, 0.001))
        << xt::tanh(x) << std::endl
        << y;
  }
}

TEST(FxpTest, Rsqrt) {
  // GIVEN
  xt::xarray<float> x = {0.36, 1.25, 2.5, 32, 123, 234.75, 556.6, 12142};
  xt::xarray<float> expected_y = 1.0F / xt::sqrt(x);

  // fxp_fraction_bits = 18(default value for FM64)
  {
    SPUContext ctx = test::makeSPUContext();

    Value a = test::makeValue(&ctx, x, VIS_SECRET);
    Value c = f_rsqrt(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_F32);

    auto y = dump_public_as<float>(&ctx, reveal(&ctx, c));
    EXPECT_TRUE(xt::allclose(expected_y, y, 0.01, 0.001))
        << expected_y << std::endl
        << y;
  }

  // fxp_fraction_bits = 17
  {
    RuntimeConfig config;
    config.set_protocol(ProtocolKind::REF2K);
    config.set_field(FieldType::FM64);
    config.set_fxp_fraction_bits(17);
    SPUContext ctx = test::makeSPUContext(config, nullptr);

    Value a = test::makeValue(&ctx, x, VIS_SECRET);
    Value c = f_rsqrt(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_F32);

    auto y = dump_public_as<float>(&ctx, reveal(&ctx, c));
    EXPECT_TRUE(xt::allclose(expected_y, y, 0.01, 0.001))
        << expected_y << std::endl
        << y;
  }

  {
    RuntimeConfig config;
    config.set_protocol(ProtocolKind::REF2K);
    config.set_field(FieldType::FM64);
    config.set_fxp_fraction_bits(16);
    SPUContext ctx = test::makeSPUContext(config, nullptr);

    xt::random::seed(0);
    xt::xarray<float> x = xt::random::rand<float>({200, 1}, 256, 4096);
    Value a = test::makeValue(&ctx, x, VIS_SECRET);
    Value c = f_rsqrt(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_F32);

    auto z = dump_public_as<float>(&ctx, reveal(&ctx, c));
    auto e = 1.0F / xt::sqrt(x);
    auto r = xt::abs(z - e) / e * 100;
    auto mm = xt::minmax(r)();
    SPDLOG_INFO("err radio [{} , {}]", mm[0], mm[1]);
  }
}

TEST(FxpTest, Sqrt) {
  // GIVEN
  xt::xarray<float> x = {0.36, 1.25, 2.5, 32, 123, 234.75, 556.6, 12142};
  xt::xarray<float> expected_y = xt::sqrt(x);

  // fxp_fraction_bits = 18(default value for FM64)
  {
    SPUContext ctx = test::makeSPUContext();

    Value a = test::makeValue(&ctx, x, VIS_SECRET);
    Value c = f_sqrt(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_F32);

    auto y = dump_public_as<float>(&ctx, reveal(&ctx, c));
    EXPECT_TRUE(xt::allclose(expected_y, y, 0.01, 0.001))
        << expected_y << std::endl
        << y;
  }

  // fxp_fraction_bits = 17
  {
    RuntimeConfig config;
    config.set_protocol(ProtocolKind::REF2K);
    config.set_field(FieldType::FM64);
    config.set_fxp_fraction_bits(17);
    SPUContext ctx = test::makeSPUContext(config, nullptr);

    Value a = test::makeValue(&ctx, x, VIS_SECRET);
    Value c = f_sqrt(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_F32);

    auto y = dump_public_as<float>(&ctx, reveal(&ctx, c));
    EXPECT_TRUE(xt::allclose(expected_y, y, 0.01, 0.001))
        << expected_y << std::endl
        << y;
  }
}

}  // namespace spu::kernel::hal
