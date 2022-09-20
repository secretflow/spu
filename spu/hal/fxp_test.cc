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

#include "spu/hal/fxp.h"

#include "gtest/gtest.h"
#include "xtensor/xio.hpp"

#include "spu/hal/constants.h"
#include "spu/hal/test_util.h"

namespace spu::hal {

TEST(FxpTest, Reciprocal) {
  // GIVEN
  HalContext ctx = test::makeRefHalContext();

  // default fxp bits is 18 for FM64.
  xt::xarray<float> x{
      {1.0, -2.0, -15000}, {-0.5, 3.14, 15000}, {10000, 60000, 260000}};

  // public reciprocal
  {
    Value a = constant(&ctx, x);
    Value c = f_reciprocal(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_FXP);

    auto y = test::dump_public_as<float>(&ctx, c);
    EXPECT_TRUE(xt::allclose(1.0f / x, y, 0.001, 0.0001))
        << (1.0 / x) << std::endl
        << y;
  }

  // secret reciprocal
  {
    Value a = const_secret(&ctx, x);
    Value c = f_reciprocal(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_FXP);

    auto y = test::dump_public_as<float>(&ctx, _s2p(&ctx, c).asFxp());
    EXPECT_TRUE(xt::allclose(1.0f / x, y, 0.001, 0.0001))
        << (1.0 / x) << std::endl
        << y;
  }
}

TEST(FxpTest, Div) {
  // GIVEN
  HalContext ctx = test::makeRefHalContext();

  xt::xarray<float> x{{1.0, -200000.0, 7000000, -0.5, 314000, 1.5}};
  xt::xarray<float> y{{1.0, 200000.0, 200000, 100, 3.14, 0.003}};

  // public div
  {
    Value a = constant(&ctx, x);
    Value b = constant(&ctx, y);
    Value c = f_div(&ctx, a, b);
    EXPECT_EQ(c.dtype(), DT_FXP);

    auto z = test::dump_public_as<float>(&ctx, c);
    EXPECT_TRUE(xt::allclose(x / y, z, 0.001, 0.0001)) << (x / y) << std::endl
                                                       << z;
  }

  // secret div
  {
    Value a = const_secret(&ctx, x);
    Value b = const_secret(&ctx, y);
    Value c = f_div(&ctx, a, b);
    EXPECT_EQ(c.dtype(), DT_FXP);

    auto z = test::dump_public_as<float>(&ctx, _s2p(&ctx, c).asFxp());
    EXPECT_TRUE(xt::allclose(x / y, z, 0.01, 0.001)) << (x / y) << std::endl
                                                     << z;
  }
}

TEST(FxpTest, ExponentialPublic) {
  HalContext ctx = test::makeRefHalContext();

  xt::xarray<float> x{
      -500.0, -100.0, -16.7, -14.3, -12.5, -11.0, -9.9, -6.7,
      -3.0,   -1.0,   -0.5,  0.5,   1.0,   1.5,   1.7,  2.1,
      6.7,    8.0,    10.5,  12.5,  14.3,  16.7,  18.0, 20.0,
  };

  Value a = constant(&ctx, x);
  Value c = f_exp(&ctx, a);
  EXPECT_EQ(c.dtype(), DT_FXP);

  auto y = test::dump_public_as<float>(&ctx, c);
  EXPECT_TRUE(xt::allclose(xt::exp(x), y, 0.01, 0.001))
      << xt::exp(x) << std::endl
      << y;
}

TEST(FxpTest, ExponentialTaylorSeries) {
  HalContext ctx = test::makeRefHalContext();

  // GIVEN
  xt::xarray<float> x{
      // -600.0 fail
      -500.0, -100.0, -16.7, -14.3, -12.5, -11.0, -9.9, -6.7,
      -3.0,   -1.0,   -0.5,  0.5,   1.0,   1.5,   1.7,  2.1,
      // 2.2 fail
  };

  Value a = const_secret(&ctx, x);
  Value c = detail::exp_taylor_series(&ctx, a);
  EXPECT_EQ(c.dtype(), DT_FXP);

  auto y = test::dump_public_as<float>(&ctx, _s2p(&ctx, c).asFxp());
  EXPECT_TRUE(xt::allclose(xt::exp(x), y, 0.01, 0.001))
      << xt::exp(x) << std::endl
      << y;
}

TEST(FxpTest, ExponentialPade) {
  HalContext ctx = test::makeRefHalContext();

  xt::xarray<float> x = xt::linspace<float>(-22., 22., 4000);

  Value a = const_secret(&ctx, x);
  Value c = detail::exp_pade_approx(&ctx, a);
  EXPECT_EQ(c.dtype(), DT_FXP);

  auto y = test::dump_public_as<float>(&ctx, _s2p(&ctx, c).asFxp());
  EXPECT_TRUE(xt::allclose(xt::exp(x), y, 0.01, 0.001))
      << xt::exp(x) << std::endl
      << y;
}

TEST(FxpTest, Log) {
  // GIVEN
  HalContext ctx = test::makeRefHalContext();

  xt::xarray<float> x{{0.05, 0.5}, {5, 50}};
  // public log
  {
    Value a = constant(&ctx, x);
    Value c = f_log(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_FXP);

    auto y = test::dump_public_as<float>(&ctx, c);
    EXPECT_TRUE(xt::allclose(xt::log(x), y, 0.01, 0.001))
        << xt::log(x) << std::endl
        << y;
  }

  // secret log
  {
    Value a = const_secret(&ctx, x);
    Value c = f_log(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_FXP);

    auto y = test::dump_public_as<float>(&ctx, _s2p(&ctx, c).asFxp());
    // low precision
    EXPECT_TRUE(xt::allclose(xt::log(x), y, 0.01, 0.001))
        << xt::log(x) << std::endl
        << y;
  }
}

TEST(FxpTest, Log2) {
  // GIVEN
  HalContext ctx = test::makeRefHalContext();

  xt::xarray<float> x{{0.05, 0.5}, {5, 50}};
  // public log
  {
    Value a = constant(&ctx, x);
    Value c = f_log2(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_FXP);

    auto y = test::dump_public_as<float>(&ctx, c);
    EXPECT_TRUE(xt::allclose(xt::log2(x), y, 0.01, 0.001))
        << xt::log2(x) << std::endl
        << y;
  }

  // secret log
  {
    Value a = const_secret(&ctx, x);
    Value c = f_log2(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_FXP);

    auto y = test::dump_public_as<float>(&ctx, _s2p(&ctx, c).asFxp());
    // low precision
    EXPECT_TRUE(xt::allclose(xt::log2(x), y, 0.01, 0.001))
        << xt::log2(x) << std::endl
        << y;
  }
}

TEST(FxpTest, Log1p) {
  // GIVEN
  HalContext ctx = test::makeRefHalContext();

  xt::xarray<float> x{{0.5, 2.0}, {0.9, 1.8}};

  // public log1p
  {
    Value a = constant(&ctx, x);
    Value c = f_log1p(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_FXP);

    auto y = test::dump_public_as<float>(&ctx, c);
    EXPECT_TRUE(xt::allclose(xt::log1p(x), y, 0.01, 0.001))
        << xt::log1p(x) << std::endl;
  }

  // secret log1p
  {
    Value a = const_secret(&ctx, x);
    Value c = f_log1p(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_FXP);

    auto y = test::dump_public_as<float>(&ctx, _s2p(&ctx, c).asFxp());
    // low precision
    EXPECT_TRUE(xt::allclose(xt::log1p(x), y, 0.01, 0.001))
        << xt::log1p(x) << std::endl
        << y;
  }
}

TEST(FxpTest, abs) {
  // GIVEN
  HalContext ctx = test::makeRefHalContext();

  xt::xarray<float> x{{0.5, -2.0}, {0.9, -1.8}};

  // public abs
  {
    Value a = constant(&ctx, x);
    Value c = f_abs(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_FXP);

    auto y = test::dump_public_as<float>(&ctx, c);
    EXPECT_TRUE(xt::allclose(xt::abs(x), y, 0.01, 0.05))
        << xt::abs(x) << std::endl
        << y;
  }

  // secret abs
  {
    Value a = const_secret(&ctx, x);
    Value c = f_abs(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_FXP);

    auto y = test::dump_public_as<float>(&ctx, _s2p(&ctx, c).asFxp());
    // low precision
    EXPECT_TRUE(xt::allclose(xt::abs(x), y, 0.1, 0.5))
        << xt::abs(x) << std::endl
        << y;
  }
}

TEST(FxpTest, floor) {
  // GIVEN
  HalContext ctx = test::makeRefHalContext();

  xt::xarray<float> x{{0.5, -0.5}, {-20.0, 31.8}, {0, 5.0}, {-5.0, -31.8}};

  // public floor
  {
    Value a = constant(&ctx, x);
    Value c = f_floor(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_FXP);

    auto y = test::dump_public_as<float>(&ctx, c);
    EXPECT_TRUE(xt::allclose(xt::floor(x), y, 0.01, 0.001))
        << xt::floor(x) << std::endl
        << y;
  }

  // secret floor
  {
    Value a = const_secret(&ctx, x);
    Value c = f_floor(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_FXP);

    auto y = test::dump_public_as<float>(&ctx, _s2p(&ctx, c).asFxp());
    // low precision
    EXPECT_TRUE(xt::allclose(xt::floor(x), y, 0.01, 0.001))
        << xt::floor(x) << std::endl
        << y;
  }
}

TEST(FxpTest, ceil) {
  // GIVEN
  HalContext ctx = test::makeRefHalContext();

  xt::xarray<float> x{{0.5, -0.5}, {-20.0, 31.8}, {0, 5.0}, {-5.0, -31.8}};

  // public ceil
  {
    Value a = constant(&ctx, x);
    Value c = f_ceil(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_FXP);

    auto y = test::dump_public_as<float>(&ctx, c);
    EXPECT_TRUE(xt::allclose(xt::ceil(x), y, 0.01, 0.001))
        << xt::ceil(x) << std::endl
        << y;
  }

  // secret ceil
  {
    Value a = const_secret(&ctx, x);
    Value c = f_ceil(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_FXP);

    auto y = test::dump_public_as<float>(&ctx, _s2p(&ctx, c).asFxp());
    // low precision
    EXPECT_TRUE(xt::allclose(xt::ceil(x), y, 0.01, 0.001))
        << xt::ceil(x) << std::endl
        << y;
  }
}

TEST(FxpTest, Exp2) {
  // GIVEN
  HalContext ctx = test::makeRefHalContext();

  xt::xarray<float> x{{0.1, 0.2, 0.5, 0.7, 0.9},
                      {-0.1, -0.2, -0.5, -0.7, -0.9},
                      {10.1, 20.2, 25.5, 26.7, 26.9},
                      {1.1, 5.2, 7.5, 9.7, 15.9},
                      {-1.1, -5.2, -7.5, -9.7, -15.9}};

  // secret exp
  {
    Value a = const_secret(&ctx, x);
    Value c = f_exp2(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_FXP);

    auto y = test::dump_public_as<float>(&ctx, _s2p(&ctx, c).asFxp());
    EXPECT_TRUE(xt::allclose(xt::exp2(x), y, 0.01, 0.001))
        << xt::exp2(x) << std::endl
        << y;
  }
}

TEST(FxpTest, Tanh) {
  // GIVEN
  HalContext ctx = test::makeRefHalContext();

  xt::xarray<float> x{{0.1, 0.2, 0.5, 0.7, 0.9, 2.1, 2.5, 4.0},
                      {-0.1, -0.2, -0.5, -0.7, -0.9, -2.1, -2.5, -4.0}};

  // secret exp
  {
    Value a = const_secret(&ctx, x);
    Value c = f_tanh(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_FXP);

    auto y = test::dump_public_as<float>(&ctx, _s2p(&ctx, c).asFxp());
    EXPECT_TRUE(xt::allclose(xt::tanh(x), y, 0.01, 0.001))
        << xt::tanh(x) << std::endl
        << y;
  }
}

TEST(FxpTest, SqrtInv) {
  // GIVEN
  xt::xarray<float> x{0.36, 1.25, 2.5, 32, 123, 234.75, 556.6, 12142};
  xt::xarray<float> expected_y = 1.0f / xt::sqrt(x);

  // fxp_fraction_bits = 18(default value for FM64)
  {
    HalContext ctx = test::makeRefHalContext();

    Value a = const_secret(&ctx, x);
    Value c = f_sqrt_inv(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_FXP);

    auto y = test::dump_public_as<float>(&ctx, _s2p(&ctx, c).asFxp());
    EXPECT_TRUE(xt::allclose(expected_y, y, 0.01, 0.001))
        << expected_y << std::endl
        << y;
  }

  // fxp_fraction_bits = 17
  {
    RuntimeConfig config;
    config.set_protocol(ProtocolKind::REF2K);
    config.set_field(FieldType::FM64);
    config.set_sigmoid_mode(RuntimeConfig::SIGMOID_REAL);
    config.set_fxp_fraction_bits(17);
    HalContext ctx = test::makeRefHalContext(config);

    Value a = const_secret(&ctx, x);
    Value c = f_sqrt_inv(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_FXP);

    auto y = test::dump_public_as<float>(&ctx, _s2p(&ctx, c).asFxp());
    EXPECT_TRUE(xt::allclose(expected_y, y, 0.01, 0.001))
        << expected_y << std::endl
        << y;
  }
}

}  // namespace spu::hal
