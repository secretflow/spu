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

#include "libspu/kernel/hal/fxp_base.h"

#include "gtest/gtest.h"
#include "xtensor/xio.hpp"  // TODO: dropme

#include "libspu/core/parallel_utils.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/test_util.h"

namespace spu::kernel::hal {
namespace {

template <typename T = float>
xt::xarray<T> xarrayMMul(const xt::xarray<T>& x, const xt::xarray<T>& y) {
  size_t m = x.shape(0);
  size_t n = y.shape(1);
  SPU_ENFORCE(x.shape(1) == y.shape(0));
  size_t k = x.shape(1);

  xt::xarray<T> ret({m, n}, static_cast<T>(0));

  pforeach(0, m * n, [&](int64_t idx) {
    size_t r = idx / n;
    size_t c = idx % n;
    T tmp = 0;
    for (size_t i = 0; i < k; i++) {
      tmp += x.at(r, i) * y.at(i, c);
    }
    ret.at(r, c) = tmp;
  });

  return ret;
}

}  // namespace

class FxpMmulTest
    : public ::testing::TestWithParam<std::tuple<size_t, size_t, size_t>> {};

INSTANTIATE_TEST_SUITE_P(
    FxpMmulTestInstance, FxpMmulTest,
    testing::Values(std::make_tuple(8UL, 9UL, 10UL)
                    // TODO: test tiling features.
                    // ,std::make_tuple(100UL, 100UL, 300000UL)
                    // ,std::make_tuple(3000UL, 3000UL, 6000UL)  //
                    ),
    [](const testing::TestParamInfo<std::tuple<size_t, size_t, size_t>>& p) {
      return fmt::format("{}x{}_{}x{}", std::get<0>(p.param),
                         std::get<2>(p.param), std::get<2>(p.param),
                         std::get<1>(p.param));
    });

// too slowly, disable for now. wait for linalg::matmul optimization.
TEST_P(FxpMmulTest, Works) {
  // GIVEN
  SPUContext ctx = test::makeSPUContext();

  size_t m = std::get<0>(GetParam());
  size_t n = std::get<1>(GetParam());
  size_t k = std::get<2>(GetParam());

  xt::random::seed(time(nullptr));
  xt::xarray<float> x = xt::random::rand<float>({m, k}, 0.25, 1);
  xt::xarray<float> y = xt::random::rand<float>({k, n}, 0.25, 1);

  auto t_z = xarrayMMul(x, y);

  {  // public
    Value a = constant(&ctx, x, DT_F32);
    Value b = constant(&ctx, y, DT_F32);
    Value c = f_mmul(&ctx, a, b);
    EXPECT_EQ(c.dtype(), DT_F32);

    auto z = dump_public_as<float>(&ctx, c);
    EXPECT_TRUE(xt::allclose(t_z, z, 0.01, 0.001)) << t_z << std::endl << z;
  }

  {  // secret
    Value a = test::makeValue(&ctx, x, VIS_SECRET);
    Value b = test::makeValue(&ctx, y, VIS_SECRET);
    Value c = f_mmul(&ctx, a, b);
    EXPECT_EQ(c.dtype(), DT_F32);

    auto z = dump_public_as<float>(&ctx, reveal(&ctx, c));
    EXPECT_TRUE(xt::allclose(t_z, z, 0.01, 0.001)) << t_z << std::endl << z;
  }
}

TEST(FxpTest, Reciprocal) {
  // GIVEN
  SPUContext ctx = test::makeSPUContext();

  // default fxp bits is 18 for FM64.
  xt::xarray<float> x = {
      {1.0, -2.0, -15000},
      {-0.5, 3.14, 15000},
      {10000, 60000, 260000},
  };

  // public reciprocal
  {
    Value a = constant(&ctx, x, DT_F32);
    Value c = f_reciprocal(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_F32);

    auto y = dump_public_as<float>(&ctx, c);
    EXPECT_TRUE(xt::allclose(1.0F / x, y, 0.001, 0.0001))
        << (1.0 / x) << std::endl
        << y;
  }

  // secret reciprocal
  {
    Value a = test::makeValue(&ctx, x, VIS_SECRET);
    Value c = f_reciprocal(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_F32);

    auto y = dump_public_as<float>(&ctx, reveal(&ctx, c));
    EXPECT_TRUE(xt::allclose(1.0F / x, y, 0.001, 0.0001))
        << (1.0 / x) << std::endl
        << y;
  }
}

TEST(FxpTest, Div) {
  // GIVEN
  SPUContext ctx = test::makeSPUContext();

  xt::xarray<float> x = {{1.0, -200000.0, 7000000, -0.5, 314000, 1.5}};
  xt::xarray<float> y = {{1.0, 200000.0, 200000, 100, 3.14, 0.003}};

  // public div
  {
    Value a = constant(&ctx, x, DT_F32);
    Value b = constant(&ctx, y, DT_F32);
    Value c = f_div(&ctx, a, b);
    EXPECT_EQ(c.dtype(), DT_F32);

    auto z = dump_public_as<float>(&ctx, c);
    EXPECT_TRUE(xt::allclose(x / y, z, 0.001, 0.0001)) << (x / y) << std::endl
                                                       << z;
  }

  // secret div
  {
    Value a = test::makeValue(&ctx, x, VIS_SECRET);
    Value b = test::makeValue(&ctx, y, VIS_SECRET);
    Value c = f_div(&ctx, a, b);
    EXPECT_EQ(c.dtype(), DT_F32);

    auto z = dump_public_as<float>(&ctx, reveal(&ctx, c));
    EXPECT_TRUE(xt::allclose(x / y, z, 0.01, 0.001)) << (x / y) << std::endl
                                                     << z;
  }
  {
    xt::random::seed(0);
    xt::xarray<float> x = xt::random::rand<float>({200, 1}, 1, 128);
    xt::xarray<float> y = xt::random::rand<float>({200, 1}, 256, 4096);
    Value a = test::makeValue(&ctx, x, VIS_SECRET);
    Value b = test::makeValue(&ctx, y, VIS_SECRET);
    Value c = f_div(&ctx, a, b);
    EXPECT_EQ(c.dtype(), DT_F32);

    auto z = dump_public_as<float>(&ctx, reveal(&ctx, c));
    auto e = x / y;
    auto r = xt::abs(z - e) / e * 100;
    auto mm = xt::minmax(r)();
    SPDLOG_INFO("err radio [{} , {}]", mm[0], mm[1]);
  }
}

TEST(FxpTest, Abs) {
  // GIVEN
  SPUContext ctx = test::makeSPUContext();

  xt::xarray<float> x = {{0.5, -2.0}, {0.9, -1.8}};

  // public abs
  {
    Value a = constant(&ctx, x, DT_F32);
    Value c = f_abs(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_F32);

    auto y = dump_public_as<float>(&ctx, c);
    EXPECT_TRUE(xt::allclose(xt::abs(x), y, 0.01, 0.05))
        << xt::abs(x) << std::endl
        << y;
  }

  // secret abs
  {
    Value a = test::makeValue(&ctx, x, VIS_SECRET);
    Value c = f_abs(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_F32);

    auto y = dump_public_as<float>(&ctx, reveal(&ctx, c));
    // low precision
    EXPECT_TRUE(xt::allclose(xt::abs(x), y, 0.1, 0.5))
        << xt::abs(x) << std::endl
        << y;
  }
}

TEST(FxpTest, Floor) {
  // GIVEN
  SPUContext ctx = test::makeSPUContext();

  xt::xarray<float> x = {{0.5, -0.5}, {-20.0, 31.8}, {0, 5.0}, {-5.0, -31.8}};

  // public floor
  {
    Value a = constant(&ctx, x, DT_F32);
    Value c = f_floor(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_F32);

    auto y = dump_public_as<float>(&ctx, c);
    EXPECT_TRUE(xt::allclose(xt::floor(x), y, 0.01, 0.001))
        << xt::floor(x) << std::endl
        << y;
  }

  // secret floor
  {
    Value a = test::makeValue(&ctx, x, VIS_SECRET);
    Value c = f_floor(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_F32);

    auto y = dump_public_as<float>(&ctx, reveal(&ctx, c));
    // low precision
    EXPECT_TRUE(xt::allclose(xt::floor(x), y, 0.01, 0.001))
        << xt::floor(x) << std::endl
        << y;
  }
}

TEST(FxpTest, Ceil) {
  // GIVEN
  SPUContext ctx = test::makeSPUContext();

  xt::xarray<float> x = {{0.5, -0.5}, {-20.0, 31.8}, {0, 5.0}, {-5.0, -31.8}};

  // public ceil
  {
    Value a = constant(&ctx, x, DT_F32);
    Value c = f_ceil(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_F32);

    auto y = dump_public_as<float>(&ctx, c);
    EXPECT_TRUE(xt::allclose(xt::ceil(x), y, 0.01, 0.001))
        << xt::ceil(x) << std::endl
        << y;
  }

  // secret ceil
  {
    Value a = test::makeValue(&ctx, x, VIS_SECRET);
    Value c = f_ceil(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_F32);

    auto y = dump_public_as<float>(&ctx, reveal(&ctx, c));
    // low precision
    EXPECT_TRUE(xt::allclose(xt::ceil(x), y, 0.01, 0.001))
        << xt::ceil(x) << std::endl
        << y;
  }
}

}  // namespace spu::kernel::hal
