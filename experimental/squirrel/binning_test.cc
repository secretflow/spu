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

#include "experimental/squirrel/binning.h"

#include <random>

#include "gtest/gtest.h"

namespace squirrel::test {
class BinningTest : public ::testing::Test {};

TEST_F(BinningTest, Numeric) {
  std::default_random_engine rdv;
  std::uniform_real_distribution<double> uniform(-100., 100.);
  size_t nsample = 100;
  size_t nbuckets = 20;
  std::vector<size_t> shape{nsample};
  std::vector<size_t> y_shape{nsample, 1};

  xt::xarray<double> x(shape);
  xt::xarray<double> y(y_shape);
  std::generate_n(x.data(), x.size(), [&]() { return uniform(rdv); });
  std::generate_n(y.data(), y.size(), [&]() { return uniform(rdv); });

  Binning dp(nbuckets);
  dp.Fit({x.data(), x.size()});
  auto& thresholds = dp.bin_thresholds();
  auto mapping = dp.Transform(x);
  // A given value x will be mapped into bin[i] iff
  // bining_thresholds[i - 1] < x <= binning_thresholds[i]
  for (size_t i = 0; i < nsample; ++i) {
    int bin = mapping[i];
    EXPECT_LE(x(i), thresholds[bin]);
    if (bin > 0) {
      EXPECT_GT(x(i), thresholds[bin - 1]);
    }
  }

  mapping = dp.Transform(y);
  // A given value x will be mapped into bin[i] iff
  // bining_thresholds[i - 1] < x <= binning_thresholds[i]
  for (size_t i = 0; i < nsample; ++i) {
    int bin = mapping[i];
    EXPECT_LE(y(i, 0), thresholds[bin]);
    if (bin > 0) {
      EXPECT_GT(y(i, 0), thresholds[bin - 1]);
    }
  }
}

TEST_F(BinningTest, Category) {
  std::default_random_engine rdv;
  std::uniform_int_distribution<size_t> uniform(0, 3);
  size_t nsample = 10;
  size_t nbuckets = 5;
  std::vector<size_t> x_shape{nsample, 1};
  std::vector<size_t> y_shape{1, nsample};
  xt::xarray<double> x(x_shape);
  xt::xarray<double> y(y_shape);
  std::generate_n(x.data(), x.size(), [&]() { return (double)uniform(rdv); });
  std::generate_n(y.data(), y.size(), [&]() { return (double)uniform(rdv); });

  Binning dp(nbuckets);
  dp.Fit(x);
  auto& thresholds = dp.bin_thresholds();
  auto mapping = dp.Transform(x);
  for (size_t i = 0; i < nsample; ++i) {
    size_t bin = mapping[i];
    EXPECT_LE(x(i, 0), thresholds[bin]);
    if (bin > 0) {
      EXPECT_GT(x(i, 0), thresholds[bin - 1]);
    }
  }

  mapping = dp.Transform(y);
  for (size_t i = 0; i < nsample; ++i) {
    size_t bin = mapping[i];
    EXPECT_LE(y(0, i), thresholds[bin]);
    if (bin > 0) {
      EXPECT_GT(y(0, i), thresholds[bin - 1]);
    }
  }
}

TEST_F(BinningTest, LargerValue) {
  std::default_random_engine rdv;
  std::uniform_real_distribution<double> uniform(-100., 100.);
  size_t nsample = 100;
  size_t nbuckets = 20;
  std::vector<size_t> shape{nsample};
  std::vector<size_t> y_shape{nsample, 1};

  xt::xarray<double> x(shape);
  xt::xarray<double> y(y_shape);
  std::generate_n(x.data(), x.size(), [&]() { return uniform(rdv); });
  std::generate_n(y.data(), y.size(), [&]() { return uniform(rdv); });
  y[0] = 1000.;
  y[3] = 200.;

  Binning dp(nbuckets);
  dp.Fit(x);
  auto& thresholds = dp.bin_thresholds();
  auto mapping = dp.Transform(y);
  // A given value x will be mapped into bin[i] iff
  // bining_thresholds[i - 1] < x <= binning_thresholds[i]
  for (size_t i = 0; i < nsample; ++i) {
    int bin = mapping[i];
    EXPECT_LE(y(i, 0), thresholds[bin]);
    if (bin > 0) {
      EXPECT_GT(y(i, 0), thresholds[bin - 1]);
    }
  }
}
}  // namespace squirrel::test
