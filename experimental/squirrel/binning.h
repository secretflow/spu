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

#pragma once

#include "absl/types/span.h"
#include "xtensor/xarray.hpp"

namespace squirrel {

double MissingValueSymbol();

bool IsMissing(double x);

// Discretize variable into equal-sized buckets based on rank or based on
// sample quantiles.
class Binning {
 public:
  // nbuckets >= 3. At least 2 distinct bins + 1 bin for infinity
  explicit Binning(size_t nbuckets);

  const std::vector<double>& bin_thresholds() const { return bin_thresholds_; }

  inline size_t nbuckets() const { return nbuckets_; }

  // Discretize variable into equal-sized buckets based on rank or based on
  // sample quantiles.
  // A given value x will be mapped into bin[i] iff
  // bin_thresholds[i - 1] < x <= bin_thresholds[i]
  //
  // The last bin indicates infty: bin_thresholds[-1] = infty
  void Fit(absl::Span<const double> x);

  // Missing values will be mapped to the last bin.
  std::vector<uint16_t> Transform(absl::Span<const double> x);

 private:
  size_t nbuckets_{0};
  std::vector<double> bin_thresholds_;
};

}  // namespace squirrel
