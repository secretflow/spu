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

#include <numeric>
#include <vector>

#include "xtensor/xsort.hpp"

#include "libspu/core/prelude.h"

namespace squirrel {
#define ENABLE_MISSING_VALUE 0

template <typename T>
inline T CeilDiv(T a, T b) {
  SPU_ENFORCE(b > 0);
  return (a + b - 1) / b;
}

static inline bool math_eq(double x, double y) {
  constexpr double eps = 1e-10;
  return std::abs(x - y) < eps;
}

static inline bool math_neq(double x, double y) { return !math_eq(x, y); }

double MissingValueSymbol() { return std::numeric_limits<double>::infinity(); }

bool IsMissing(double x) { return x == MissingValueSymbol(); }

Binning::Binning(size_t nbuckets) : nbuckets_(nbuckets) {
  SPU_ENFORCE(nbuckets_ >= 3);
}

// Discretize variable into equal-sized buckets based on rank or based on
// sample quantiles.
// REF `secretflow/ml/boost/ss_xgb_v/core/tree_worker.py#_qcut
void Binning::Fit(const absl::Span<const double> x) {
  SPU_ENFORCE(not x.empty(), "can not quantize empty input");

  const size_t nsamples = x.size();
  const size_t max_bins = nbuckets_;
  std::vector<double> sorted_x(x.data(), x.data() + nsamples);
  std::sort(sorted_x.begin(), sorted_x.end());

  std::vector<double> bin_thresholds;
  std::vector<double> value_category;
  size_t remain_samples = nsamples;
  size_t expected_idx = CeilDiv(remain_samples, max_bins);
  bool fast_skip = false;
  double last_value = std::numeric_limits<double>::max();
  size_t idx = 0;
  while (idx < nsamples) {
    double v = sorted_x[idx];
    bool is_diff = math_neq(v, last_value);

    if (!fast_skip && is_diff) {
      if (value_category.size() <= max_bins) {
        value_category.push_back(v);
      } else {
        fast_skip = true;
      }
      last_value = v;
    }

    if (idx >= expected_idx && is_diff) {
      bin_thresholds.push_back(v);
      size_t nn = bin_thresholds.size();
      if (nn + 1 == max_bins) {
        break;
      }
      remain_samples = nsamples - idx;
      size_t expected_bin_count = CeilDiv(remain_samples, max_bins - nn);
      expected_idx = idx + expected_bin_count;
      last_value = v;
    }

    if (!fast_skip || idx >= expected_idx) {
      idx += 1;
    } else {
      idx = expected_idx;
    }
  }

  if (value_category.size() <= max_bins) {
    // Use category as split point.
    // NOTE: bin_thresholds = value_category[1:]
    bin_thresholds =
        std::vector<double>(value_category.begin() + 1, value_category.end());
  }

  // add `infty` for test samples that might larger than the train samples
  bin_thresholds.push_back(std::numeric_limits<double>::infinity());

  SPU_ENFORCE(bin_thresholds.size() <= nbuckets_);

  std::swap(bin_thresholds_, bin_thresholds);
}

std::vector<uint16_t> Binning::Transform(absl::Span<const double> x) {
  size_t n = x.size();
  std::vector<uint16_t> ret(n);
  for (size_t i = 0; i < n; ++i) {
    auto lower =
        std::lower_bound(bin_thresholds_.begin(), bin_thresholds_.end(), x[i]);
    // x(i) <= *lower
    auto bin = std::distance(bin_thresholds_.begin(), lower);
    ret[i] = static_cast<uint16_t>(bin);
  }
  return ret;
}

}  // namespace squirrel
