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

#pragma once

#include <cmath>
#include <memory>
#include <string>

namespace spu::psi {

inline constexpr double kEpsilonPsi = 4;
inline constexpr double kErrorRate = 1.e-12;

double ComputeEpsilon1(size_t n, double epsilon2);

double ComputeEpsilon2(size_t n, double epsilon = kEpsilonPsi);

inline double ComputePSubKeep(double epsilon2) {
  double a = std::exp(epsilon2);
  return a / (1 + a);
}

double CalibrateAnalyticGaussianMechanism(double epsilon, double delta,
                                          double GS, double tol = kErrorRate);
}  // namespace spu::psi
