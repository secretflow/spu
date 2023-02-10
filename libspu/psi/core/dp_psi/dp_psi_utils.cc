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

#include "libspu/psi/core/dp_psi/dp_psi_utils.h"

#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <utility>
#include <vector>

namespace spu::psi {

double ComputeEpsilon2(size_t n, double epsilon) {
  double epsilon1;
  double epsilon2_mid;
  double epsilon2_min;
  double epsilon2_max;
  double err_epsilon;
  epsilon2_min = 2.5;
  epsilon2_max = kEpsilonPsi;
  epsilon2_mid = epsilon2_min + (epsilon2_max - epsilon2_min) / 2;

  do {
    epsilon1 = ComputeEpsilon1(n, epsilon2_mid);

    err_epsilon = epsilon - epsilon1 - epsilon2_mid;

    if (std::abs(err_epsilon) > kErrorRate) {
      if (err_epsilon > 0) {
        epsilon2_min = epsilon2_mid;
      } else {
        epsilon2_max = epsilon2_mid;
      }
      epsilon2_mid = epsilon2_min + (epsilon2_max - epsilon2_min) / 2;
    } else {
      break;
    }

  } while (true);

  return epsilon2_mid;
}

inline double ComputeDelta1(size_t n) {
  double delta1;
  delta1 = (10 * n);
  delta1 = 1.0 / delta1;

  return delta1;
}

double ComputeEpsilon1(size_t n, double epsilon2) {
  double epsilon1;
  double delta1 = ComputeDelta1(n);
  double p_sub_keep = ComputePSubKeep(epsilon2);

  double m =
      n * p_sub_keep - std::sqrt(2 * n * p_sub_keep * std::log(2 / delta1));

  epsilon1 =
      std::sqrt(32 * std::log(4 / delta1) / m) * (1 - (n * p_sub_keep - m) / n);

  return epsilon1;
}

inline double Phi(double t) {
  return 0.5 * (1.0 + std::erf(t / std::sqrt(2.0)));
}

inline double caseA(double epsilon, double s) {
  return Phi(std::sqrt(epsilon * s)) -
         std::exp(epsilon) * Phi(-std::sqrt(epsilon * (s + 2.0)));
}

inline double caseB(double epsilon, double s) {
  return Phi(-std::sqrt(epsilon * s)) -
         std::exp(epsilon) * Phi(-std::sqrt(epsilon * (s + 2.0)));
}

std::pair<double, double> DoublingTrick(
    const std::function<bool(double)>& predicate_stop, double s_inf,
    double s_sup) {
  while (!predicate_stop(s_sup)) {
    s_inf = s_sup;
    s_sup = 2.0 * s_inf;
  }
  return std::make_pair(s_inf, s_sup);
}

double BinarySearch(const std::function<bool(double)>& predicate_stop,
                    const std::function<bool(double)>& predicate_left,
                    double s_inf, double s_sup) {
  double s_mid = s_inf + (s_sup - s_inf) / 2.0;
  while (!predicate_stop(s_mid)) {
    if (predicate_left(s_mid)) {
      s_sup = s_mid;
    } else {
      s_inf = s_mid;
    }
    s_mid = s_inf + (s_sup - s_inf) / 2.0;
  }

  return s_mid;
}

double CalibrateAnalyticGaussianMechanism(double epsilon, double delta,
                                          double gs, double tol) {
  double alpha;
  double delta_thr = caseA(epsilon, 0.0);
  const double EPSINON = 0.00001;

  if (std::abs(delta - delta_thr) < EPSINON) {
    alpha = 1.0;
  } else {
    std::function<bool(double)> predicate_stop_DT;
    std::function<bool(double)> predicate_stop_BS;
    std::function<bool(double)> predicate_left_BS;
    std::function<double(double)> function_s_to_alpha;
    std::function<double(double)> function_s_to_delta;

    if (delta > delta_thr) {
      predicate_stop_DT = [&](double s) {
        return (caseA(epsilon, s) >= delta);
      };
      function_s_to_delta = [&](double s) { return caseA(epsilon, s); };
      predicate_left_BS = [&](double s) { return (caseA(epsilon, s) > delta); };
      function_s_to_alpha = [&](double s) {
        return (std::sqrt(1.0 + s / 2.0) - std::sqrt(s / 2.0));
      };
    } else {
      predicate_stop_DT = [&](double s) {
        return (caseB(epsilon, s) <= delta);
      };
      function_s_to_delta = [&](double s) { return caseB(epsilon, s); };
      predicate_left_BS = [&](double s) { return (caseB(epsilon, s) < delta); };
      function_s_to_alpha = [&](double s) {
        return (std::sqrt(1.0 + s / 2.0) + std::sqrt(s / 2.0));
      };
    }

    predicate_stop_BS = [&](double s) {
      return (std::abs(function_s_to_delta(s) - delta) <= tol);
    };

    auto s_pair = DoublingTrick(predicate_stop_DT, 0.0, 1.0);
    double s_inf = s_pair.first;
    double s_sup = s_pair.second;

    double s_final =
        BinarySearch(predicate_stop_BS, predicate_left_BS, s_inf, s_sup);
    alpha = function_s_to_alpha(s_final);
  }

  double sigma = alpha * gs / std::sqrt(2.0 * epsilon);

  return sigma;
}

}  // namespace spu::psi
