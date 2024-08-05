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
#include <utility>
#include <vector>

#include "spdlog/spdlog.h"
#include "yacl/link/link.h"

#include "libspu/psi/core/ecdh_psi.h"

namespace spu::psi {

// bernoulli distribution probability for sub/up samples
struct DpPsiOptions {
  explicit DpPsiOptions(double bob_p = 0.9, double epsilon = 3.0)
      : p1(bob_p), alice_epsilon(epsilon) {
    double e_epsilon = std::exp(alice_epsilon);
    p2 = e_epsilon / (1 + e_epsilon);
    q = 1 - p2;
    SPDLOG_INFO("DpPsiOptions p1:{} epsilon:{} p2:{}, q:{}", p1, epsilon, p2,
                q);
  }

  // bob SubSampling
  double p1;

  //
  double alice_epsilon;

  // alice SubSampling
  double p2;
  // alice UpSampling
  double q;
};

/**
 * @brief
 *
 * @param dp_psi_options: dp psi options
 * @param link_ctx : link for send/recv
 * @param items : data
 * @param sub_sample_size : alice subsample size
 * @param up_sample_size : fake items size in intersection
 * @param curve : ecc curve type, default 25519
 * @return size_t : return intersection size
 */
size_t RunDpEcdhPsiAlice(const DpPsiOptions& dp_psi_options,
                         const std::shared_ptr<yacl::link::Context>& link_ctx,
                         const std::vector<std::string>& items,
                         size_t* sub_sample_size, size_t* up_sample_size,
                         CurveType curve = CurveType::CURVE_25519);

/**
 * @brief
 *
 * @param dp_psi_options : dp psi options
 * @param link_ctx : link for send/recv
 * @param items : data
 * @param sub_sample_size : bob subsample size
 * @param curve : ecc curve type, default 25519
 * @return std::vector<size_t> : return intersection idx
 */
std::vector<size_t> RunDpEcdhPsiBob(
    const DpPsiOptions& dp_psi_options,
    const std::shared_ptr<yacl::link::Context>& link_ctx,
    const std::vector<std::string>& items, size_t* sub_sample_size,
    CurveType curve = CurveType::CURVE_25519);

}  // namespace spu::psi
