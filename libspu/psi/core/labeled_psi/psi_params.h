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

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "apsi/psi_params.h"
#include "yacl/link/link.h"

#include "libspu/core/prelude.h"

#include "libspu/psi/core/labeled_psi/serializable.pb.h"

namespace spu::psi {

struct SEALParams {
  size_t poly_modulus_degree;
  // plain_modulus or plain_modulus_bits
  size_t plain_modulus = 0;
  size_t plain_modulus_bits = 0;
  std::vector<int> coeff_modulus_bits;

  size_t GetPlainModulusBits() {
    // plain_modulus_bits
    if (plain_modulus_bits > 0) {
      return plain_modulus_bits;
    } else if (plain_modulus > 0) {
      // get plain_modulus_bits by plain_modulus
      return std::floor(std::log2(plain_modulus));
    } else {
      SPU_THROW(
          "SEALParams error, must set plain_modulus or plain_modulus_bits");
    }
  }
};

/**
 * @brief Get the Psi Params object
 *
 * @param nr receiver's items size
 * @param ns sender's items size
 * @return apsi::PSIParams
 */
apsi::PSIParams GetPsiParams(size_t nr, size_t ns);

/**
 * @brief Serialize apsi::PSIParams to yacl::Buffer
 *
 * @param psi_params  apsi::PSIParams
 * @return yacl::Buffer
 */
yacl::Buffer PsiParamsToBuffer(const apsi::PSIParams &psi_params);

/**
 * @brief DeSerialize yacl::Buffer to apsi::PSIParams
 *
 * @param buffer  PSIParams bytes buffer
 * @return apsi::PSIParams
 */
apsi::PSIParams ParsePsiParamsProto(const yacl::Buffer &buffer);
apsi::PSIParams ParsePsiParamsProto(
    const proto::LabelPsiParamsProto &psi_params_proto);

}  // namespace spu::psi
