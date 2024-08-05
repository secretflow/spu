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

#include "libspu/psi/core/bc22_psi/emp_vole.h"

#include <utility>

#include "spdlog/spdlog.h"
#include "yacl/crypto/utils/rand.h"

namespace spu::psi {

WolverineVole::WolverineVole(PsiRoleType psi_role,
                             std::shared_ptr<yacl::link::Context> link_ctx)
    : party_((psi_role == PsiRoleType::Sender) ? emp::ALICE : emp::BOB),
      link_ctx_(std::move(link_ctx)) {
  // set EmpIoAdapter
  for (size_t i = 0; i < kVoleSilentOTThreads; ++i) {
    silent_ios_[i] = std::make_unique<EmpIoAdapter>(link_ctx_);
    ios_[i] = silent_ios_[i].get();
  }

  emp_zk_vole_ = std::make_unique<VoleTriple<EmpIoAdapter>>(
      party_, kVoleSilentOTThreads, ios_);

  SPDLOG_INFO("party {}, begin svole setup",
              (party_ == emp::ALICE) ? "alice" : "bob");
  Setup();
  SPDLOG_INFO("party {}, after svole setup",
              (party_ == emp::ALICE) ? "alice" : "bob");
}

void WolverineVole::Setup() {
  if (party_ == emp::ALICE) {
    delta_ = yacl::crypto::RandSeed();
    delta_ =
        delta_ & (static_cast<WolverineVoleFieldType>(0xFFFFFFFFFFFFFFFFLL));
    delta_ = mod(delta_, pr);
    emp_zk_vole_->setup(delta_);
  } else {
    emp_zk_vole_->setup();
  }
}

std::vector<WolverineVoleFieldType> WolverineVole::Extend(size_t vole_num) {
  std::vector<WolverineVoleFieldType> vole_blocks(vole_num);

  emp_zk_vole_->extend(vole_blocks.data(), vole_blocks.size());

  return vole_blocks;
}

std::vector<WolverineVoleFieldType> GetPolynomialCoefficients(
    const std::vector<std::string>& bin_data) {
  SPU_ENFORCE(bin_data.size() <= 3);

  std::vector<WolverineVoleFieldType> block_coeffs(bin_data.size());

  std::vector<WolverineVoleFieldType> t(bin_data.size());
  for (size_t i = 0; i < bin_data.size(); ++i) {
    SPU_ENFORCE(bin_data[0].length() <= sizeof(WolverineVoleFieldType), "{}>{}",
                bin_data[0].length(), sizeof(WolverineVoleFieldType));
    t[i] = 0;
    std::memcpy(&t[i], bin_data[i].data(), bin_data[i].length());
    t[i] = mod(t[i], pr);
  }

  if (bin_data.size() == 1) {
    block_coeffs[0] = pr - mod(t[0], pr);
  } else if (bin_data.size() == 2) {
    block_coeffs[0] = mult_mod(t[0], t[1]);
    block_coeffs[1] = pr - mod(t[0] + t[1], pr);
  } else if (bin_data.size() == 3) {
    WolverineVoleFieldType d01;
    WolverineVoleFieldType d02;
    WolverineVoleFieldType d12;

    d01 = mult_mod(t[0], t[1]);
    d02 = mult_mod(t[0], t[2]);
    d12 = mult_mod(t[1], t[2]);

    WolverineVoleFieldType tmp = mult_mod(d01, t[2]);
    while (tmp > pr) {
      tmp -= pr;
    }

    block_coeffs[0] = pr - tmp;

    block_coeffs[1] = mod(d01 + d02 + d12, pr);
    block_coeffs[2] = pr - mod(t[0] + t[1] + t[2], pr);
  }

  return block_coeffs;
}

WolverineVoleFieldType EvaluatePolynomial(
    absl::Span<const WolverineVoleFieldType> coeffs, WolverineVoleFieldType x,
    WolverineVoleFieldType high_coeff) {
  std::vector<WolverineVoleFieldType> xp(coeffs.size() + 1);

  std::vector<WolverineVoleFieldType> block_coeffs;
  for (unsigned __int128 coeff : coeffs) {
    block_coeffs.push_back(static_cast<WolverineVoleFieldType>(coeff));
  }

  WolverineVoleFieldType xx = mod(x, pr);

  xp[0] = 1;
  block_coeffs.push_back(high_coeff);

  for (size_t i = 1; i < xp.size(); ++i) {
    xp[i] = mult_mod(xp[i - 1], xx);
  }

  WolverineVoleFieldType result = 0;
  for (size_t i = 0; i < xp.size(); ++i) {
    result += mult_mod(xp[i], block_coeffs[i]);
    result = mod(result, pr);
  }

  return result;
}

WolverineVoleFieldType EvaluatePolynomial(
    absl::Span<const WolverineVoleFieldType> coeffs, std::string_view x,
    WolverineVoleFieldType high_coeff) {
  WolverineVoleFieldType block_x = 0;

  SPU_ENFORCE(x.length() <= sizeof(WolverineVoleFieldType));
  std::memcpy(&block_x, x.data(), x.length());

  return EvaluatePolynomial(coeffs, block_x, high_coeff);
}

}  // namespace spu::psi
