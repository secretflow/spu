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

#include <memory>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "emp-tool/utils/block.h"
#include "emp-tool/utils/f2k.h"
#include "emp-zk/emp-vole/emp-vole.h"
#include "yacl/link/link.h"

#include "libspu/core/prelude.h"
#include "libspu/psi/core/communication.h"
#include "libspu/psi/utils/emp_io_adapter.h"
#include "libspu/psi/utils/serialize.h"

namespace spu::psi {

inline constexpr size_t kVoleSilentOTThreads = 1;

// VOLE
// Wolverine: Fast, Scalable, and Communication-Efficient
// Zero-Knowledge Proofs for Boolean and Arithmetic Circuits
// https://eprint.iacr.org/2020/925
// https://github.com/emp-toolkit/emp-zk

using WolverineVoleFieldType = __uint128_t;

class WolverineVole {
 public:
  WolverineVole(PsiRoleType psi_role,
                std::shared_ptr<yacl::link::Context> link_ctx);

  // extend baseVole get vole_num voles
  // Filed: mersenne prime 2^61 - 1
  // wi = delta * ui + vi
  // alice : delta, wi
  // bob : ui || vi as one __uint128_t
  std::vector<WolverineVoleFieldType> Extend(size_t vole_num);

  // get delta
  WolverineVoleFieldType Delta() {
    if (party_ == emp::ALICE) {
      return delta_;
    } else {
      SPU_THROW("party: {} without delta", party_);
    }
  }

 private:
  // setup
  // alice set delta
  // call baseVole
  void Setup();

  int party_;
  std::shared_ptr<yacl::link::Context> link_ctx_;

  WolverineVoleFieldType delta_;

  std::array<std::unique_ptr<EmpIoAdapter>, kVoleSilentOTThreads> silent_ios_;
  EmpIoAdapter *ios_[kVoleSilentOTThreads];
  std::unique_ptr<VoleTriple<EmpIoAdapter>> emp_zk_vole_;
};

std::vector<WolverineVoleFieldType> GetPolynomialCoefficients(
    const std::vector<std::string> &bin_data);

WolverineVoleFieldType EvaluatePolynomial(
    absl::Span<const WolverineVoleFieldType> coeffs, std::string_view x,
    WolverineVoleFieldType high_coeff = 1);

WolverineVoleFieldType EvaluatePolynomial(
    absl::Span<const WolverineVoleFieldType> coeffs, WolverineVoleFieldType x,
    WolverineVoleFieldType high_coeff);

}  // namespace spu::psi
