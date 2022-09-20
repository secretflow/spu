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
#include "emp-zk/emp-vole/constants.h"
#include "emp-zk/emp-vole/emp-vole.h"
#include "yasl/base/exception.h"
#include "yasl/link/link.h"

#include "spu/crypto/ot/silent/cheetah_io_channel.h"
#include "spu/psi/core/communication.h"
#include "spu/psi/utils/serialize.h"

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
                const std::shared_ptr<yasl::link::Context> &link_ctx);

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
      YASL_THROW("party: {} without delta", party_);
    }
  }

 private:
  // setup
  // alice set delta
  // call baseVole
  void Setup();

  int party_;
  std::shared_ptr<yasl::link::Context> link_ctx_;

  WolverineVoleFieldType delta_;

  std::array<std::unique_ptr<CheetahIo>, kVoleSilentOTThreads> silent_ios_;
  CheetahIo *ios_[kVoleSilentOTThreads];
  std::unique_ptr<VoleTriple<CheetahIo>> emp_zk_vole_;
};

std::vector<WolverineVoleFieldType> GetPolynoimalCoefficients(
    const std::vector<std::string> &bin_data);

WolverineVoleFieldType EvaluatePolynoimal(
    absl::Span<const WolverineVoleFieldType> coeffs, std::string_view x,
    WolverineVoleFieldType high_coeff = 1);

WolverineVoleFieldType EvaluatePolynoimal(
    absl::Span<const WolverineVoleFieldType> coeffs, WolverineVoleFieldType x,
    WolverineVoleFieldType high_coeff);

}  // namespace spu::psi
