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

#include <utility>

#include "absl/types/span.h"

#include "libspu/mpc/common/prg_tensor.h"

namespace spu::mpc::semi2k {

class TrustedParty {
 public:
  using Seeds = absl::Span<const PrgSeed>;
  struct Operand {
    PrgArrayDesc desc;
    Seeds seeds;
    bool transpose{false};
  };

  static MemRef adjustMul(absl::Span<Operand>);

  static MemRef adjustSquare(absl::Span<Operand>);

  static MemRef adjustDot(absl::Span<Operand>);

  static MemRef adjustAnd(absl::Span<Operand>);

  static MemRef adjustTrunc(absl::Span<Operand>, size_t bits);

  static std::pair<MemRef, MemRef> adjustTruncPr(absl::Span<Operand>,
                                                 size_t bits);

  static MemRef adjustRandBit(absl::Span<Operand>);

  static MemRef adjustEqz(absl::Span<Operand>);

  static MemRef adjustPerm(absl::Span<Operand>,
                           absl::Span<const int64_t> perm_vec);
};

}  // namespace spu::mpc::semi2k
