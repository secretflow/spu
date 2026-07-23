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

#include "libspu/mpc/kernel.h"

namespace spu::mpc::semi2k {

// Given [x*2^fxp] mod 2k for x
// compute [exp(x) * 2^fxp] mod 2^k
// Example:
// spu::mpc::semi2k::ExpA exp;
// outp = exp.proc(&kcontext, ring2k_shr);
class ExpA : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "exp_a"; }

  ce::CExpr latency() const override { return ce::Const(2); }

  ce::CExpr comm() const override { return 2 * ce::K(); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};

}  // namespace spu::mpc::semi2k
