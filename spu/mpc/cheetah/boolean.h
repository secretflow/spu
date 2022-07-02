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

#include "spu/mpc/kernel.h"
#include "spu/mpc/semi2k/boolean.h"
#include "spu/mpc/util/cexpr.h"

namespace spu::mpc::cheetah {

using util::Const;
using util::K;
using util::Log;
using util::N;

typedef spu::mpc::semi2k::ZeroB ZeroB;

typedef spu::mpc::semi2k::B2P B2P;

typedef spu::mpc::semi2k::P2B P2B;

typedef spu::mpc::semi2k::AndBP AndBP;

class AndBB : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "and_bb";

  util::CExpr latency() const override { return Const(1); }

  util::CExpr comm() const override { return K() * 2 * (N() - 1); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

typedef spu::mpc::semi2k::XorBP XorBP;

typedef spu::mpc::semi2k::XorBB XorBB;

typedef spu::mpc::semi2k::LShiftB LShiftB;

typedef spu::mpc::semi2k::RShiftB RShiftB;

typedef spu::mpc::semi2k::ARShiftB ARShiftB;

typedef spu::mpc::semi2k::BitrevB BitrevB;

}  // namespace spu::mpc::cheetah
