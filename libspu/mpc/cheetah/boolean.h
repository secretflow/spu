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

#include "libspu/mpc/kernel.h"
#include "libspu/mpc/semi2k/boolean.h"

namespace spu::mpc::cheetah {

using CommonTypeB = spu::mpc::semi2k::CommonTypeB;

using CastTypeB = spu::mpc::semi2k::CastTypeB;

using B2P = spu::mpc::semi2k::B2P;

using P2B = spu::mpc::semi2k::P2B;

using AndBP = spu::mpc::semi2k::AndBP;

class AndBB : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "and_bb";

  Kind kind() const override { return Kind::Dynamic; }

  ce::CExpr latency() const override { return ce::Const(1); }

  ce::CExpr comm() const override { return ce::K() * 2 * (ce::N() - 1); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

using XorBP = spu::mpc::semi2k::XorBP;

using XorBB = spu::mpc::semi2k::XorBB;

using LShiftB = spu::mpc::semi2k::LShiftB;

using RShiftB = spu::mpc::semi2k::RShiftB;

using ARShiftB = spu::mpc::semi2k::ARShiftB;

using BitrevB = spu::mpc::semi2k::BitrevB;

}  // namespace spu::mpc::cheetah
