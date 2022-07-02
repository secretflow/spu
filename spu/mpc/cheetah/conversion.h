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
#include "spu/mpc/semi2k/conversion.h"
#include "spu/mpc/util/cexpr.h"

namespace spu::mpc::cheetah {

using util::Const;
using util::K;
using util::Log;
using util::N;

typedef spu::mpc::semi2k::AddBB AddBB;

typedef spu::mpc::semi2k::A2B A2B;

class B2A : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "b2a";

  util::CExpr latency() const override { return Const(0); }

  util::CExpr comm() const override { return Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& x) const override;
};

}  // namespace spu::mpc::cheetah
