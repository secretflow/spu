// Copyright 2023 Ant Group Co., Ltd.
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

namespace spu::mpc::aby3 {

class RandPermM : public RandKernel {
 public:
  static constexpr char kBindName[] = "rand_perm_m";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const Shape& shape) const override;
};

class PermAM : public PermKernel {
 public:
  static constexpr char kBindName[] = "perm_am";

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  const NdArrayRef& perm) const override;
};

class PermAP : public PermKernel {
 public:
  static constexpr char kBindName[] = "perm_ap";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  const NdArrayRef& perm) const override;
};

class InvPermAM : public PermKernel {
 public:
  static constexpr char kBindName[] = "inv_perm_am";

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  const NdArrayRef& perm) const override;
};

class InvPermAP : public PermKernel {
 public:
  static constexpr char kBindName[] = "inv_perm_ap";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  const NdArrayRef& perm) const override;
};

}  // namespace spu::mpc::aby3
