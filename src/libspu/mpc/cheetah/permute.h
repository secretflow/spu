// Copyright 2024 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "libspu/mpc/kernel.h"

namespace spu::mpc::cheetah {

/// Some notes about the permutation based shuffle:
///  1. The implementation just requires `mul_aa` (or `mul_av`, `mul_a1b`) to do
///  the secret-swap.
///  2. In general, this implementation is almost protocol-agnostic and is easy
///  to be applied on other MPC protocols.
///    - For n-pc non-replica share, just call `SecureInvPerm` n times.
///    - For n-pc replica share (like ABY3), just call `SecureInvPerm` and
///    `ReSharing` n times.
///  3. The reason which we still choose to implement this in MPC layer is that,
///  to support Perm or InvPerm (and Secret Radix Sort) under the "Additive"
///  semantic of perm (refer to hlo/permute.h for more details), SPU currently
///  implements them relying on some perm-related kernels in MPC layer.
class RandPermM : public RandKernel {
 public:
  static constexpr const char* kBindName() { return "rand_perm_m"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const Shape& shape,
                  FieldType perm_field = FieldType::FM64) const override;
};

// Note: you should regard it conceptually just a SHUFFLE kernel, which
// rearrange the input with an unknown permutation.
class PermAM : public PermKernel {
 public:
  static constexpr const char* kBindName() { return "perm_am"; };

  // more precisely, if using waksman net to do permutation:
  // the latency = 2 * (2 * log(n) - 1) * (latency of mul_av)
  // the comm = 2 * ( \sum_1^n ceil(log(i)) ) * (comm of mul_av)
  //
  // Note: if `mul_av` is not supported, then `mul_aa` will be used.
  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  const NdArrayRef& perm) const override;
};

class PermAP : public PermKernel {
 public:
  static constexpr const char* kBindName() { return "perm_ap"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  const NdArrayRef& perm) const override;
};

// Note: you should regard it conceptually just a UNSHUFFLE kernel, which
// rearrange the inputs permuted before to the origin order.
class InvPermAM : public PermKernel {
 public:
  static constexpr const char* kBindName() { return "inv_perm_am"; }

  // more precisely, if using waksman net to do permutation:
  // the latency = 2 * (2 * log(n) - 1) * (latency of mul_av)
  // the comm = 2 * ( \sum_1^n ceil(log(i)) ) * (comm of mul_av)
  //
  // Note: if `mul_av` is not supported, then `mul_aa` will be used.
  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  const NdArrayRef& perm) const override;
};

class InvPermAP : public PermKernel {
 public:
  static constexpr const char* kBindName() { return "inv_perm_ap"; }

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  const NdArrayRef& perm) const override;
};

class InvPermAV : public PermKernel {
 public:
  static constexpr const char* kBindName() { return "inv_perm_av"; }

  // more precisely, if using waksman net to do permutation:
  // the latency = 1 * (2 * log(n) - 1) * (latency of mul_av)
  // the comm = 1 * ( \sum_1^n ceil(log(i)) ) * (comm of mul_av)
  //
  // Note: if `mul_av` is not supported, then `mul_aa` will be used.
  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  const NdArrayRef& perm) const override;
};

}  // namespace spu::mpc::cheetah
