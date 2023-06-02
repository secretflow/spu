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
#include "libspu/mpc/semi2k/arithmetic.h"

namespace spu::mpc::cheetah {

using RandA = spu::mpc::semi2k::RandA;

using P2A = spu::mpc::semi2k::P2A;

using A2P = spu::mpc::semi2k::A2P;

using V2A = spu::mpc::semi2k::V2A;

using A2V = spu::mpc::semi2k::A2V;

using NotA = spu::mpc::semi2k::NotA;

using AddAP = spu::mpc::semi2k::AddAP;

using AddAA = spu::mpc::semi2k::AddAA;

using MulAP = spu::mpc::semi2k::MulAP;

using MatMulAP = spu::mpc::semi2k::MatMulAP;

using LShiftA = spu::mpc::semi2k::LShiftA;

class TruncAWithSign : public TruncAWithSignKernel {
 public:
  static constexpr char kBindName[] = "trunc_a_with_sign";

  Kind kind() const override { return Kind::Dynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& x, size_t bits,
                bool is_positive) const override;

  bool hasMsbError() const override { return false; }

  TruncLsbRounding lsbRounding() const override {
    return TruncLsbRounding::Probabilistic;
  }
};

class TruncA : public TruncAKernel {
 public:
  static constexpr char kBindName[] = "trunc_a";

  Kind kind() const override { return Kind::Dynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& x,
                size_t bits) const override;

  bool hasMsbError() const override { return false; }

  TruncLsbRounding lsbRounding() const override {
    return TruncLsbRounding::Probabilistic;
  }
};

class MsbA2B : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "msb_a2b";

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& x) const override;
};

class EqualAA : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "equal_aa";

  Kind kind() const override { return Kind::Dynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& x,
                const ArrayRef& y) const override;
};

class EqualAP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "equal_ap";

  Kind kind() const override { return Kind::Dynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& x,
                const ArrayRef& y) const override;
};

class MulA1B : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "mul_a1b";

  Kind kind() const override { return Kind::Dynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& ashr,
                const ArrayRef& bshr) const override;
};

class MulAA : public BinaryKernel {
 private:
  ArrayRef mulDirectly(KernelEvalContext* ctx, const ArrayRef& lhs,
                       const ArrayRef& rhs) const;

  ArrayRef mulWithBeaver(KernelEvalContext* ctx, const ArrayRef& lhs,
                         const ArrayRef& rhs) const;

 public:
  static constexpr char kBindName[] = "mul_aa";

  Kind kind() const override { return Kind::Dynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& x,
                const ArrayRef& y) const override;
};

class MatMulAA : public MatmulKernel {
 public:
  static constexpr char kBindName[] = "mmul_aa";

  Kind kind() const override { return Kind::Dynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& x, const ArrayRef& y,
                size_t m, size_t n, size_t k) const override;
};

class Conv2DAA : public Conv2DKernel {
 public:
  static constexpr char kBindName[] = "conv2d_aa";

  Kind kind() const override { return Kind::Dynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& tensor,
                const ArrayRef& filter, size_t N, size_t H, size_t W, size_t C,
                size_t O, size_t h, size_t w, size_t stride_h,
                size_t stride_w) const override;
};

}  // namespace spu::mpc::cheetah
