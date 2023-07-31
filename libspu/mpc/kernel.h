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

#include "libspu/core/context.h"
#include "libspu/core/prelude.h"

namespace spu::mpc {

class RandKernel : public Kernel {
 public:
  void evaluate(KernelEvalContext* ctx) const override;
  virtual NdArrayRef proc(KernelEvalContext* ctx, const Shape& shape) const = 0;
};

class UnaryKernel : public Kernel {
 public:
  void evaluate(KernelEvalContext* ctx) const override;
  virtual NdArrayRef proc(KernelEvalContext* ctx,
                          const NdArrayRef& in) const = 0;
};

class RevealToKernel : public Kernel {
 public:
  void evaluate(KernelEvalContext* ctx) const override;
  virtual NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                          size_t rank) const = 0;
};

class ShiftKernel : public Kernel {
 public:
  void evaluate(KernelEvalContext* ctx) const override;
  virtual NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                          size_t bits) const = 0;
};

class BinaryKernel : public Kernel {
 public:
  void evaluate(KernelEvalContext* ctx) const override;
  virtual NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                          const NdArrayRef& rhs) const = 0;
};

class MatmulKernel : public Kernel {
 public:
  void evaluate(KernelEvalContext* ctx) const override;
  virtual NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& a,
                          const NdArrayRef& b) const = 0;
};

class Conv2DKernel : public Kernel {
 public:
  void evaluate(KernelEvalContext* ctx) const override;

  // tensor: NxHxWxC
  // filter: hxwxCxO
  virtual NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& tensor,
                          const NdArrayRef& filter, int64_t stride_h,
                          int64_t stride_w) const = 0;
};

class BitrevKernel : public Kernel {
 public:
  void evaluate(KernelEvalContext* ctx) const override;

  virtual NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                          size_t start, size_t end) const = 0;
};

enum class TruncLsbRounding {
  // For protocols like SecureML/ABY3, the LSB is random.
  Random,
  // For DEK19/EGK+20, the LSB is probabilistic, More precisely, for
  //    y = x/2` + u, where u ∈ [0, 1).
  // The result has probability of u to be x/2`+1, and probability 1-u to be
  // x/2`.
  Probabilistic,
  // For some deterministic truncation, the LSB is deterministic.
  Nearest,
};

class TruncAKernel : public Kernel {
 public:
  void evaluate(KernelEvalContext* ctx) const override;

  // For protocol like SecureML, the most significant bit may have error with
  // low probability, which lead to huge calculation error.
  //
  // Return true if the protocol has this kind of error.
  virtual bool hasMsbError() const = 0;

  virtual TruncLsbRounding lsbRounding() const = 0;

  virtual NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                          size_t bits, SignType sign) const = 0;
};

class BitSplitKernel : public Kernel {
 public:
  void evaluate(KernelEvalContext* ctx) const override;
  virtual NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                          size_t stride) const = 0;
};

class CastTypeKernel : public Kernel {
  void evaluate(KernelEvalContext* ctx) const override;

  virtual NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                          const Type& to_type) const = 0;
};

}  // namespace spu::mpc
