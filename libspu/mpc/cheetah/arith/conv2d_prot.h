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
//
#pragma once

#include "libspu/core/ndarray_ref.h"
#include "libspu/mpc/cheetah/arith/common.h"
#include "libspu/mpc/cheetah/rlwe/types.h"

namespace spu::mpc::cheetah {

class TensorEncoder;
class ModulusSwitchHelper;

bool IsSameInputShape(const ArrayRef& base, const Shape3D& shape);

class Conv2DProtocol {
 public:
  struct Meta {
    int64_t input_batch = 1;  // N
    int64_t num_kernels;      // O
    Shape3D input_shape;      // HxWxC
    Shape3D kernel_shape;     // hxwxI
    Shape2D window_strides;
  };

  explicit Conv2DProtocol(const seal::SEALContext& context,
                          const ModulusSwitchHelper& ms_helper);

  size_t GetKernelSize(const Meta& meta) const {
    return GetKernelSize(meta, GetSubTensorShape(meta));
  }

  size_t GetInputSize(const Meta& meta) const {
    return GetInputSize(meta, GetSubTensorShape(meta));
  }

  size_t GetOutSize(const Meta& meta) const {
    return GetOutSize(meta, GetSubTensorShape(meta));
  }

  size_t GetKernelSize(const Meta& meta, const Shape3D& subshape) const;

  size_t GetInputSize(const Meta& meta, const Shape3D& subshape) const;

  size_t GetOutSize(const Meta& meta, const Shape3D& subshape) const;

  Shape3D GetSubTensorShape(const Conv2DProtocol::Meta& meta) const;

  void EncodeInput(const ArrayRef& input, const Meta& meta, bool need_encrypt,
                   absl::Span<RLWEPt> out) const;

  void EncodeKernels(const ArrayRef& kernels, const Meta& meta,
                     bool need_encrypt, absl::Span<RLWEPt> out) const;

  void ExtractLWEsInplace(const Meta& meta, absl::Span<RLWECt> rlwe) const;

  bool IsValidMeta(const Meta& meta) const;

  ArrayRef ParseResult(FieldType field, const Meta& meta,
                       absl::Span<const RLWEPt> rlwe) const;

  ArrayRef ParseResult(FieldType field, const Meta& meta,
                       absl::Span<const RLWEPt> rlwe,
                       const ModulusSwitchHelper& ms) const;

  void Compute(absl::Span<const RLWEPt> tensor, absl::Span<const RLWEPt> kernel,
               const Meta& meta, absl::Span<RLWEPt> out) const;

  void Compute(absl::Span<const RLWECt> tensor, absl::Span<const RLWEPt> kernel,
               const Meta& meta, absl::Span<RLWECt> out) const;

  void Compute(absl::Span<const RLWEPt> tensor, absl::Span<const RLWECt> kernel,
               const Meta& meta, absl::Span<RLWECt> out) const;

 private:
  // accum += x * y
  template <class O, class I, class K>
  void FusedMulAddInplace(O& accum, const I& x, const K& y) const;

  bool IsValidSubShape(const Shape3D& shape) const;

  void EncodeSingleKernel(const ArrayRef& kernel, const Meta& meta,
                          bool scaleup, absl::Span<RLWEPt> out) const;

  // work horse
  template <typename I, typename K, typename O>
  void ComputeOneKernel(absl::Span<const I> input, absl::Span<const K> kernel,
                        absl::Span<O> out) const;

 private:
  int64_t poly_deg_{0};

  seal::SEALContext context_;
  std::shared_ptr<TensorEncoder> tencoder_{nullptr};
};

}  // namespace spu::mpc::cheetah
