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

#include "libspu/core/array_ref.h"
#include "libspu/core/ndarray_ref.h"
#include "libspu/core/shape_util.h"
#include "libspu/mpc/cheetah/arith/conv2d_prot.h"

namespace spu::mpc::cheetah {

// A slice of a multi-dim tensor with zero padding.
// The slice is defined by `offsets` and `extents`
template <int Dim>
struct SlicedTensor {
 public:
  using Shape = std::array<int64_t, Dim>;

  SlicedTensor(const ArrayRef &base, const Shape &base_shape,
               const Shape &offsets, const Shape &extents);

  static SlicedTensor<Dim> Wrap(const ArrayRef &base, const Shape &base_shape,
                                const Shape &offsets, const Shape &extents);

  SlicedTensor(const SlicedTensor<Dim> &oth) = default;

  SlicedTensor(SlicedTensor<Dim> &&oth) noexcept = default;

  SlicedTensor<Dim> &operator=(const SlicedTensor<Dim> &oth) = delete;

  template <typename T>
  T at(absl::Span<const int64_t> idx) const {
    // sementic check
    SPU_ENFORCE(idx.size() == static_cast<size_t>(Dim));
    for (int i = 0; i < Dim; ++i) {
      SPU_ENFORCE(idx[i] >= 0 && zero_pad_extents_[i]);
    }

    // zero padding
    std::array<int64_t, Dim> index;
    for (int i = 0; i < Dim; ++i) {
      if (idx[i] >= extents_[i]) return static_cast<T>(0);
      index[i] = idx[i] + offsets_[i];
    }
    int64_t offset = calcFlattenOffset(index, base_shape_, flatten_strides_);
    SPU_ENFORCE(offset >= 0 && offset < base_.numel());
    return base_.at<T>(offset);
  }

  Shape shape() const { return zero_pad_extents_; }

  int64_t numel() const { return calcNumel(shape()); }

  FieldType field() const {
    const Type &eltype = base_.eltype();
    return eltype.as<Ring2k>()->field();
  }

  void ZeroPadAs(const Shape &extents) {
    for (size_t d = 0; d < Dim; ++d) {
      SPU_ENFORCE(extents[d] > 0);
    }
    zero_pad_extents_ = extents;
  }

 private:
  const ArrayRef &base_;
  Shape base_shape_;
  Shape offsets_;
  Shape extents_;
  Shape flatten_strides_;

  Shape zero_pad_extents_;
};

// forward
using Sliced3DTensor = SlicedTensor<3>;
using Sliced4DTensor = SlicedTensor<4>;

struct InputIndexer;
struct KernelIndexer;

class Conv2DHelper {
 public:
  Conv2DHelper(const Conv2DProtocol::Meta &meta, const Shape3D &subshape);

  int64_t slice_size(int d) const {
    SPU_ENFORCE(d >= 0 && d < 3);
    return slices_[d];
  }

  int64_t num_slices() const { return calcNumel(slices_); }

  Shape3D GetSliceShape(const Shape3D &indices) const;

  Sliced3DTensor Slice(const ArrayRef &base, const Shape3D &base_shape,
                       const Shape3D &slice_index) const;

  void GetResultCoefficients(Shape3D slice_index,
                             std::vector<size_t> *coefficients,
                             Shape3D *oshape = nullptr) const;

 private:
  Conv2DProtocol::Meta meta_;
  Shape3D subshape_;
  Shape3D partition_windows_;

  Shape3D slices_;
};

struct InputIndexer {
 public:
  InputIndexer(Shape3D input_shape, Shape3D kernel_shape);

  int64_t operator()(int64_t h, int64_t w, int64_t c) const;

 private:
  Shape3D shape_;
  int64_t offset_;
};

struct KernelIndexer {
 public:
  KernelIndexer(Shape3D input_shape, Shape3D kernel_shape);

  int64_t operator()(int64_t h, int64_t w, int64_t c) const;

  int64_t index_begin() const { return begin_; }

 private:
  Shape3D shape_;
  int64_t row_nskip_, offset_, begin_;
};

}  // namespace spu::mpc::cheetah
