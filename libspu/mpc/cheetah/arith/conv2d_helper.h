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
#include "libspu/core/array_ref.h"
#include "libspu/core/ndarray_ref.h"
#include "libspu/core/shape_util.h"
#include "libspu/mpc/cheetah/arith/conv2d_prot.h"

namespace spu::mpc::cheetah {

// forward
struct Sliced3DTensor;
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

// A slice of a 3D tensor with zero padding.
// The slice is defined by `offsets` and `extents`
struct Sliced3DTensor {
 private:
  Sliced3DTensor(const ArrayRef &base, const Shape3D &base_shape,
                 const Shape3D &offsets, const Shape3D &extents);

 public:
  static Sliced3DTensor Wrap(const ArrayRef &base, const Shape3D &base_shape,
                             const Shape3D &offsets, const Shape3D &extents);

  Sliced3DTensor(const Sliced3DTensor &oth) = default;

  Sliced3DTensor(Sliced3DTensor &&oth) = default;

  Sliced3DTensor &operator=(const Sliced3DTensor &oth) = delete;

  template <typename T>
  T at(int64_t h, int64_t w, int64_t c) const {
    // NOTE: HxWxC order
    constexpr int kH = 0;
    constexpr int kW = 1;
    constexpr int kC = 2;

    // sementic check
    SPU_ENFORCE(h >= 0 && h < zero_pad_extents_[kH]);
    SPU_ENFORCE(w >= 0 && w < zero_pad_extents_[kW]);
    SPU_ENFORCE(c >= 0 && c < zero_pad_extents_[kC]);

    // zero padding
    if (c >= extents_[kC]) {
      return static_cast<T>(0);
    }
    if (h < 0 || h >= extents_[kH]) {
      return static_cast<T>(0);
    }
    if (w < 0 || w >= extents_[kW]) {
      return static_cast<T>(0);
    }

    std::array<int64_t, 3> index = {h + offsets_[kH], w + offsets_[kW],
                                    c + offsets_[kC]};
    // see core/ndarray_ref.h
    int64_t offset =
        spu::detail::calcFlattenOffset(index, base_shape_, flatten_strides_);
    SPU_ENFORCE(offset >= 0 && offset < base_.numel());
    return base_.at<T>(offset);
  }

  Shape3D shape() const { return zero_pad_extents_; }

  int64_t numel() const { return calcNumel(shape()); }

  FieldType field() const;

  void ZeroPadAs(const Shape3D &extents) {
    for (size_t d = 0; d < 3; ++d) {
      SPU_ENFORCE(extents[d] > 0);
    }
    zero_pad_extents_ = extents;
  }

 private:
  const ArrayRef &base_;
  Shape3D base_shape_;
  Shape3D offsets_;
  Shape3D extents_;
  Shape3D flatten_strides_;

  Shape3D zero_pad_extents_;
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
