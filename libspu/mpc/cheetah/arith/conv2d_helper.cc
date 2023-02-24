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
#include "libspu/mpc/cheetah/arith/conv2d_helper.h"

namespace spu::mpc::cheetah {
// Layout:
//   NxHxWxC for input
//   HxWxIxO for kernel
[[maybe_unused]] constexpr int kH = 0;
[[maybe_unused]] constexpr int kW = 1;
[[maybe_unused]] constexpr int kC = 2;
[[maybe_unused]] constexpr int kO = 3;

InputIndexer::InputIndexer(Shape3D ishape, Shape3D fshape) {
  SPU_ENFORCE_EQ(ishape[kC], fshape[kC]);
  shape_ = ishape;
  offset_ = shape_[kH] * shape_[kW];
}

int64_t InputIndexer::operator()(int64_t h, int64_t w, int64_t c) const {
  SPU_ENFORCE(c >= 0 && h >= 0 && w >= 0,
              fmt::format("invalid negative index ({}, {}, {})", c, h, w));

  SPU_ENFORCE(c < shape_[kC] && h < shape_[kH] && w < shape_[kW],
              fmt::format("index out-of-bound ({}, {}, {})", c, h, w));

  return c * offset_ + h * shape_[kW] + w;
}

KernelIndexer::KernelIndexer(Shape3D ishape, Shape3D fshape) {
  shape_ = fshape;
  SPU_ENFORCE_EQ(ishape[kC], fshape[kC]);
  row_nskip_ = ishape[kW];

  offset_ = ishape[kH] * ishape[kW];
  // O = HW*(C-1) + W*(h-1) + (h'-1)
  begin_ = offset_ * (fshape[kC] - 1) + ishape[kW] * (fshape[kH] - 1) +
           fshape[kW] - 1;
}

int64_t KernelIndexer::operator()(int64_t h, int64_t w, int64_t c) const {
  SPU_ENFORCE(c >= 0 && h >= 0 && w >= 0,
              fmt::format("invalid negative index ({}, {}, {})", c, h, w));

  SPU_ENFORCE(c < shape_[kC] && h < shape_[kH] && w < shape_[kW],
              fmt::format("index out-of-bound ({}, {}, {})", c, h, w));

  // O - c*H*W - l*W - l'
  return begin_ - c * offset_ - h * row_nskip_ - w;
}

template <int Dim>
SlicedTensor<Dim>::SlicedTensor(const ArrayRef &base, const Shape &base_shape,
                                const Shape &offsets, const Shape &extents)
    : base_(base),
      base_shape_(base_shape),
      offsets_(offsets),
      extents_(extents),
      zero_pad_extents_(extents) {
  SPU_ENFORCE_EQ(base_.numel(), calcNumel(base_shape_));
  flatten_strides_[Dim - 1] = 1;
  for (int d = Dim - 1; d > 0; --d) {
    flatten_strides_[d - 1] = base_shape_[d] * flatten_strides_[d];
  }
}

template <int Dim>
SlicedTensor<Dim> SlicedTensor<Dim>::Wrap(const ArrayRef &base,
                                          const Shape &shape,
                                          const Shape &offsets,
                                          const Shape &extents) {
  SPU_ENFORCE_EQ(base.numel(), calcNumel(shape));

  for (int d = 0; d < Dim; ++d) {
    SPU_ENFORCE(extents[d] > 0 && shape[d] >= extents[d]);
    SPU_ENFORCE(offsets[d] >= 0);
  }

  return SlicedTensor<Dim>(base, shape, offsets, extents);
}

Conv2DHelper::Conv2DHelper(const Conv2DProtocol::Meta &meta,
                           const Shape3D &subshape)
    : meta_(meta), subshape_(subshape) {
  for (int d : {0, 1, 2}) {
    SPU_ENFORCE(subshape[d] > 0 && subshape[d] <= meta_.input_shape[d]);
  }
  // Ref: Section 3.2.1 of Cheetah's paper
  // We need to partition input of HxWxC into H'xW'xC sub-blocks
  // H' = Hs - hw + 1 and W' = Ws - hw + 1
  for (int d : {kH, kW}) {
    // We need a smaller partition window when kernel > 1
    int64_t goback = meta_.kernel_shape[d] - 1;
    partition_windows_[d] = subshape_[d] - goback;
    slices_[d] = CeilDiv(meta_.input_shape[d] - goback, partition_windows_[d]);
  }

  // Conv2D do not stride on the channel.
  partition_windows_[kC] = subshape_[kC];
  slices_[kC] = CeilDiv(meta_.input_shape[kC], partition_windows_[kC]);
}

Shape3D Conv2DHelper::GetSliceShape(
    const std::array<int64_t, 3> &indices) const {
  Shape3D sliced_shape;
  for (int d = 0; d < 3; ++d) {
    SPU_ENFORCE(indices[d] >= 0 && indices[d] < slices_[d]);
    int64_t start = indices[d] * partition_windows_[d];
    int64_t end = std::min(start + subshape_[d], meta_.input_shape[d]);
    sliced_shape[d] = end - start;
  }
  return sliced_shape;
}

Sliced3DTensor Conv2DHelper::Slice(
    const ArrayRef &base, const Shape3D &shape,
    const std::array<int64_t, 3> &indices) const {
  SPU_ENFORCE_EQ(base.numel(), calcNumel(shape));
  Shape3D extents = GetSliceShape(indices);

  Shape3D offsets;
  for (int d = 0; d < 3; ++d) {
    offsets[d] = indices[d] * partition_windows_[d];

    // santi check
    int64_t clipped_size =
        std::min(meta_.input_shape[d], offsets[d] + extents[d]);
    clipped_size -= offsets[d];
    SPU_ENFORCE(clipped_size > 0);
  }
  return Sliced3DTensor::Wrap(base, shape, offsets, extents);
}

void Conv2DHelper::GetResultCoefficients(std::array<int64_t, 3> indices,
                                         std::vector<size_t> *coefficients,
                                         Shape3D *oshape) const {
  SPU_ENFORCE(coefficients != nullptr);

  Shape3D ishape = subshape_;
  Shape3D kshape = meta_.kernel_shape;
  indices[kC] = 0;
  Shape3D slice_shape = GetSliceShape(indices);
  // We omit the 0-pad over the channel axis.
  ishape[kC] = slice_shape[kC];
  kshape[kC] = slice_shape[kC];

  // result coefficient indices are computed
  // via using InputIndexer with "O" offset.
  InputIndexer input_indexer(ishape, kshape);
  KernelIndexer kernel_indexer(ishape, kshape);

  Shape3D out_shape;
  std::array<int64_t, 2> offsets;
  const auto &strides = meta_.window_strides;
  for (int d : {kH, kW}) {
    // handle subtensor on the margins
    int64_t tmp = (indices[d] * partition_windows_[d]) % strides[d];
    offsets[d] = (meta_.window_strides[d] - tmp) % strides[d];

    out_shape[d] =
        (slice_shape[d] - kshape[d] + strides[d] - offsets[d]) / strides[d];
  }
  out_shape[kC] = 1;

  coefficients->resize(calcNumel(out_shape));
  auto *dst_ptr = coefficients->data();
  auto O = kernel_indexer.index_begin();
  for (int h = 0; h < out_shape[kH]; ++h) {
    for (int w = 0; w < out_shape[kW]; ++w) {
      *dst_ptr++ = static_cast<size_t>(
          O + input_indexer(h * meta_.window_strides[0] + offsets[0],
                            w * meta_.window_strides[1] + offsets[1], 0));
    }
  }

  if (oshape != nullptr) {
    *oshape = out_shape;
  }
}

}  // namespace spu::mpc::cheetah
