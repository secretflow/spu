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

// Reference:
// https://github.com/numpy/numpy/blob/c652fcbd9c7d651780ea56f078c8609932822cf7/numpy/core/src/multiarray/shape.c#L371
static bool attempt_nocopy_reshape(const NdArrayRef &old,
                                   absl::Span<const int64_t> new_shape,
                                   std::vector<int64_t> &new_strides);

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

NdArrayRef SliceDim(const NdArrayRef &in, int64_t dim, int64_t at) {
  int64_t ndim = in.ndim();
  std::vector<int64_t> shape = in.shape();
  SPU_ENFORCE(dim >= 0 && dim < ndim);
  SPU_ENFORCE(at >= 0 && at < shape[dim]);

  std::vector<int64_t> start_indices(ndim, 0);
  std::vector<int64_t> end_indices = shape;
  std::vector<int64_t> strides(ndim, 1);
  // clip on the target dimension
  start_indices[dim] = at;
  end_indices[dim] = at + 1;

  std::vector<int64_t> new_shape(ndim, 0);
  std::vector<int64_t> new_strides(in.strides());
  for (int64_t idx = 0; idx < ndim; ++idx) {
    SPU_ENFORCE(end_indices[idx] <= shape[idx],
                "Slice end at axis {} = {} is larger than input shape {}", idx,
                end_indices[idx], shape[idx]);
    new_shape[idx] = std::max(end_indices[idx] - start_indices[idx],
                              static_cast<int64_t>(0));
    if (!strides.empty()) {
      auto n = new_shape[idx] / strides[idx];
      auto q = new_shape[idx] % strides[idx];
      new_shape[idx] = n + static_cast<int64_t>(q != 0);
      new_strides[idx] *= strides[idx];
    }
  }

  // Ref to hal::slice()
  auto ret = NdArrayRef(in.buf(), in.eltype(), new_shape, new_strides,
                        &in.at(start_indices) - in.buf()->data<std::byte>());
  // Ref to hal::reshape()
  std::vector<int64_t> to_shape = new_shape;
  // drop the target dimension
  to_shape.erase(to_shape.begin() + dim);
  new_strides.resize(to_shape.size());
  if (attempt_nocopy_reshape(ret, to_shape, new_strides)) {
    return NdArrayRef(ret.buf(), ret.eltype(), to_shape, new_strides,
                      ret.offset());
  }
  // we need to make clone in this case for reshape
  ret = ret.clone();  // compact clone
  return NdArrayRef(ret.buf(), ret.eltype(), to_shape);
}

Sliced3DTensor::Sliced3DTensor(const NdArrayRef &base, const Shape3D &offsets,
                               const Shape3D &extents)
    : base_(base),
      offsets_(offsets),
      extents_(extents),
      mock_extents_(extents) {}

Sliced3DTensor Sliced3DTensor::Wrap(const NdArrayRef &base,
                                    const Shape3D &offsets,
                                    const Shape3D &extents) {
  SPU_ENFORCE_EQ(base.ndim(), 3UL);

  for (int d = 0; d < 3; ++d) {
    SPU_ENFORCE(extents[d] > 0 &&
                base.dim(d) >= static_cast<size_t>(extents[d]));
    SPU_ENFORCE(offsets[d] >= 0);
  }

  return Sliced3DTensor(base, offsets, extents);
}

FieldType Sliced3DTensor::field() const {
  const Type &eltype = base_.eltype();
  return eltype.as<Ring2k>()->field();
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

Sliced3DTensor Conv2DHelper::partition(
    const NdArrayRef &base, const std::array<int64_t, 3> &indices) const {
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
  return Sliced3DTensor::Wrap(base, offsets, extents);
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

bool attempt_nocopy_reshape(const NdArrayRef &old,
                            absl::Span<const int64_t> new_shape,
                            std::vector<int64_t> &new_strides) {
  size_t oldnd;
  std::vector<int64_t> olddims(old.shape().size());
  std::vector<int64_t> oldstrides(old.strides().size());
  size_t oi;
  size_t oj;
  size_t ok;
  size_t ni;
  size_t nj;
  size_t nk;

  oldnd = 0;
  /*
   * Remove axes with dimension 1 from the old array. They have no effect
   * but would need special cases since their strides do not matter.
   */
  for (oi = 0; oi < old.shape().size(); oi++) {
    if (old.shape()[oi] != 1) {
      olddims[oldnd] = old.shape()[oi];
      oldstrides[oldnd] = old.strides()[oi];
      oldnd++;
    }
  }

  /* oi to oj and ni to nj give the axis ranges currently worked with */
  oi = 0;
  oj = 1;
  ni = 0;
  nj = 1;
  while (ni < new_shape.size() && oi < oldnd) {
    auto np = new_shape[ni];
    auto op = olddims[oi];

    while (np != op) {
      if (np < op) {
        /* Misses trailing 1s, these are handled later */
        np *= new_shape[nj++];
      } else {
        op *= olddims[oj++];
      }
    }

    /* Check whether the original axes can be combined */
    for (ok = oi; ok < oj - 1; ok++) {
      if (oldstrides[ok] != olddims[ok + 1] * oldstrides[ok + 1]) {
        /* not contiguous enough */
        return false;
      }
    }

    /* Calculate new strides for all axes currently worked with */
    new_strides[nj - 1] = oldstrides[oj - 1];
    for (nk = nj - 1; nk > ni; nk--) {
      new_strides[nk - 1] = new_strides[nk] * new_shape[nk];
    }

    ni = nj++;
    oi = oj++;
  }

  for (size_t idx = 0; idx < new_shape.size(); ++idx) {
    if (new_shape[idx] == 1) {
      // During attempt_nocopy_reshape strides for 1 sized dimensions are not
      // set to 0, which can be a problem if this value is later broadcasted
      // in this dimension, so force set to 0 here
      new_strides[idx] = 0;
    }
  }

  return true;
}

}  // namespace spu::mpc::cheetah
