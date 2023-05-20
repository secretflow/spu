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

#include "libspu/core/ndarray_ref.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <utility>

#include "fmt/format.h"
#include "fmt/ostream.h"

#include "libspu/core/parallel_utils.h"

namespace spu {
namespace {

std::vector<int64_t> deducePadShape(
    absl::Span<const int64_t> input_shape,
    absl::Span<const int64_t> edge_padding_low,
    absl::Span<const int64_t> edge_padding_high,
    absl::Span<const int64_t> interior_padding) {
  std::vector<int64_t> dims;
  SPU_ENFORCE(edge_padding_low.size() == input_shape.size());
  SPU_ENFORCE(edge_padding_high.size() == input_shape.size());
  SPU_ENFORCE(interior_padding.size() == input_shape.size());
  for (size_t i = 0; i < input_shape.size(); i++) {
    dims.emplace_back(edge_padding_low[i] + edge_padding_high[i] +
                      interior_padding[i] * (input_shape[i] - 1) +
                      input_shape[i]);
  }

  return dims;
}

// Reference:
// https://github.com/numpy/numpy/blob/c652fcbd9c7d651780ea56f078c8609932822cf7/numpy/core/src/multiarray/shape.c#L371
static bool attempt_nocopy_reshape(const NdArrayRef& old,
                                   absl::Span<const int64_t> new_shape,
                                   std::vector<int64_t>& new_strides) {
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

}  // namespace

// full constructor
NdArrayRef::NdArrayRef(std::shared_ptr<yacl::Buffer> buf, Type eltype,
                       std::vector<int64_t> shape, std::vector<int64_t> strides,
                       int64_t offset)
    : buf_(std::move(buf)),
      eltype_(std::move(eltype)),
      shape_(std::move(shape)),
      strides_(std::move(strides)),
      offset_(offset) {}

// constructor, view buf as a compact buffer with given shape.
NdArrayRef::NdArrayRef(std::shared_ptr<yacl::Buffer> buf, Type eltype,
                       absl::Span<const int64_t> shape)
    : NdArrayRef(std::move(buf),             // buf
                 std::move(eltype),          // eltype
                 shape,                      // shape
                 makeCompactStrides(shape),  // strides
                 0                           // offset
      ) {}

// constructor, create a new buffer of elements and ref to it.
NdArrayRef::NdArrayRef(const Type& eltype, absl::Span<const int64_t> shape)
    : NdArrayRef(std::make_shared<yacl::Buffer>(calcNumel(shape) *
                                                eltype.size()),  // buf
                 eltype,                                         // eltype
                 shape,                                          // shape
                 makeCompactStrides(shape),                      // strides
                 0                                               // offset
      ) {}

size_t NdArrayRef::dim(size_t idx) const {
  SPU_ENFORCE(idx < ndim());
  return shape_[idx];
}

NdArrayRef NdArrayRef::as(const Type& new_ty, bool force) const {
  if (!force) {
    SPU_ENFORCE(elsize() == new_ty.size(),
                "viewed type={} not equal to origin type={}", new_ty, eltype());
    return NdArrayRef(buf(), new_ty, shape(), strides(), offset());
  }
  // Force view, we need to adjust strides
  auto distance = ((strides().empty() ? 1 : strides().back()) * elsize());
  SPU_ENFORCE(distance % new_ty.size() == 0);

  std::vector<int64_t> new_strides = strides();
  std::transform(new_strides.begin(), new_strides.end(), new_strides.begin(),
                 [&](int64_t s) { return (elsize() * s) / new_ty.size(); });

  return NdArrayRef(buf(), new_ty, shape(), new_strides, offset());
}

int64_t NdArrayRef::numel() const { return calcNumel(shape()); }

bool NdArrayRef::isCompact() const {
  return makeCompactStrides(shape()) == strides();
}

NdArrayRef NdArrayRef::clone() const {
  NdArrayRef res(eltype(), shape());

  auto elsize = res.elsize();

  auto src_iter = cbegin();

  auto* ret_ptr = static_cast<std::byte*>(res.data());

  for (int64_t idx = 0, e = numel(); idx < e; ++idx) {
    std::memcpy(ret_ptr + idx * elsize, src_iter.getRawPtr(), elsize);
    ++src_iter;
  }

  return res;
}

void NdArrayRef::copy_slice(const NdArrayRef& src,
                            absl::Span<const int64_t> src_base,
                            absl::Span<const int64_t> dst_base,
                            int64_t num_copy) {
  NdArrayRef::Iterator src_iter(src, src_base);
  NdArrayRef::Iterator dst_iter(*this, dst_base);
  auto elsize = this->elsize();
  for (int64_t counter = 0; counter < num_copy;
       ++counter, ++src_iter, ++dst_iter) {
#ifdef ITER_DEBUG
    SPU_ENFORCE(src_iter.validate());
    SPU_ENFORCE(dst_iter.validate());
    // SPDLOG_INFO(src_iter);
    // SPDLOG_INFO(dst_iter);
#endif
    std::memcpy(dst_iter.getRawPtr(), src_iter.getRawPtr(), elsize);
  }
}

NdArrayRef NdArrayRef::broadcast_to(absl::Span<const int64_t> to_shape,
                                    absl::Span<const int64_t> in_dims) const {
  for (auto d : in_dims) {
    SPU_ENFORCE(d < (int64_t)to_shape.size() && d >= 0,
                "Broadcast dim {} out of valid range [0, {})", d,
                to_shape.size());
  }

  std::vector<int64_t> new_strides(to_shape.size(), 0);

  if (!in_dims.empty()) {
    for (size_t idx = 0; idx < in_dims.size(); ++idx) {
      new_strides[in_dims[idx]] = strides()[idx];
    }
  } else {
    for (size_t idx = 0; idx < strides().size(); ++idx) {
      new_strides[new_strides.size() - 1 - idx] =
          strides()[strides().size() - 1 - idx];
    }
  }

  return NdArrayRef(buf(), eltype(), to_shape, new_strides, offset());
}

NdArrayRef NdArrayRef::reshape(absl::Span<const int64_t> to_shape) const {
  // Nothing to reshape
  if (shape() == to_shape) {
    return *this;
  }

  SPU_ENFORCE(calcNumel(shape()) == calcNumel(to_shape),
              "reshape from {} to {} is changing numel", shape(), to_shape);

  std::vector<int64_t> new_strides(to_shape.size(), 0);
  if (attempt_nocopy_reshape(*this, to_shape, new_strides)) {
    // No copy reshape
    return NdArrayRef(buf(), eltype(), to_shape, new_strides, offset());
  }

  auto compact_clone = clone();
  return NdArrayRef(compact_clone.buf(), compact_clone.eltype(), to_shape);
}

NdArrayRef NdArrayRef::slice(absl::Span<const int64_t> start_indices,
                             absl::Span<const int64_t> end_indices,
                             absl::Span<const int64_t> slice_strides) const {
  SPU_ENFORCE(shape().size() == start_indices.size());
  SPU_ENFORCE(shape().size() == end_indices.size());
  SPU_ENFORCE(slice_strides.empty() ||
              (shape().size() == slice_strides.size()));

  std::vector<int64_t> new_shape(shape().size(), 0);
  std::vector<int64_t> new_strides(strides());
  for (size_t idx = 0; idx < shape().size(); ++idx) {
    SPU_ENFORCE(end_indices[idx] <= shape()[idx],
                "Slice end at axis {} = {} is larger than input shape {}", idx,
                end_indices[idx], shape()[idx]);
    new_shape[idx] = std::max(end_indices[idx] - start_indices[idx],
                              static_cast<int64_t>(0));

    if (new_shape[idx] == 1) {
      new_strides[idx] = 0;
    } else if (!slice_strides.empty()) {
      auto n = new_shape[idx] / slice_strides[idx];
      auto q = new_shape[idx] % slice_strides[idx];
      new_shape[idx] = n + static_cast<int64_t>(q != 0);
      new_strides[idx] *= slice_strides[idx];
    }
  }

  return NdArrayRef(buf(), eltype(), new_shape, new_strides,
                    &at(start_indices) - buf()->data<std::byte>());
}

NdArrayRef NdArrayRef::slice_scalar_at(
    absl::Span<const int64_t> indices) const {
  return NdArrayRef(buf(), eltype(), {}, {},
                    &at(indices) - buf()->data<std::byte>());
}

NdArrayRef NdArrayRef::transpose(absl::Span<const int64_t> permutation) const {
  std::vector<int64_t> perm(shape().size());
  if (permutation.empty()) {
    for (size_t i = 0; i < perm.size(); ++i) {
      perm[i] = static_cast<int64_t>(shape().size()) - 1 - i;
    }
  } else {
    std::vector<int64_t> reverse_permutation(shape().size(), -1);
    SPU_ENFORCE(permutation.size() == shape().size(),
                "axes don't match array, permutation = {}, input shape = {}",
                fmt::join(permutation, "x"), fmt::join(shape(), "x"));

    for (size_t i = 0; i < permutation.size(); i++) {
      auto axis = permutation[i];
      SPU_ENFORCE(reverse_permutation[axis] == -1,
                  "repeated axis in transpose");
      reverse_permutation[axis] = i;
      perm[i] = axis;
    }
  }

  std::vector<int64_t> ret_shape(shape().size());
  std::vector<int64_t> ret_strides(strides().size());

  for (size_t i = 0; i < shape().size(); i++) {
    ret_shape[i] = shape()[perm[i]];
    ret_strides[i] = strides()[perm[i]];
  }

  return NdArrayRef{buf(), eltype(), ret_shape, ret_strides, offset()};
}

NdArrayRef NdArrayRef::reverse(absl::Span<const int64_t> dimensions) const {
  std::vector<int64_t> new_strides = strides();
  int64_t el_offset = 0;

  for (int64_t axis : dimensions) {
    SPU_ENFORCE(axis < static_cast<int64_t>(shape().size()));
    new_strides[axis] *= -1;
    el_offset += strides()[axis] * (shape()[axis] - 1);
  }

  return NdArrayRef(buf(), eltype(), shape(), new_strides,
                    offset() + el_offset * elsize());
}

NdArrayRef NdArrayRef::expand(absl::Span<const int64_t> to_shape) const {
  SPU_ENFORCE(numel() == 1, "Only support expanding scalar");
  NdArrayRef ret(eltype(), to_shape);
  // compute number of elements need to copy
  size_t numel = ret.numel();
  size_t num_bytes = numel * elsize();
  size_t bytes_copied = elsize();

  // Copy first element
  std::memcpy(ret.data(), data(), elsize());

  while (bytes_copied != num_bytes) {
    size_t copy_size = std::min(bytes_copied, num_bytes - bytes_copied);
    std::memcpy(static_cast<char*>(ret.data()) + bytes_copied, ret.data(),
                copy_size);
    bytes_copied += copy_size;
  }
  return ret;
}

NdArrayRef NdArrayRef::concatenate(absl::Span<const NdArrayRef> others,
                                   const size_t& axis) const {
  std::vector<int64_t> result_shape = shape();
  for (const auto& o : others) {
    result_shape[axis] += o.shape()[axis];
  }

  // Preallocate output buffer
  NdArrayRef result(eltype(), result_shape);

  // Copy self
  std::vector<int64_t> base(shape().size(), 0);

  std::vector<int64_t> slice_start(shape().size(), 0);
  std::vector<int64_t> slice_end = shape();
  std::vector<int64_t> slice_stride(shape().size(), 1);

  {
    auto r1 = result.slice(slice_start, slice_end, slice_stride);
    r1.copy_slice(*this, base, base, numel());
    slice_start[axis] = slice_end[axis];
  }

  // Copy other slices
  for (const auto& o : others) {
    slice_end[axis] += o.shape()[axis];
    auto s = result.slice(slice_start, slice_end, slice_stride);
    s.copy_slice(o, base, base, o.numel());
    slice_start[axis] = slice_end[axis];
  }

  return result;
}

NdArrayRef NdArrayRef::pad(const NdArrayRef& padding_value,
                           absl::Span<const int64_t> edge_padding_low,
                           absl::Span<const int64_t> edge_padding_high,
                           absl::Span<const int64_t> interior_padding) const {
  auto result = padding_value.expand(deducePadShape(
      shape(), edge_padding_low, edge_padding_high, interior_padding));

  const auto& result_shape = result.shape();
  const auto& input_shape = shape();

  // auto elsize = result.elsize();

  yacl::parallel_for(0, numel(), 1024, [&](int64_t begin, int64_t end) {
    std::vector<int64_t> unflatten = unflattenIndex(begin, input_shape);

    std::vector<int64_t> target_index(result_shape.size());
    for (int64_t idx = begin; idx < end; ++idx) {
      bool valid = true;
      for (size_t i = 0; i < unflatten.size(); ++i) {
        // Interior padding occurs logically before edge padding, so in the case
        // of negative edge padding elements are removed from the
        // interior-padded operand.
        target_index[i] =
            edge_padding_low[i] + unflatten[i] * (interior_padding[i] + 1);

        // Account for negative low and high padding: skip assignment if the
        // any target index is out of range.
        if (target_index[i] < 0 || target_index[i] >= result_shape[i]) {
          valid = false;
          break;
        }
      }
      if (valid) {
        std::memcpy(&result.at(target_index), &at(unflatten), elsize());
      }
      bumpIndices<int64_t>(shape(), absl::MakeSpan(unflatten));
    }
  });

  return result;
}

NdArrayRef NdArrayRef::linear_gather(absl::Span<const int64_t> indices) const {
  SPU_ENFORCE(shape().size() == 1);

  NdArrayRef result(eltype(), {static_cast<int64_t>(indices.size())});

  auto result_iter = result.begin();

  const auto* src_ptr = static_cast<const std::byte*>(data());

  auto elsize = this->elsize();

  for (const auto& idx : indices) {
    std::memcpy(result_iter.getRawPtr(), src_ptr + idx * strides_[0] * elsize,
                elsize);
    ++result_iter;
  }

  return result;
}

NdArrayRef& NdArrayRef::linear_scatter(const NdArrayRef& new_values,
                                       absl::Span<const int64_t> indices) {
  SPU_ENFORCE(shape().size() == 1);
  SPU_ENFORCE(new_values.eltype() == eltype(),
              "new value eltype = {}, expected = {}", new_values.eltype(),
              eltype());

  auto new_values_iter = new_values.cbegin();

  auto* dst_ptr = static_cast<std::byte*>(data());
  auto elsize = this->elsize();

  for (const auto& idx : indices) {
    std::memcpy(dst_ptr + idx * strides_[0] * elsize,
                new_values_iter.getRawPtr(), elsize);
    ++new_values_iter;
  }

  return *this;
}

void NdArrayRef::eliminate_zero_stride() {
  bool has_valid_zero_stride = false;
  // If shape[dim] == 1, 0 stride is fine
  for (size_t idim = 0; idim < shape_.size(); ++idim) {
    if (shape_[idim] != 1 && strides_[idim] == 0) {
      has_valid_zero_stride = true;
      break;
    }
  }

  if (!has_valid_zero_stride) {
    return;
  }

  // Get a clone
  auto clone = this->clone();

  // Swap to cloned
  std::swap(*this, clone);
}

void NdArrayRef::update_slice(const NdArrayRef& new_value,
                              absl::Span<const int64_t> start_indices) {
  if (new_value.numel() == 0) {
    return;
  }

  eliminate_zero_stride();

  auto elsize = this->elsize();

  // Fast path for scalar copy...
  if (new_value.numel() == 1) {
    NdArrayRef::Iterator in(*this, start_indices);
    std::memcpy(in.getRawPtr(), new_value.data(), elsize);
    return;
  }
  // Slice copy
  std::vector<int64_t> end_indices(start_indices.begin(), start_indices.end());
  for (size_t idx = 0; idx < end_indices.size(); ++idx) {
    end_indices[idx] += new_value.shape()[idx];
  }

  auto slice = this->slice(start_indices, end_indices,
                           std::vector<int64_t>(start_indices.size(), 1));

  // Just a sanity check....
  SPU_ENFORCE(slice.buf_->data() == this->buf_->data());

  auto src_iter = new_value.cbegin();
  auto src_end = new_value.cend();
  auto dst_iter = slice.begin();
  auto dst_end = slice.end();
  for (; src_iter != src_end; ++src_iter, ++dst_iter) {
    std::memcpy(dst_iter.getRawPtr(), src_iter.getRawPtr(), elsize);
  }
}

NdArrayRef unflatten(const ArrayRef& arr, absl::Span<const int64_t> shape) {
  SPU_ENFORCE(arr.numel() == calcNumel(shape),
              "unflatten numel mismatch, expected={}, got={}", calcNumel(shape),
              arr.numel());

  if (arr.stride() == 0) {
    return NdArrayRef(arr.buf(), arr.eltype(), shape,
                      std::vector<int64_t>(shape.size(), 0), arr.offset());
  }

  ArrayRef compact = arr.isCompact() ? arr : arr.clone();

  auto strides = makeCompactStrides(shape);
  return NdArrayRef(compact.buf(), compact.eltype(), shape, std::move(strides),
                    compact.offset());
}

namespace {
bool onlyStrideInnerMostDim(absl::Span<const int64_t> shape,
                            absl::Span<const int64_t> strides) {
  auto expect_stride = strides.back() * shape.back();
  for (int64_t dim = strides.size() - 2; dim >= 0; --dim) {
    if (strides[dim] != expect_stride) {
      return false;
    }
    expect_stride *= shape[dim];
  }
  return true;
}
}  // namespace

ArrayRef flatten(const NdArrayRef& ndarr) {
  if (ndarr.isCompact()) {
    // if compact, direct treat it as a 1D array.
    // TODO: optimize me, in some cases, we may treat it as a strided 1d-array
    // even ndarray is not compact.
    return ArrayRef(ndarr.buf(), ndarr.eltype(), ndarr.numel(), 1,
                    ndarr.offset());
  }

  // Basically it's a scalar broadcasted into some shape
  if (std::all_of(ndarr.strides().begin(), ndarr.strides().end(),
                  [](int64_t in) { return in == 0; })) {
    // SPDLOG_INFO("fast zero stride flatten");
    return ArrayRef(ndarr.buf(), ndarr.eltype(), ndarr.numel(), 0,
                    ndarr.offset());
  }

  // Check if only inner most dim has strides
  if (onlyStrideInnerMostDim(ndarr.shape(), ndarr.strides())) {
    // SPDLOG_INFO("fast innermost only stride flatten");
    return ArrayRef(ndarr.buf(), ndarr.eltype(), ndarr.numel(),
                    ndarr.strides().back(), ndarr.offset());
  }

  // create a compact clone, it's save here since underline layer will never
  // modify inplace.
  auto compact = ndarr.clone();
  return ArrayRef(compact.buf(), ndarr.eltype(), ndarr.numel(), 1,
                  compact.offset());
}

NdArrayRef::Iterator& NdArrayRef::Iterator::operator++() {
  if (!invalid_) {
    int64_t idim;
    for (idim = shape_.size() - 1; idim >= 0; --idim) {
      if (++coord_[idim] == shape_[idim]) {
        // Once a dimension is done, just unwind by strides
        coord_[idim] = 0;
        ptr_ -= (shape_[idim] - 1) * strides_[idim] * elsize_;
      } else {
        ptr_ += strides_[idim] * elsize_;
        break;
      }
    }
    // Mark invalid
    if (idim == -1) {
      invalid_ = true;
    }
  }
  return *this;
}

std::ostream& operator<<(std::ostream& out, const NdArrayRef& v) {
  out << fmt::format("NdArrayRef<{}x{}>", fmt::join(v.shape(), "x"),
                     v.eltype().toString());
  return out;
}

}  // namespace spu
