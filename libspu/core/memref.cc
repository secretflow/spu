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

#include "libspu/core/memref.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <utility>

namespace spu {
namespace {

Shape deducePadShape(const Shape& input_shape, const Sizes& edge_padding_low,
                     const Sizes& edge_padding_high) {
  Shape dims;
  SPU_ENFORCE(edge_padding_low.size() == input_shape.size());
  SPU_ENFORCE(edge_padding_high.size() == input_shape.size());
  for (size_t i = 0; i < input_shape.size(); i++) {
    SPU_ENFORCE(edge_padding_low[i] >= 0 && edge_padding_high[i] >= 0,
                "Negative padding is not supported");
    dims.emplace_back(edge_padding_low[i] + edge_padding_high[i] +
                      input_shape[i]);
  }

  return dims;
}

Visibility getVisibilityFromType(const Type& ty) {
  if (ty.isa<Secret>()) {
    return VIS_SECRET;
  } else if (ty.isa<Public>()) {
    return VIS_PUBLIC;
  } else if (ty.isa<Private>()) {
    return VIS_PRIVATE;
  } else {
    return VIS_INVALID;
  }
}

// Reference:
// https://github.com/numpy/numpy/blob/c652fcbd9c7d651780ea56f078c8609932822cf7/numpy/core/src/multiarray/shape.c#L371
static bool attempt_nocopy_reshape(const MemRef& old,
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

std::pair<bool, Stride> can_use_fast_indexing(const Shape& shape,
                                              const Strides& strides) {
  Shape stripped_shape;
  Strides stripped_strides;

  for (size_t idx = 0; idx < shape.size(); ++idx) {
    // Strip all dim == 1
    if (shape[idx] == 1) {
      continue;
    }
    stripped_shape.emplace_back(shape[idx]);
    stripped_strides.emplace_back(strides[idx]);
  }

  if (stripped_shape.isScalar()) {
    return {true, 0};  // This is eventually a scalar...
  }

  auto linear_strides = stripped_strides.back();
  auto compact_strides = makeCompactStrides(stripped_shape);

  // So idea here:
  // Let's say there is an array with shape (3,4), with strides (4, 1), it is a
  // compact array But with (8, 2), which means we are only skipping certain
  // columns, it is still capable to do fast indexing.
  // But when the strides is (16, 2), which means we also skipping certain rows,
  // to compute actual offset, we have to use a slow path.

  // See MemRefTest.Indexing test point.
  for (size_t idx = 0; idx < compact_strides.size(); ++idx) {
    if (linear_strides * compact_strides[idx] != stripped_strides[idx]) {
      return {false, 0};
    }
  }
  return {true, linear_strides};
}

}  // namespace

// full constructor
MemRef::MemRef(std::shared_ptr<yacl::Buffer> buf, Type eltype,
               const Shape& shape, const Strides& strides, int64_t offset,
               bool is_complex)
    : buf_(std::move(buf)),
      eltype_(std::move(eltype)),
      shape_(shape.begin(), shape.end()),
      strides_(strides.begin(), strides.end()),
      offset_(offset),
      is_complex_(is_complex) {
  std::tie(use_fast_indexing_, fast_indexing_stride_) =
      can_use_fast_indexing(shape_, strides_);
}

// constructor, view buf as a compact buffer with given shape.
MemRef::MemRef(std::shared_ptr<yacl::Buffer> buf, Type eltype,
               const Shape& shape, bool is_complex)
    : MemRef(std::move(buf),             // buf
             std::move(eltype),          // eltype
             shape,                      // shape
             makeCompactStrides(shape),  // strides
             0,                          // offset
             is_complex                  // is_complex
      ) {}

// constructor, create a new buffer of elements and ref to it.
MemRef::MemRef(const Type& eltype, const Shape& shape, bool is_complex)
    : MemRef(std::make_shared<yacl::Buffer>(shape.numel() * eltype.size() *
                                            (is_complex ? 2 : 1)),  // buf
             eltype,                                                // eltype
             shape,                                                 // shape
             makeCompactStrides(shape),                             // strides
             0,                                                     // offset
             is_complex  // is_complex
      ) {}

MemRef MemRef::as(const Type& new_ty, bool force) const {
  if (!force) {
    SPU_ENFORCE(elsize() == new_ty.size(),
                "viewed type={} not equal to origin type={}", new_ty, eltype());
    return MemRef(buf(), new_ty, shape(), strides(), offset());
  }
  // Force view, we need to adjust strides
  auto distance = ((strides().empty() ? 1 : strides().back()) * elsize());
  SPU_ENFORCE(distance % new_ty.size() == 0);

  Strides new_strides = strides();
  std::transform(new_strides.begin(), new_strides.end(), new_strides.begin(),
                 [&](int64_t s) { return (elsize() * s) / new_ty.size(); });

  return MemRef(buf(), new_ty, shape(), new_strides, offset());
}

MemRef MemRef::clone() const {
  MemRef res(eltype(), shape(), is_complex_);

  auto elsize = res.elsize();

  auto src_iter = cbegin();

  auto* ret_ptr = static_cast<std::byte*>(res.data());

  for (int64_t idx = 0, e = numel(); idx < e; ++idx, ++src_iter) {
    std::memcpy(ret_ptr + idx * elsize, src_iter.getRawPtr(), elsize);
  }

  return res;
}

void MemRef::copy_slice(const MemRef& src, const Index& src_base,
                        const Index& dst_base, int64_t num_copy) {
  MemRef::Iterator src_iter(src, src_base);
  MemRef::Iterator dst_iter(*this, dst_base);
  auto elsize = this->elsize();
  for (int64_t counter = 0; counter < num_copy;
       ++counter, ++src_iter, ++dst_iter) {
#ifdef ITER_DEBUG
    SPU_ENFORCE(src_iter.validate());
    SPU_ENFORCE(dst_iter.validate());
    // SPDLOG_INFO(src_iter);
    // SPDLOG_INFO(dst_iter);
#endif
    std::memcpy(&*dst_iter, &*src_iter, elsize);
  }
}

MemRef MemRef::broadcast_to(const Shape& to_shape, const Axes& in_dims) const {
  for (auto d : in_dims) {
    SPU_ENFORCE(d < (int64_t)to_shape.size() && d >= 0,
                "Broadcast dim {} out of valid range [0, {})", d,
                to_shape.size());
  }

  Strides new_strides(to_shape.size(), 0);

  // TODO: check to_shape match broadcasting rules.
  if (!in_dims.empty()) {
    for (size_t dim = 0, current_dim_idx = 0, in_dims_idx = 0;
         dim < to_shape.size(); ++dim) {
      if (in_dims_idx < in_dims.size() &&
          in_dims[in_dims_idx] == static_cast<int64_t>(dim)) {
        new_strides[dim] = 0;
        ++in_dims_idx;
      } else {
        new_strides[dim] = strides_[current_dim_idx];
        ++current_dim_idx;
      }
    }
  } else {
    for (size_t idx = 0; idx < strides().size(); ++idx) {
      new_strides[new_strides.size() - 1 - idx] =
          strides()[strides().size() - 1 - idx];
    }
  }

  return MemRef(buf(), eltype(), to_shape, new_strides, offset());
}

MemRef MemRef::reshape(const Shape& to_shape) const {
  // Nothing to reshape
  if (shape() == to_shape) {
    return *this;
  }

  SPU_ENFORCE(shape().numel() == to_shape.numel(),
              "reshape from {} to {} is changing numel", shape(), to_shape);

  // Reshape empty is always a noop
  if (to_shape.numel() == 0) {
    return MemRef(buf(), eltype(), to_shape, makeCompactStrides(to_shape),
                  offset());
  }

  Strides new_strides(to_shape.size(), 0);
  if (attempt_nocopy_reshape(*this, to_shape, new_strides)) {
    // No copy reshape
    return MemRef(buf(), eltype(), to_shape, new_strides, offset());
  }

  auto compact_clone = clone();
  return MemRef(compact_clone.buf(), compact_clone.eltype(), to_shape);
}

MemRef MemRef::slice(const Index& offset, const Shape& sizes,
                     const Strides& strides) const {
  SPU_ENFORCE(shape().size() == offset.size());
  SPU_ENFORCE(shape().size() == sizes.size());
  SPU_ENFORCE(strides.empty() || (shape().size() == strides.size()));

  Strides new_strides(this->strides());
  for (size_t idx = 0; idx < shape().size(); ++idx) {
    auto s = strides.empty() ? 1 : strides[idx];
    SPU_ENFORCE(
        offset[idx] + (sizes[idx] - 1) * s + 1 <= shape()[idx],
        "Cannot extract {} elements at dim {} with stride {}, dim size = {}",
        sizes[idx], offset[idx], s, shape()[idx]);
    if (sizes[idx] == 1) {
      new_strides[idx] = 0;
    } else if (!strides.empty()) {
      new_strides[idx] *= strides[idx];
    }
  }

  return MemRef(buf(), eltype(), sizes, new_strides,
                &at(offset) - buf()->data<std::byte>());
}

MemRef MemRef::slice_scalar_at(const Index& indices) const {
  return MemRef(buf(), eltype(), {}, {},
                &at(indices) - buf()->data<std::byte>());
}

MemRef MemRef::transpose() const {
  Axes perm;
  perm.resize(shape().size());
  std::iota(perm.rbegin(), perm.rend(), 0);
  return transpose(perm);
}

MemRef MemRef::transpose(const Axes& perm) const {
  // sanity check.
  SPU_ENFORCE_EQ(perm.size(), shape().size());
  std::set<int64_t> uniq(perm.begin(), perm.end());
  SPU_ENFORCE_EQ(uniq.size(), perm.size(), "perm={} is not unique", perm);

  Shape ret_shape(shape().size());
  Strides ret_strides(strides().size());
  for (size_t i = 0; i < shape().size(); i++) {
    ret_shape[i] = shape()[perm[i]];
    ret_strides[i] = strides()[perm[i]];
  }

  return MemRef{buf(), eltype(), ret_shape, ret_strides, offset()};
}

MemRef MemRef::reverse(const Axes& dimensions) const {
  Strides new_strides = strides();
  int64_t el_offset = 0;

  for (int64_t axis : dimensions) {
    SPU_ENFORCE(axis < static_cast<int64_t>(shape().size()));
    new_strides[axis] *= -1;
    el_offset += strides()[axis] * (shape()[axis] - 1);
  }

  return MemRef(buf(), eltype(), shape(), new_strides,
                offset() + el_offset * elsize());
}

MemRef MemRef::expand(const Shape& to_shape) const {
  SPU_ENFORCE(numel() == 1, "Only support expanding scalar");
  MemRef ret(eltype(), to_shape);
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

MemRef MemRef::concatenate(absl::Span<const MemRef> others,
                           int64_t axis) const {
  Shape result_shape = shape();
  for (const auto& o : others) {
    result_shape[axis] += o.shape()[axis];
  }

  // Preallocate output buffer
  MemRef result(eltype(), result_shape);

  // Copy self
  Index base(shape().size(), 0);
  Index slice_start(shape().size(), 0);
  Index slice_end(shape().begin(), shape().end());
  Strides slice_stride(shape().size(), 1);

  auto r1 = result.slice(slice_start, this->shape(), slice_stride);
  r1.copy_slice(*this, base, base, numel());
  slice_start[axis] += this->shape()[axis];

  // Copy other slices
  for (const auto& o : others) {
    auto s = result.slice(slice_start, o.shape(), slice_stride);
    s.copy_slice(o, base, base, o.numel());
    slice_start[axis] += o.shape()[axis];
  }

  return result;
}

MemRef MemRef::pad(const MemRef& padding_value, const Sizes& edge_padding_low,
                   const Sizes& edge_padding_high) const {
  auto result = padding_value.expand(
      deducePadShape(shape(), edge_padding_low, edge_padding_high));

  const auto& result_shape = result.shape();
  const auto& input_shape = shape();

  pforeach(0, numel(), [&](int64_t begin, int64_t end) {
    auto unflatten = unflattenIndex(begin, input_shape);

    Index target_index(result_shape.size());
    for (int64_t idx = begin; idx < end; ++idx) {
      for (size_t i = 0; i < unflatten.size(); ++i) {
        target_index[i] = edge_padding_low[i] + unflatten[i];
      }
      std::memcpy(&result.at(target_index), &at(unflatten), elsize());
      bumpIndices(shape(), absl::MakeSpan(unflatten));
    }
  });

  return result;
}

MemRef MemRef::linear_gather(const Index& indices) const {
  SPU_ENFORCE(shape().size() == 1);

  MemRef result(eltype(), {static_cast<int64_t>(indices.size())});

  auto result_iter = result.begin();

  const auto* src_ptr = static_cast<const std::byte*>(data());

  auto elsize = this->elsize();

  for (const auto& idx : indices) {
    std::memcpy(&*result_iter, src_ptr + idx * strides_[0] * elsize, elsize);
    ++result_iter;
  }

  return result;
}

MemRef& MemRef::linear_scatter(const MemRef& new_values, const Index& indices) {
  SPU_ENFORCE(shape().size() == 1);
  SPU_ENFORCE(new_values.eltype() == eltype(),
              "new value eltype = {}, expected = {}", new_values.eltype(),
              eltype());

  auto new_values_iter = new_values.cbegin();

  auto* dst_ptr = static_cast<std::byte*>(data());
  auto elsize = this->elsize();

  for (const auto& idx : indices) {
    std::memcpy(dst_ptr + idx * strides_[0] * elsize, &*new_values_iter,
                elsize);
    ++new_values_iter;
  }

  return *this;
}

void MemRef::eliminate_zero_stride() {
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

void MemRef::insert_slice(const MemRef& new_value, const Index& offsets,
                          const Strides& strides) {
  if (new_value.numel() == 0) {
    return;
  }

  SPU_ENFORCE(this->eltype() == new_value.eltype(),
              "origin eltype = {}, update eltype = {}", this->eltype(),
              new_value.eltype());

  eliminate_zero_stride();

  auto elsize = this->elsize();

  // Fast path for scalar copy...
  if (new_value.numel() == 1) {
    MemRef::Iterator in(*this, offsets);
    std::memcpy(&*in, new_value.data(), elsize);
    return;
  }

  auto slice = this->slice(offsets, new_value.shape(), strides);

  // Just a sanity check....
  SPU_ENFORCE(slice.buf_->data() == this->buf_->data());

  auto src_iter = new_value.cbegin();
  auto src_end = new_value.cend();
  auto dst_iter = slice.begin();
  auto dst_end = slice.end();
  for (; src_iter != src_end; ++src_iter, ++dst_iter) {
    std::memcpy(&*dst_iter, &*src_iter, elsize);
  }
}

MemRef::Iterator& MemRef::Iterator::operator++() {
  if (index_) {
    int64_t idim;
    for (idim = shape_.size() - 1; idim >= 0; --idim) {
      if (++(*index_)[idim] == shape_[idim]) {  // NOLINT
        // Once a dimension is done, just unwind by strides
        (*index_)[idim] = 0;  // NOLINT
        ptr_ -= (shape_[idim] - 1) * strides_[idim] * elsize_;
      } else {
        ptr_ += strides_[idim] * elsize_;
        break;
      }
    }
    // Mark invalid
    if (idim == -1) {
      index_.reset();
      ptr_ = nullptr;
    }
  }
  return *this;
}

MemRef::Iterator MemRef::Iterator::operator++(int) {
  MemRef::Iterator tempIter = *this;
  ++*this;
  return tempIter;
}

MemRef makeConstantArrayRef(const Type& eltype, const Shape& shape) {
  auto buf = std::make_shared<yacl::Buffer>(eltype.size());
  memset(buf->data(), 0, eltype.size());
  return MemRef(buf,                       // buf
                eltype,                    // eltype
                shape,                     // numel
                Strides(shape.size(), 0),  // stride,
                0                          // offset
  );
}

Visibility MemRef::vtype() const { return getVisibilityFromType(eltype()); };

size_t MemRef::chunksCount(size_t max_chunk_size) const {
  size_t total = numel() * elsize();
  size_t num_chunks = (total + max_chunk_size - 1) / max_chunk_size;
  return num_chunks;
}

ValueProto MemRef::toProto(size_t max_chunk_size) const {
  SPU_ENFORCE(max_chunk_size > 0);
  SPU_ENFORCE(vtype() != VIS_INVALID, "{}", *this);

  ValueProto ret;

  auto build_chunk = [&](const void* data, size_t size, size_t num_chunks) {
    if (size == 0) {
      return;
    }
    ret.chunks.reserve(ret.chunks.size() + num_chunks);
    for (size_t i = 0; i < num_chunks; i++) {
      size_t chunk_size = std::min(max_chunk_size, size - i * max_chunk_size);

      size_t offset = i * max_chunk_size;
      ValueChunkProto chunk;
      chunk.set_total_bytes(size);
      chunk.set_chunk_offset(offset);
      if (chunk_size > 0) {
        chunk.set_content(static_cast<const uint8_t*>(data) + offset,
                          chunk_size);
      }
      ret.chunks.emplace_back(std::move(chunk));
    }
  };

  const size_t num_chunks = chunksCount(max_chunk_size);

  if (isCompact()) {
    build_chunk(data(), numel() * elsize(), num_chunks);
  } else {
    // Make a compact clone
    auto copy = clone();
    SPU_ENFORCE(copy.isCompact(), "Must be a compact copy.");
    build_chunk(copy.data(), copy.buf()->size(), num_chunks);
  }

  ret.meta.CopyFrom(toMetaProto());

  return ret;
}

ValueMetaProto MemRef::toMetaProto() const {
  SPU_ENFORCE(vtype() != VIS_INVALID);

  ValueMetaProto proto;
  proto.set_is_complex(isComplex());
  proto.set_visibility(vtype());
  for (const auto& d : shape()) {
    proto.mutable_shape()->add_dims(d);
  }
  proto.set_storage_type(eltype().toString());
  return proto;
}

MemRef MemRef::fromProto(const ValueProto& value) {
  const auto& meta = value.meta;
  const auto eltype = Type::fromString(meta.storage_type());

  // vtype is deduced from storage_type.
  SPU_ENFORCE(meta.visibility() == getVisibilityFromType(eltype),
              "visibility {} does not match storage_type {}", meta.visibility(),
              eltype);

  Shape shape(meta.shape().dims().begin(), meta.shape().dims().end());

  const auto& chunks = value.chunks;
  const size_t total_bytes = chunks.empty() ? 0 : chunks[0].total_bytes();

  std::map<size_t, const ValueChunkProto*> ordered_chunks;
  for (const auto& s : chunks) {
    SPU_ENFORCE(ordered_chunks.insert({s.chunk_offset(), &s}).second,
                "Repeated chunk_offset {} found", s.chunk_offset());
  }

  MemRef data(eltype, shape, meta.is_complex());
  SPU_ENFORCE(static_cast<size_t>(data.buf()->size()) == total_bytes);

  size_t chunk_end_pos = 0;
  for (const auto& [offset, chunk] : ordered_chunks) {
    SPU_ENFORCE(offset == chunk_end_pos,
                "offset {} is not match to last chunk's end pos", offset);
    memcpy(data.data<uint8_t>() + offset, chunk->content().data(),
           chunk->content().size());
    chunk_end_pos += chunk->content().size();
  }

  SPU_ENFORCE(total_bytes == chunk_end_pos);

  return data;
}

std::ostream& operator<<(std::ostream& out, const MemRef& v) {
  if (v.isPrivate()) {
    out << fmt::format("Value<{}x{},s={},o={}>", fmt::join(v.shape(), "x"),
                       v.vtype(), fmt::join(v.strides(), ","),
                       v.eltype().as<Private>()->owner());
  } else {
    out << fmt::format("Value<{}x{},s={}>", fmt::join(v.shape(), "x"),
                       v.vtype(), fmt::join(v.strides(), ","));
  }
  return out;
}

std::ostream& operator<<(std::ostream& out, const std::vector<MemRef>& v) {
  out << fmt::format("{}", fmt::join(v, ","));
  return out;
}

}  // namespace spu
