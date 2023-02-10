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
#include "libspu/core/shape_util.h"

namespace spu {
namespace detail {

size_t calcFlattenOffset(absl::Span<const int64_t> indices,
                         absl::Span<const int64_t> shape,
                         absl::Span<const int64_t> strides) {
  if (!shape.empty() && strides.empty()) {
    return calcFlattenOffset(indices, shape, makeCompactStrides(shape));
  }

  int64_t offset = 0;
  for (int64_t idx = indices.size() - 1; idx >= 0; --idx) {
    offset += indices[idx] * strides[idx];
  }
  return offset;
}

}  // namespace detail

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
    return {buf(), new_ty, shape(), strides(), offset()};
  }
  // Force view, we need to adjust strides
  auto distance = ((strides().empty() ? 1 : strides().back()) * elsize());
  SPU_ENFORCE(distance % new_ty.size() == 0);

  std::vector<int64_t> new_strides = strides();
  std::transform(new_strides.begin(), new_strides.end(), new_strides.begin(),
                 [&](int64_t s) { return (elsize() * s) / new_ty.size(); });

  return {buf(), new_ty, shape(), new_strides, offset()};
}

int64_t NdArrayRef::numel() const { return calcNumel(shape()); }

bool NdArrayRef::isCompact() const {
  return makeCompactStrides(shape()) == strides();
}

NdArrayRef NdArrayRef::clone() const {
  NdArrayRef res(eltype(), shape());

  auto elsize = res.elsize();

  // FIXME(xiaochen): Once we have a proper iterator, just replace the following
  // helper function with next()
  auto next_ = [&](std::vector<int64_t>& coord, const std::byte*& frm_ptr) {
    for (int64_t idim = shape().size() - 1; idim >= 0; --idim) {
      if (++coord[idim] == shape()[idim]) {
        // Once a dimension is done, just unwind by strides
        coord[idim] = 0;
        frm_ptr -= (shape()[idim] - 1) * strides()[idim] * elsize;
      } else {
        frm_ptr += strides()[idim] * elsize;
        break;
      }
    }
  };

  auto* ret_ptr = static_cast<std::byte*>(res.data());

  std::vector<int64_t> indices(shape().size(), 0);
  const auto* frm_ptr = &at(indices);

  for (int64_t idx = 0; idx < numel(); ++idx) {
    std::memcpy(ret_ptr + idx * elsize, frm_ptr, elsize);
    next_(indices, frm_ptr);
  }

  return res;
}

NdArrayRef unflatten(const ArrayRef& arr, absl::Span<const int64_t> shape) {
  SPU_ENFORCE(arr.numel() == calcNumel(shape),
              "unflatten numel mismatch, expected={}, got={}", calcNumel(shape),
              arr.numel());

  if (arr.stride() == 0) {
    return {arr.buf(), arr.eltype(), shape,
            std::vector<int64_t>(shape.size(), 0), arr.offset()};
  }

  // FIXME: due to the current implementation,
  SPU_ENFORCE(arr.isCompact(), "FIXME: impl assume array is flatten, got {}",
              arr);

  auto strides = makeCompactStrides(shape);
  return {arr.buf(), arr.eltype(), shape, std::move(strides), arr.offset()};
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

  // SPDLOG_INFO("slow flatten..., in strides = {}, shape = {}",
  // fmt::join(ndarr.strides(), "x"), fmt::join(ndarr.shape(), "x"));

  // create a compact clone, it's save here since underline layer will never
  // modify inplace.
  auto compact = ndarr.clone();
  return {compact.buf(), ndarr.eltype(), ndarr.numel(), 1, compact.offset()};
}

std::ostream& operator<<(std::ostream& out, const NdArrayRef& v) {
  out << fmt::format("NdArrayRef<{}x{}>", fmt::join(v.shape(), "x"),
                     v.eltype().toString());
  return out;
}

}  // namespace spu
