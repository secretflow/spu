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

#include "libspu/core/array_ref.h"

#include <numeric>

#include "fmt/format.h"
#include "fmt/ostream.h"

#include "libspu/core/parallel_utils.h"

namespace spu {
namespace detail {

void strided_copy(int64_t numel, int64_t elsize, void* dst, int64_t dstride,
                  void const* src, int64_t sstride) {
  const char* src_itr = static_cast<const char*>(src);
  char* dst_itr = static_cast<char*>(dst);

  if (dstride == 1 && sstride == 1) {
    std::memcpy(dst_itr, src_itr, elsize * numel);
  } else {
    dstride *= elsize;
    sstride *= elsize;

    pforeach(0, numel, [&](int64_t idx) {
      std::memcpy(&dst_itr[idx * dstride], &src_itr[idx * sstride], elsize);
    });
  }
}

}  // namespace detail

ArrayRef::ArrayRef(std::shared_ptr<yacl::Buffer> buf, Type eltype,
                   int64_t numel, int64_t stride, int64_t offset)
    : buf_(std::move(buf)),
      eltype_(std::move(eltype)),
      numel_(numel),
      stride_(stride),
      offset_(offset) {
  // sanity check.
  if (numel != 0) {
    const auto elsize = static_cast<int64_t>(eltype_.size());
    const auto bufsize = buf_->size();
    SPU_ENFORCE(offset >= 0 && offset + elsize <= bufsize);
    SPU_ENFORCE(
        (offset + stride * (numel - 1) >= 0) &&
            (offset + stride * (numel - 1) + elsize <= bufsize),
        "sanity failed, eltype={}, offset={}, stride={}, numel={}, buf.size={}",
        eltype_, offset, stride, numel, bufsize);
  }
}

ArrayRef::ArrayRef(const Type& eltype, size_t numel)
    : ArrayRef(std::make_shared<yacl::Buffer>(numel * eltype.size()),
               eltype,  // eltype
               numel,   // numel
               1,       // stride,
               0        // offset
      ) {}

ArrayRef makeConstantArrayRef(const Type& eltype, size_t numel) {
  auto buf = std::make_shared<yacl::Buffer>(eltype.size());
  memset(buf->data(), 0, eltype.size());
  return ArrayRef(buf,     // buf
                  eltype,  // eltype
                  numel,   // numel
                  0,       // stride,
                  0        // offset
  );
}

bool ArrayRef::isCompact() const { return stride_ == 1 || numel_ == 0; }

std::shared_ptr<yacl::Buffer> ArrayRef::getOrCreateCompactBuf() const {
  if (isCompact() && offset_ == 0) {
    return buf();
  }
  return clone().buf();
}

ArrayRef ArrayRef::clone() const {
  ArrayRef res(eltype(), numel());

  detail::strided_copy(numel(), elsize(), res.data(), res.stride(), data(),
                       stride());
  return res;
}

ArrayRef ArrayRef::as(const Type& new_ty, bool force) const {
  if (!force) {
    SPU_ENFORCE(elsize() == new_ty.size(),
                "viewed type={} not equal to origin type={}", new_ty, eltype());
  }

  return ArrayRef(buf(), new_ty, numel(), stride(), offset());
}

ArrayRef ArrayRef::slice(int64_t start, int64_t stop, int64_t stride) const {
  // From
  // https://numpy.org/doc/stable/user/basics.indexing.html#slicing-and-striding
  //
  // The basic slice syntax is i:j:k where i is the starting index, j is the
  // stopping index, and k is the step (). This selects the m elements (in the
  // corresponding dimension) with index values i, i + k, …, i + (m - 1) k
  // where and q and r are the quotient and remainder obtained by dividing j -
  // i by k: j - i = q k + r, so that i + (m - 1) k < j.
  SPU_ENFORCE(start < numel_, "start={}, numel_={}", start, numel_);

  const int64_t q = (stop - start) / stride;
  const int64_t r = (stop - start) % stride;
  const int64_t m = q + static_cast<int64_t>(r != 0);

  const int64_t n_stride = stride_ * stride;
  const int64_t n_offset = offset_ + start * stride_ * elsize();

  return ArrayRef(buf(), eltype_, m, n_stride, n_offset);
}

bool ArrayRef::operator==(const ArrayRef& other) const {
  if (numel() != other.numel() || eltype() != other.eltype()) {
    return false;
  }

  for (int64_t idx = 0; idx < numel(); idx++) {
    const auto* a = &at(idx);
    const auto* b = &other.at(idx);

    if (memcmp(a, b, elsize()) != 0) {
      return false;
    }
  }

  return true;
}

std::ostream& operator<<(std::ostream& out, const ArrayRef& v) {
  out << fmt::format("ArrayRef<{}x{}>", v.numel(), v.eltype().toString());
  return out;
}

}  // namespace spu
