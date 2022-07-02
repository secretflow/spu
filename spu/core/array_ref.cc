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

#include "spu/core/array_ref.h"

#include <numeric>

#include "fmt/format.h"
#include "fmt/ostream.h"

namespace spu {

ArrayRef::ArrayRef(std::shared_ptr<yasl::Buffer> buf, Type eltype,
                   int64_t numel, int64_t stride, int64_t offset)
    : buf_(std::move(buf)),
      eltype_(std::move(eltype)),
      numel_(numel),
      stride_(stride),
      offset_(offset) {
  // sanity check.
  YASL_ENFORCE(offset + stride * numel <= buf_->size());
}

ArrayRef::ArrayRef(Type eltype, size_t numel)
    : ArrayRef(std::make_shared<yasl::Buffer>(numel * eltype.size()),
               eltype,  // eltype
               numel,   // numel
               1,       // stride,
               0        // offset
      ) {}

bool ArrayRef::isCompact() const { return stride_ == 1; }

std::shared_ptr<yasl::Buffer> ArrayRef::getOrCreateCompactBuf() const {
  if (isCompact()) {
    // iff stride_ != 0, should we return the original buffer?
    return buf();
  }
  return clone().buf();
}

ArrayRef ArrayRef::clone() const {
  ArrayRef res(eltype(), numel());

  for (int64_t idx = 0; idx < numel(); idx++) {
    const auto* frm = &at(idx);
    auto* dst = &res.at(idx);

    std::memcpy(dst, frm, elsize());
  }

  return res;
}

ArrayRef ArrayRef::as(const Type& new_ty, bool force) const {
  if (!force) {
    YASL_ENFORCE(elsize() == new_ty.size(),
                 "viewed type={} not equal to origin type={}", new_ty,
                 eltype());
  }

  return {buf(), new_ty, numel(), stride(), offset()};
}

ArrayRef ArrayRef::slice(int64_t start, int64_t stop, int64_t stride) {
  // From
  // https://numpy.org/doc/stable/user/basics.indexing.html#slicing-and-striding
  //
  // The basic slice syntax is i:j:k where i is the starting index, j is the
  // stopping index, and k is the step (). This selects the m elements (in the
  // corresponding dimension) with index values i, i + k, …, i + (m - 1) k
  // where and q and r are the quotient and remainder obtained by dividing j -
  // i by k: j - i = q k + r, so that i + (m - 1) k < j.
  YASL_ENFORCE(start < numel_, "start={}, numel_={}", start, numel_);

  const int64_t q = (stop - start) / stride;
  const int64_t r = (stop - start) % stride;
  const int64_t m = q + static_cast<int64_t>(r != 0);

  const int64_t n_stride = stride_ * stride;
  const int64_t n_offset = offset_ + start * stride_ * elsize();

  return {buf(), eltype_, m, n_stride, n_offset};
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
