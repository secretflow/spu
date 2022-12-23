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

#include "spu/kernel/value.h"

#include <algorithm>
#include <cstdint>

#include "fmt/format.h"
#include "fmt/ostream.h"
#include "yacl/base/exception.h"

#include "spu/core/ndarray_ref.h"
#include "spu/core/shape_util.h"

namespace spu {
namespace {

Visibility getVisibilityFromType(const Type& ty) {
  if (ty.isa<Secret>()) {
    return VIS_SECRET;
  } else if (ty.isa<Public>()) {
    return VIS_PUBLIC;
  } else {
    return VIS_INVALID;
  }
}

}  // namespace

Value::Value(NdArrayRef data, DataType dtype)
    : data_(std::move(data)), dtype_(dtype) {}

Visibility Value::vtype() const {
  return getVisibilityFromType(storage_type());
};

Value& Value::setDtype(DataType new_dtype, bool force) {
  if (new_dtype == dtype_) {
    return *this;
  }

  if (!force && dtype_ != DT_INVALID) {
    YACL_THROW("Invalid set new dtype, cur={}, new={}", dtype_, new_dtype);
  }

  dtype_ = new_dtype;
  return *this;
}

// #define SANITY_ELEWRITE

void Value::copyElementFrom(const Value& v, absl::Span<const int64_t> input_idx,
                            absl::Span<const int64_t> output_idx,
                            int64_t elsize) {
#ifdef SANITY_ELEWRITE
  for (size_t idx = 0; idx < shape().size(); ++idx) {
    if (shape()[idx] != 1) {
      YACL_ENFORCE(strides()[idx] != 0,
                   "Copy into a broadcast value is not safe");
    }
  }
#endif
  memcpy(&data_.at(output_idx), &v.data_.at(input_idx),
         elsize == -1 ? data_.elsize() : elsize);
}

Value Value::getElementAt(absl::Span<const int64_t> index) const {
  // TODO: use NdArrayRef.slice to implement this function.
  YACL_ENFORCE(dtype() != DT_INVALID);

  std::vector<int64_t> start_index(index.size(), 0);
  const int64_t new_offset =
      &data_.at(index) - &data_.at(start_index) + data_.offset();

  NdArrayRef data = {data_.buf(),
                     data_.eltype(),
                     {},  // shape, empty as scalar
                     {},  // stride
                     new_offset};
  return Value(data, dtype_);
}

Value Value::getElementAt(int64_t idx) const {
  return getElementAt(unflattenIndex(idx, shape()));
}

ValueProto Value::toProto() const {
  ValueProto proto;
  proto.set_data_type(dtype_);
  proto.set_visibility(vtype());
  proto.set_storage_type(data_.eltype().toString());
  for (const auto& d : shape()) {
    proto.mutable_shape()->add_dims(d);
  }
  if (data_.isCompact()) {
    proto.set_content(data_.data(), numel() * data_.elsize());
  } else {
    // Make a compact clone
    auto copy = data_.clone();
    YACL_ENFORCE(copy.isCompact(), "Must be a compact copy.");
    proto.set_content(copy.data(), copy.buf()->size());
  }
  return proto;
}

ValueMeta Value::toMetaProto() const {
  ValueMeta proto;
  proto.set_data_type(dtype_);
  proto.set_visibility(vtype());
  for (const auto& d : shape()) {
    proto.mutable_shape()->add_dims(d);
  }
  return proto;
}

Value Value::fromProto(const ValueProto& proto) {
  const auto eltype = Type::fromString(proto.storage_type());

  YACL_ENFORCE(proto.data_type() != DT_INVALID, "invalid data type={}",
               proto.data_type());

  // vtype is deduced from storage_type.
  YACL_ENFORCE(proto.visibility() == getVisibilityFromType(eltype),
               "visibility {} does not match storage_type {}",
               proto.visibility(), eltype);

  std::vector<int64_t> shape(proto.shape().dims().begin(),
                             proto.shape().dims().end());

  NdArrayRef data(eltype, shape);
  YACL_ENFORCE(static_cast<size_t>(data.buf()->size()) ==
               proto.content().size());
  memcpy(data.data(), proto.content().c_str(), data.buf()->size());

  return Value(data, proto.data_type());
}

Value Value::clone() const { return Value(data_.clone(), dtype()); }

std::ostream& operator<<(std::ostream& out, const Value& v) {
  out << fmt::format("Value<{}x{}{},s={}>", fmt::join(v.shape(), "x"),
                     v.vtype(), v.dtype(), fmt::join(v.strides(), "x"));
  return out;
}

}  // namespace spu
