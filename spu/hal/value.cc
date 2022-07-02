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

#include "spu/hal/value.h"

#include "fmt/format.h"
#include "fmt/ostream.h"
#include "yasl/base/exception.h"

#include "spu/core/ndarray_ref.h"
#include "spu/core/shape_util.h"

namespace spu::hal {
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
    YASL_THROW("Invalid set new dtype, cur={}, new={}", dtype_, new_dtype);
  }

  dtype_ = new_dtype;
  return *this;
}

void Value::copyElementFrom(const Value& v, absl::Span<const int64_t> input_idx,
                            absl::Span<const int64_t> output_idx) {
  YASL_ENFORCE(v.dtype() == dtype(), "dtype mismatch, from={}, to={}",
               v.dtype(), dtype());
  YASL_ENFORCE(v.storage_type() == storage_type(),
               "storage_type mismatch, from={}, to={}", v.storage_type(),
               storage_type());
  YASL_ENFORCE(v.vtype() == vtype(), "vtype mismatch, from={}, to={}",
               v.vtype(), vtype());

  memcpy(&data_.at(output_idx), &v.data_.at(input_idx), data_.elsize());
}

Value Value::getElementAt(absl::Span<const int64_t> index) const {
  // TODO: use NdArrayRef.slice to implement this function.
  YASL_ENFORCE(dtype() != DT_INVALID);

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
    YASL_ENFORCE(copy.isCompact(), "Must be a compact copy.");
    proto.set_content(copy.data(), copy.buf()->size());
  }
  return proto;
}

Value Value::fromProto(const ValueProto& proto) {
  const auto eltype = Type::fromString(proto.storage_type());

  YASL_ENFORCE(proto.data_type() != DT_INVALID, "invalid data type={}",
               proto.data_type());

  // vtype is deduced from storage_type.
  YASL_ENFORCE(proto.visibility() == getVisibilityFromType(eltype),
               "visibility {} does not match storage_type {}",
               proto.visibility(), eltype);

  std::vector<int64_t> shape(proto.shape().dims().begin(),
                             proto.shape().dims().end());

  NdArrayRef data(eltype, shape);
  YASL_ENFORCE(static_cast<size_t>(data.buf()->size()) ==
               proto.content().size());
  memcpy(data.data(), proto.content().c_str(), data.buf()->size());

  return Value(data, proto.data_type());
}

Value Value::clone() const { return Value(data_.clone(), dtype()); }

std::ostream& operator<<(std::ostream& out, const Value& v) {
  out << fmt::format("Value<{}x{}{}>", fmt::join(v.shape(), "x"), v.vtype(),
                     v.dtype());
  return out;
}

}  // namespace spu::hal
