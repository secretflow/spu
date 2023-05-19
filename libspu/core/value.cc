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

#include "libspu/core/value.h"

#include <algorithm>
#include <cstdint>

#include "fmt/format.h"
#include "fmt/ostream.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/prelude.h"
#include "libspu/core/shape_util.h"

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
    SPU_THROW("Invalid set new dtype, cur={}, new={}", dtype_, new_dtype);
  }

  dtype_ = new_dtype;
  return *this;
}

ValueProto Value::toProto() const {
  SPU_ENFORCE(dtype_ != DT_INVALID && vtype() != VIS_INVALID);

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
    SPU_ENFORCE(copy.isCompact(), "Must be a compact copy.");
    proto.set_content(copy.data(), copy.buf()->size());
  }
  return proto;
}

ValueMeta Value::toMetaProto() const {
  SPU_ENFORCE(dtype_ != DT_INVALID && vtype() != VIS_INVALID);

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

  SPU_ENFORCE(proto.data_type() != DT_INVALID, "invalid data type={}",
              proto.data_type());

  // vtype is deduced from storage_type.
  SPU_ENFORCE(proto.visibility() == getVisibilityFromType(eltype),
              "visibility {} does not match storage_type {}",
              proto.visibility(), eltype);

  std::vector<int64_t> shape(proto.shape().dims().begin(),
                             proto.shape().dims().end());

  NdArrayRef data(eltype, shape);
  SPU_ENFORCE(static_cast<size_t>(data.buf()->size()) ==
              proto.content().size());
  memcpy(data.data(), proto.content().c_str(), data.buf()->size());

  return Value(data, proto.data_type());
}

Value Value::clone() const { return Value(data_.clone(), dtype()); }

std::ostream& operator<<(std::ostream& out, const Value& v) {
  out << fmt::format("Value<{}x{}{},s={}>", fmt::join(v.shape(), "x"),
                     v.vtype(), v.dtype(), fmt::join(v.strides(), ","));
  return out;
}

std::tuple<ArrayRef, Shape, DataType> UnwrapValue(const Value& val) {
  return std::make_tuple(flatten(val.data()), Shape(val.shape()), val.dtype());
}

Value WrapValue(const ArrayRef& arr, absl::Span<int64_t const> shape,
                DataType dtype) {
  auto ndarr = unflatten(arr, shape);
  return Value(ndarr, dtype);
}

}  // namespace spu
