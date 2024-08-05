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
  SPU_ENFORCE(new_dtype != DT_INVALID);

  if (new_dtype == dtype_) {
    return *this;
  }

  if (!force && dtype_ != DT_INVALID) {
    SPU_THROW("Invalid set new dtype, cur={}, new={}", dtype_, new_dtype);
  }

  dtype_ = new_dtype;
  return *this;
}

size_t Value::chunksCount(size_t max_chunk_size) const {
  size_t total = numel() * data_.elsize();
  size_t num_chunks = (total + max_chunk_size - 1) / max_chunk_size;
  return num_chunks;
}

ValueProto Value::toProto(size_t max_chunk_size) const {
  SPU_ENFORCE(max_chunk_size > 0);
  SPU_ENFORCE(dtype_ != DT_INVALID && vtype() != VIS_INVALID);

  ValueProto ret;

  auto build_chunk = [&](const void* data, size_t size, size_t num_chunks) {
    if (size == 0) {
      return;
    }
    ret.chunks.reserve(num_chunks);
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

  if (data_.isCompact()) {
    build_chunk(data_.data(), numel() * data_.elsize(), num_chunks);
  } else {
    // Make a compact clone
    auto copy = data_.clone();
    SPU_ENFORCE(copy.isCompact(), "Must be a compact copy.");
    build_chunk(copy.data(), copy.buf()->size(), num_chunks);
  }

  ret.meta.CopyFrom(toMetaProto());

  return ret;
}

ValueMetaProto Value::toMetaProto() const {
  SPU_ENFORCE(dtype_ != DT_INVALID && vtype() != VIS_INVALID);

  ValueMetaProto proto;
  proto.set_data_type(dtype_);
  proto.set_visibility(vtype());
  for (const auto& d : shape()) {
    proto.mutable_shape()->add_dims(d);
  }
  proto.set_storage_type(data_.eltype().toString());
  return proto;
}

Value Value::fromProto(const ValueProto& value) {
  const auto& meta = value.meta;
  const auto eltype = Type::fromString(meta.storage_type());

  SPU_ENFORCE(meta.data_type() != DT_INVALID, "invalid data type={}",
              meta.data_type());

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

  NdArrayRef data(eltype, shape);
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

  return Value(data, meta.data_type());
}

Value Value::clone() const { return Value(data_.clone(), dtype()); }

std::ostream& operator<<(std::ostream& out, const Value& v) {
  out << fmt::format("Value<{}x{}{},s={}>", fmt::join(v.shape(), "x"),
                     v.vtype(), v.dtype(), fmt::join(v.strides(), ","));
  return out;
}

}  // namespace spu
