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

#pragma once

#include <memory>

#include "fmt/ostream.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/shape.h"
#include "libspu/core/type.h"
#include "libspu/core/type_util.h"
#include "libspu/core/vectorize.h"

namespace spu {

// In order to prevent a single protobuf from being larger than 2gb, a spu
// runtime value is represented by multiple chunked protobuf + meta, and
// std::vector is used to organize multiple chunks instead of repeated in
// protobuf.
struct ValueProto {
  ValueMetaProto meta;
  std::vector<ValueChunkProto> chunks;
};

class Value final {
  NdArrayRef data_;
  std::optional<NdArrayRef> imag_;
  DataType dtype_ = DT_INVALID;

 public:
  Value() = default;
  explicit Value(NdArrayRef data, DataType dtype);
  explicit Value(NdArrayRef real, NdArrayRef imag, DataType dtype);

  /// Forward ndarray methods.
  inline int64_t numel() const { return data_.numel(); }
  inline size_t elsize() const { return data_.elsize(); }
  Strides const& strides() const { return data_.strides(); }
  Shape const& shape() const { return data_.shape(); }

  // Get the concrete storage type.
  const Type& storage_type() const { return data_.eltype(); }
  Type& storage_type() { return data_.eltype(); }

  // Access the underline storage data.
  const NdArrayRef& data() const { return data_; }
  NdArrayRef& data() { return data_; }

  const std::optional<NdArrayRef>& imag() const { return imag_; }
  std::optional<NdArrayRef>& imag() { return imag_; }

  // Get vtype, is readonly and decided by the underline secure compute engine.
  Visibility vtype() const;
  bool isPublic() const { return vtype() == VIS_PUBLIC; }
  bool isSecret() const { return vtype() == VIS_SECRET; }
  bool isPrivate() const { return vtype() == VIS_PRIVATE; }

  // Get dtype.
  DataType dtype() const { return dtype_; }
  bool isInt() const { return isInteger(dtype()); }
  bool isFxp() const { return isFixedPoint(dtype()); }
  bool isComplex() const { return imag_.has_value(); }

  // Set dtype.
  //
  // By default, we can only set dtype when it's not set yet(=DT_INVALID),
  // unless we forcely do it.
  Value& setDtype(DataType new_dtype, bool force = false);

  // Serialize to protobuf.
  ValueProto toProto(size_t max_chunk_size) const;
  size_t chunksCount(size_t max_chunk_size) const;
  ValueMetaProto toMetaProto() const;

  // Deserialize from protobuf.
  static Value fromProto(const ValueProto& value);

  Value clone() const;
};

template <>
struct SimdTrait<Value> {
  using PackInfo = std::vector<Shape>;

  template <typename InputIt>
  static Value pack(InputIt first, InputIt last, PackInfo& pi) {
    SPU_ENFORCE(first != last);

    int64_t total_numel = 0;
    const Type ty = first->storage_type();
    const auto dtype = first->dtype();
    for (auto itr = first; itr != last; ++itr) {
      SPU_ENFORCE(itr->storage_type() == ty, "type mismatch {} != {}",
                  itr->storage_type(), ty);
      SPU_ENFORCE(itr->dtype() == dtype, "dtype mismatch {} != {}",
                  itr->dtype(), dtype);
      total_numel += itr->numel();
    }
    NdArrayRef result(ty, {total_numel});
    int64_t offset = 0;
    for (; first != last; ++first) {
      NdArrayRef slice(result.buf(), ty, first->shape(),
                       makeCompactStrides(first->shape()), offset);
      const Index start_index(first->shape().size(), 0);
      slice.copy_slice(first->data(), start_index, start_index, first->numel());
      pi.push_back(first->shape());
      offset += first->numel() * ty.size();
    }
    return Value(result, dtype);
  }

  template <typename OutputIt>
  static OutputIt unpack(const Value& v, OutputIt result, const PackInfo& pi) {
    int64_t total_num = 0;
    for (const auto& shape : pi) {
      total_num += shape.numel();
    }

    SPU_ENFORCE(v.numel() == total_num, "split number mismatch {} != {}",
                v.numel(), total_num);

    int64_t offset = 0;
    for (const auto& shape : pi) {
      auto arr = NdArrayRef(v.data().buf(), v.storage_type(), shape,
                            makeCompactStrides(shape), offset);
      *result++ = Value(arr, v.dtype());
      offset += shape.numel() * v.elsize();
    }

    return result;
  }
};

std::ostream& operator<<(std::ostream& out, const Value& v);
std::ostream& operator<<(std::ostream& out, const std::vector<Value>& v);

inline auto format_as(const Value& v) { return fmt::streamed(v); }

}  // namespace spu
