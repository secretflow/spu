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

#include "absl/types/span.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/type.h"
#include "libspu/core/type_util.h"
#include "libspu/core/vectorize.h"

namespace spu {

class Value final {
  NdArrayRef data_;
  DataType dtype_ = DT_INVALID;

 public:
  Value() = default;
  explicit Value(NdArrayRef data, DataType dtype);

  /// Forward ndarray methods.
  inline int64_t numel() const { return data_.numel(); }
  inline size_t elsize() const { return data_.elsize(); }
  std::vector<int64_t> const& strides() const { return data_.strides(); }
  std::vector<int64_t> const& shape() const { return data_.shape(); }

  // Get the concrete storage type.
  const Type& storage_type() const { return data_.eltype(); }
  Type& storage_type() { return data_.eltype(); }

  // Access the underline storage data.
  const NdArrayRef& data() const { return data_; }
  NdArrayRef& data() { return data_; }

  // Get vtype, is readonly and decided by the underline secure compute engine.
  Visibility vtype() const;
  bool isPublic() const { return vtype() == VIS_PUBLIC; }
  bool isSecret() const { return vtype() == VIS_SECRET; }

  // Get dtype.
  DataType dtype() const { return dtype_; }
  bool isInt() const { return isInteger(dtype()); }
  bool isFxp() const { return isFixedPoint(dtype()); }

  // Set dtype.
  //
  // By default, we can only set dtype when it's not set yet(=DT_INVALID),
  // unless we forcely do it.
  Value& setDtype(DataType new_dtype, bool force = false);

  // Serialize to protobuf.
  ValueProto toProto() const;
  ValueMeta toMetaProto() const;

  // Deserialize from protobuf.
  static Value fromProto(const ValueProto& proto);

  Value clone() const;
};

// Helper function to legacy kernels. TODO: drop this wraps.
std::tuple<ArrayRef, Shape, DataType> UnwrapValue(const Value& val);
Value WrapValue(const ArrayRef& arr, absl::Span<int64_t const> shape,
                DataType dtype = DT_INVALID);

template <>
struct SimdTrait<Value> {
  using Shape = std::vector<int64_t>;  // TODO: use a formal shape definition.
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
      const std::vector<int64_t> start_index(first->shape().size(), 0);
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
      total_num += calcNumel(shape);
    }

    SPU_ENFORCE(v.numel() == total_num, "split number mismatch {} != {}",
                v.numel(), total_num);

    int64_t offset = 0;
    for (const auto& shape : pi) {
      auto arr = NdArrayRef(v.data().buf(), v.storage_type(), shape,
                            makeCompactStrides(shape), offset);
      *result++ = Value(arr, v.dtype());
      offset += calcNumel(shape) * v.elsize();
    }

    return result;
  }
};

std::ostream& operator<<(std::ostream& out, const Value& v);

}  // namespace spu
