// Copyright 2025 Ant Group Co., Ltd.
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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "yacl/base/buffer.h"
#include "yacl/base/byte_container_view.h"

namespace pybind11 {
namespace detail {

template <>
struct type_caster<yacl::Buffer> {
 public:
  PYBIND11_TYPE_CASTER(yacl::Buffer, const_name("bytes"));

  bool load(handle src, bool convert) {
    // Try to load from bytes
    if (isinstance<bytes>(src)) {
      std::string_view s = src.cast<std::string_view>();
      value = yacl::Buffer(s.data(), s.size());
      return true;
    }

    // If conversion is allowed, also support numpy array conversion
    if (convert && isinstance<array>(src)) {
      auto arr = array_t < uint8_t,
           array::c_style | array::forcecast > ::ensure(src);
      if (!arr) {
        return false;
      }
      value = yacl::Buffer(arr.data(), arr.nbytes());
      return true;
    }

    return false;
  }

  static handle cast(yacl::Buffer&& src, return_value_policy, handle) {
    // Create Python bytes object directly from buffer data
    return bytes(static_cast<const char*>(src.data()), src.size()).release();
  }
};

template <>
struct type_caster<yacl::ByteContainerView> {
 public:
  PYBIND11_TYPE_CASTER(yacl::ByteContainerView, const_name("bytes"));

  bool load(handle src, bool convert) {
    // Try to load from bytes
    if (isinstance<bytes>(src)) {
      std::string_view s = src.cast<std::string_view>();
      value = yacl::ByteContainerView(s.data(), s.size());
      return true;
    }

    // If conversion is allowed, also support numpy array conversion
    if (convert && isinstance<array>(src)) {
      auto arr = array_t < uint8_t,
           array::c_style | array::forcecast > ::ensure(src);
      if (!arr) {
        return false;
      }
      value = yacl::ByteContainerView(arr.data(), arr.nbytes());
      return true;
    }

    return false;
  }

  static handle cast(yacl::ByteContainerView&& src, return_value_policy,
                     handle) {
    // Create Python bytes object directly from ByteContainerView data
    return bytes(reinterpret_cast<const char*>(src.data()), src.size())
        .release();
  }

  static handle cast(const yacl::ByteContainerView& src, return_value_policy,
                     handle) {
    // Create Python bytes object directly from ByteContainerView data
    return bytes(reinterpret_cast<const char*>(src.data()), src.size())
        .release();
  }
};

}  // namespace detail
}  // namespace pybind11
