#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "yacl/base/buffer.h"

namespace pybind11 {
namespace detail {

template <>
struct type_caster<yacl::Buffer> {
 public:
  PYBIND11_TYPE_CASTER(yacl::Buffer, const_name("numpy.ndarray[uint8]"));

  bool load(handle src, bool convert) {
    if (!isinstance<array>(src)) {
      return false;
    }
    auto arr = array_t < uint8_t,
         array::c_style | array::forcecast > ::ensure(src);
    if (!arr) {
      return false;
    }
    value = yacl::Buffer(arr.data(), arr.nbytes());
    return true;
  }

  static handle cast(yacl::Buffer&& src, return_value_policy, handle) {
    auto* buf_ptr = new yacl::Buffer(std::move(src));
    capsule cap(buf_ptr,
                [](void* data) { delete static_cast<yacl::Buffer*>(data); });
    return array_t<uint8_t>({buf_ptr->size()}, {sizeof(uint8_t)},
                            buf_ptr->data<uint8_t>(), cap)
        .release();
  }
};

}  // namespace detail
}  // namespace pybind11