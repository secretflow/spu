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

#include <mutex>

#include "libspu/core/array_ref.h"
#include "libspu/mpc/cheetah/rlwe/types.h"

namespace seal {
class Modulus;
}

namespace spu::mpc::cheetah {

using Shape2D = std::array<int64_t, 2>;
using Shape3D = std::array<int64_t, 3>;
using Shape4D = std::array<int64_t, 4>;

template <typename T>
inline T CeilDiv(T a, T b) {
  SPU_ENFORCE(b > 0);
  return (a + b - 1) / b;
}

struct EnableCPRNG {
  explicit EnableCPRNG();

  // Uniform random on prime field
  void UniformPrime(const seal::Modulus& prime, absl::Span<uint64_t> dst);

  // Uniform random poly from Rq where Rq is defined by pid
  // If pid is none set to context.first_parms_id().
  void UniformPoly(const seal::SEALContext& context, RLWEPt* poly,
                   seal::parms_id_type pid = seal::parms_id_zero);

  // Uniform random on ring 2^k
  ArrayRef CPRNG(FieldType field, size_t size);

 protected:
  mutable std::mutex counter_lock_;
  uint128_t seed_;
  uint64_t prng_counter_;
};

ArrayRef ring_conv2d(const ArrayRef& tensor, const ArrayRef& filter,
                     int64_t num_tensors, Shape3D tensor_shape,
                     int64_t num_filters, Shape3D filter_shape,
                     Shape2D window_strides);

}  // namespace spu::mpc::cheetah
