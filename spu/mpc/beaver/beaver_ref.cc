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

#include "spu/mpc/beaver/beaver_ref.h"

#include "spu/mpc/util/ring_ops.h"

namespace spu::mpc {

Beaver::Triple BeaverRef::Mul(FieldType field, size_t size) {
  return {
      ring_zeros(field, size),
      ring_zeros(field, size),
      ring_zeros(field, size),
  };
}

Beaver::Triple BeaverRef::And(FieldType field, size_t size) {
  return {
      ring_zeros(field, size),
      ring_zeros(field, size),
      ring_zeros(field, size),
  };
}

Beaver::Triple BeaverRef::Dot(FieldType field, size_t M, size_t N, size_t K) {
  return {
      ring_zeros(field, M * K),
      ring_zeros(field, K * N),
      ring_zeros(field, M * N),
  };
}

Beaver::Pair BeaverRef::Trunc(FieldType field, size_t size, size_t bits) {
  return {
      ring_zeros(field, size),
      ring_zeros(field, size),
  };
}

ArrayRef BeaverRef::RandBit(FieldType field, size_t size) {
  return ring_zeros(field, size);
}

}  // namespace spu::mpc
