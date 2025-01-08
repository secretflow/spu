// Copyright 2024 Ant Group Co., Ltd.
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

#include "libspu/mpc/shamir/value.h"

#include "libspu/core/prelude.h"
#include "libspu/mpc/shamir/type.h"

namespace spu::mpc::shamir {

NdArrayRef getBitShare(const NdArrayRef& in, size_t bit_idx) {
  SPU_ENFORCE(in.eltype().isa<BShare>());
  auto nbits = in.eltype().as<BShrTy>()->nbits();
  auto field = in.eltype().as<BShrTy>()->field();
  SPU_ENFORCE_GT(nbits, bit_idx);

  auto new_strides = in.strides();
  std::transform(new_strides.cbegin(), new_strides.cend(), new_strides.begin(),
                 [nbits](int64_t s) { return nbits * s; });
  const auto ty = makeType<AShrTy>(field);
  return NdArrayRef(in.buf(), ty, in.shape(), new_strides,
                    in.offset() + bit_idx * static_cast<int64_t>(ty.size()));
}

}  // namespace spu::mpc::shamir
