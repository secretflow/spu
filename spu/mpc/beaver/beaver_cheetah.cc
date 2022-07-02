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

#include "spu/mpc/beaver/beaver_cheetah.h"

#include "yasl/link/link.h"

#include "spu/mpc/util/ring_ops.h"

namespace spu::mpc {

BeaverCheetah::BeaverCheetah(std::shared_ptr<yasl::link::Context> lctx) {
  cheetah_he_primitives_ = std::make_shared<BeaverHE>(lctx);
}

void BeaverCheetah::set_primitives(
    std::shared_ptr<spu::CheetahPrimitives> cheetah_primitives) {
  yasl::CheckNotNull(cheetah_primitives.get());
  cheetah_ot_primitives_ = cheetah_primitives;
}

Beaver::Triple BeaverCheetah::Mul(FieldType field, size_t size) {
  yasl::CheckNotNull(cheetah_he_primitives_.get());
  return cheetah_he_primitives_->Mul(field, size);
}

Beaver::Triple BeaverCheetah::Dot(FieldType field, size_t M, size_t N,
                                  size_t K) {
  yasl::CheckNotNull(cheetah_he_primitives_.get());
  return cheetah_he_primitives_->Dot(field, M, N, K);
}

Beaver::Triple BeaverCheetah::And(FieldType field, size_t size) {
  yasl::CheckNotNull(cheetah_ot_primitives_.get());

  ArrayRef a(makeType<RingTy>(field), size);
  ArrayRef b(makeType<RingTy>(field), size);
  ArrayRef c(makeType<RingTy>(field), size);

  cheetah_ot_primitives_->nonlinear()->beaver_triple(
      (uint8_t*)a.data(), (uint8_t*)b.data(), (uint8_t*)c.data(),
      size * a.elsize() * 8, true);

  return {a, b, c};
}

Beaver::Pair BeaverCheetah::Trunc(FieldType field, size_t size, size_t bits) {
  YASL_THROW_LOGIC_ERROR("this method should not be called");
}

ArrayRef BeaverCheetah::RandBit(FieldType field, size_t size) {
  YASL_THROW_LOGIC_ERROR("this method should not be called");
}

}  // namespace spu::mpc
