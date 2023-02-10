// Copyright 2022 Ant Group Co., Ltd.
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

#include "libspu/mpc/cheetah/arith/vector_encoder.h"

#include "libspu/core/prelude.h"
#include "libspu/core/type_util.h"
#include "libspu/core/xt_helper.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {

VectorEncoder::VectorEncoder(const seal::SEALContext &context,
                             const ModulusSwitchHelper &ms_helper) {
  SPU_ENFORCE(context.parameters_set());
  auto pid0 = context.first_parms_id();
  auto pid1 = ms_helper.parms_id();
  SPU_ENFORCE_EQ(0, std::memcmp(&pid0, &pid1, sizeof(seal::parms_id_type)),
                 fmt::format("parameter set mismatch"));
  ms_helper_ = std::make_shared<ModulusSwitchHelper>(ms_helper);
  poly_deg_ = context.first_context_data()->parms().poly_modulus_degree();
}

void VectorEncoder::Forward(const ArrayRef &vec, RLWEPt *out,
                            bool scale_delta) const {
  // Place the vector elements as polynomial coefficients forwardly.
  // a0, a1, ..., an -> \sum_i ai*X^i
  yacl::CheckNotNull(out);

  size_t num_coeffs = vec.numel();
  size_t num_modulus = ms_helper_->coeff_modulus_size();
  SPU_ENFORCE_GT(num_coeffs, 0UL);
  SPU_ENFORCE(num_coeffs <= poly_deg_);

  out->parms_id() = seal::parms_id_zero;
  out->resize(seal::util::mul_safe(poly_deg_, num_modulus));

  uint64_t *dst = out->data();
  for (size_t mod_idx = 0; mod_idx < num_modulus; ++mod_idx) {
    std::fill_n(dst, poly_deg_, 0);
    absl::Span<uint64_t> dst_wrap(dst, num_coeffs);
    if (scale_delta) {
      ms_helper_->ModulusUpAt(vec, mod_idx, dst_wrap);
    } else {
      ms_helper_->CenteralizeAt(vec, mod_idx, dst_wrap);
    }
    dst += poly_deg_;
  }

  out->parms_id() = ms_helper_->parms_id();
  out->scale() = 1.;
}

void VectorEncoder::Backward(const ArrayRef &vec, RLWEPt *out,
                             bool scale_delta) const {
  // Place the vector elements as polynomial coefficients in backward.
  // a0, a1, ..., an -> a0 - \sum_{i>0} ai*X^{N-i}
  // where N defines the base ring X^N + 1.
  yacl::CheckNotNull(out);

  size_t num_coeffs = vec.numel();
  size_t num_modulus = ms_helper_->coeff_modulus_size();
  SPU_ENFORCE_GT(num_coeffs, 0UL);
  SPU_ENFORCE(num_coeffs <= poly_deg_);

  const Type &eltype = vec.eltype();
  SPU_ENFORCE(eltype.isa<RingTy>(), "must be ring_type, got={}", eltype);
  out->parms_id() = seal::parms_id_zero;
  out->resize(seal::util::mul_safe(poly_deg_, num_modulus));

  const auto field = eltype.as<Ring2k>()->field();

  DISPATCH_ALL_FIELDS(field, "Backward", [&]() {
    ArrayRef tmp_buff = ring_zeros(field, poly_deg_);
    auto xvec = xt_adapt<ring2k_t>(vec);
    auto xtmp = xt_mutable_adapt<ring2k_t>(tmp_buff);

    xtmp[0] = xvec[0];
    // reverse and sign flip
    std::transform(xvec.data() + 1, xvec.data() + num_coeffs,
                   std::reverse_iterator<ring2k_t *>(xtmp.data() + poly_deg_),
                   [](ring2k_t x) { return -x; });

    uint64_t *dst = out->data();
    for (size_t mod_idx = 0; mod_idx < num_modulus; ++mod_idx) {
      std::fill_n(dst, poly_deg_, 0);
      absl::Span<uint64_t> dst_wrap(dst, poly_deg_);

      if (scale_delta) {
        ms_helper_->ModulusUpAt(tmp_buff, mod_idx, dst_wrap);
      } else {
        ms_helper_->CenteralizeAt(tmp_buff, mod_idx, dst_wrap);
      }
      dst += poly_deg_;
    }

    // clean up sensitive data
    seal::util::seal_memzero(xtmp.data(), sizeof(ring2k_t) * poly_deg_);
  });

  out->parms_id() = ms_helper_->parms_id();
  out->scale() = 1.;
}

}  // namespace spu::mpc::cheetah
