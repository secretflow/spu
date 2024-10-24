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
                             const ModulusSwitchHelper &msh) {
  SPU_ENFORCE(context.parameters_set());
  auto pid0 = context.first_parms_id();
  auto pid1 = msh.parms_id();
  SPU_ENFORCE_EQ(0, std::memcmp(&pid0, &pid1, sizeof(seal::parms_id_type)),
                 fmt::format("parameter set mismatch"));
  msh_ = std::make_shared<ModulusSwitchHelper>(msh);
  poly_deg_ = context.first_context_data()->parms().poly_modulus_degree();
}

void VectorEncoder::Forward(const NdArrayRef &vec, RLWEPt *out,
                            bool scale_delta) const {
  // Place the vector elements as polynomial coefficients forwardly.
  // a0, a1, ..., an -> \sum_i ai*X^i
  yacl::CheckNotNull(out);

  size_t num_coeffs = vec.numel();
  size_t num_modulus = msh_->coeff_modulus_size();
  SPU_ENFORCE(vec.shape().size() == 1, "need 1D array");
  SPU_ENFORCE_GT(num_coeffs, 0UL);
  SPU_ENFORCE(num_coeffs <= poly_deg_);

  out->parms_id() = seal::parms_id_zero;
  out->resize(seal::util::mul_safe(poly_deg_, num_modulus));

  uint64_t *dst = out->data();
  for (size_t mod_idx = 0; mod_idx < num_modulus; ++mod_idx) {
    std::fill_n(dst, poly_deg_, 0);
    absl::Span<uint64_t> dst_wrap(dst, num_coeffs);

    if (scale_delta) {
      msh_->ModulusUpAt(vec, mod_idx, dst_wrap);
    } else {
      msh_->CenteralizeAt(vec, mod_idx, dst_wrap);
    }
    dst += poly_deg_;
  }

  out->parms_id() = msh_->parms_id();
  out->scale() = 1.;
}

void VectorEncoder::Backward(const NdArrayRef &vec, RLWEPt *out,
                             bool scale_delta) const {
  // Place the vector elements as polynomial coefficients in backward.
  // a0, a1, ..., an -> a0 - \sum_{i>0} ai*X^{N-i}
  // where N defines the base ring X^N + 1.
  yacl::CheckNotNull(out);
  SPU_ENFORCE(vec.shape().size() == 1, "need 1D array");

  size_t num_coeffs = vec.numel();
  size_t num_modulus = msh_->coeff_modulus_size();
  SPU_ENFORCE_GT(num_coeffs, 0UL);
  SPU_ENFORCE(num_coeffs <= poly_deg_);

  const Type &eltype = vec.eltype();
  SPU_ENFORCE(eltype.isa<RingTy>(), "must be ring_type, got={}", eltype);
  out->parms_id() = seal::parms_id_zero;
  out->resize(seal::util::mul_safe(poly_deg_, num_modulus));

  const auto field = eltype.as<Ring2k>()->field();

  DISPATCH_ALL_FIELDS(field, "Backward", [&]() {
    auto tmp_buff = ring_zeros(field, {(int64_t)poly_deg_});
    auto xvec = NdArrayView<const ring2k_t>(vec);
    auto xtmp = NdArrayView<ring2k_t>(tmp_buff);

    xtmp[0] = xvec[0];
    // reverse and sign flip
    for (size_t i = 1; i < num_coeffs; ++i) {
      xtmp[num_coeffs - 1 - i] = -xvec[i];
    }

    uint64_t *dst = out->data();
    for (size_t mod_idx = 0; mod_idx < num_modulus; ++mod_idx) {
      std::fill_n(dst, poly_deg_, 0);
      absl::Span<uint64_t> dst_wrap(dst, poly_deg_);

      if (scale_delta) {
        msh_->ModulusUpAt(tmp_buff, mod_idx, dst_wrap);
      } else {
        msh_->CenteralizeAt(tmp_buff, mod_idx, dst_wrap);
      }
      dst += poly_deg_;
    }

    // clean up sensitive data
    seal::util::seal_memzero(&tmp_buff.at<ring2k_t>(0),
                             sizeof(ring2k_t) * poly_deg_);
  });

  out->parms_id() = msh_->parms_id();
  out->scale() = 1.;
}

}  // namespace spu::mpc::cheetah
