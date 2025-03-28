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

#pragma once

#include "libspu/core/ndarray_ref.h"
#include "libspu/mpc/utils/gfmp.h"

namespace spu::mpc {

NdArrayRef gfmp_rand(FieldType field, const Shape& shape);
NdArrayRef gfmp_rand(FieldType field, const Shape& shape, uint128_t prg_seed,
                     uint64_t* prg_counter);

NdArrayRef gfmp_zeros(FieldType field, const Shape& shape);

NdArrayRef gfmp_mod(const NdArrayRef& x);
void gfmp_mod_(NdArrayRef& x);

NdArrayRef gfmp_batch_inverse(const NdArrayRef& x);

NdArrayRef gfmp_mul_mod(const NdArrayRef& x, const NdArrayRef& y);
void gfmp_mul_mod_(NdArrayRef& x, const NdArrayRef& y);

NdArrayRef gfmp_div_mod(const NdArrayRef& x, const NdArrayRef& y);
void gfmp_div_mod_(NdArrayRef& x, const NdArrayRef& y);

NdArrayRef gfmp_add_mod(const NdArrayRef& x, const NdArrayRef& y);
void gfmp_add_mod_(NdArrayRef& x, const NdArrayRef& y);

NdArrayRef gfmp_sub_mod(const NdArrayRef& x, const NdArrayRef& y);
void gfmp_sub_mod_(NdArrayRef& x, const NdArrayRef& y);

NdArrayRef gfmp_exp_mod(const NdArrayRef& x, const NdArrayRef& y);
void gfmp_exp_mod_(NdArrayRef& x, const NdArrayRef& y);

NdArrayRef gfmp_mmul_mod(const NdArrayRef& x, const NdArrayRef& y);
NdArrayRef gfmp_arshift_mod(const NdArrayRef& x, const Sizes& bits);

std::vector<NdArrayRef> gfmp_rand_shamir_shares(const NdArrayRef& x,
                                                const NdArrayRef& coeffs,
                                                size_t world_size,
                                                size_t threshold);

std::vector<NdArrayRef> gfmp_rand_shamir_shares(const NdArrayRef& x,
                                                size_t world_size,
                                                size_t threshold);

template <typename T>
NdArrayRef gfmp_reconstruct_shamir_shares(
    absl::Span<const NdArrayRef> shares, size_t world_size, size_t threshold,
    std::vector<T> reconstruction_vector) {
  SPU_ENFORCE(std::all_of(shares.begin(), shares.end(),
                          [&](const NdArrayRef& x) {
                            return x.eltype() == shares[0].eltype() &&
                                   x.shape() == shares[0].shape() &&
                                   x.eltype().isa<GfmpTy>();
                          }),
              "Share shape and type should be the same");
  SPU_ENFORCE_GT(shares.size(), threshold,
                 "Shares size and threshold are not matched");
  SPU_ENFORCE(world_size >= threshold * 2 + 1 && threshold >= 1,
              "invalid party numbers {} or threshold {}", world_size,
              threshold);
  const auto* ty = shares[0].eltype().as<GfmpTy>();
  const auto field = ty->field();
  const auto numel = shares[0].numel();
  NdArrayRef out(makeType<GfmpTy>(field), shares[0].shape());

  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _out(out);
    pforeach(0, numel, [&](int64_t idx) {
      ring2k_t secret = 0;
      // TODO optimize me: the reconstruction vector for a fixed point can be
      // pre-computed
      for (size_t i = 0; i < shares.size(); ++i) {
        NdArrayView<ring2k_t> _share(shares[i]);
        secret = add_mod(
            secret, mul_mod(_share[idx],
                            static_cast<ring2k_t>(reconstruction_vector[i])));
      }
      _out[idx] = secret;
    });
  });
  return out;
}

NdArrayRef gfmp_reconstruct_shamir_shares(absl::Span<const NdArrayRef> shares,
                                          size_t world_size, size_t threshold);

}  // namespace spu::mpc
