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

#include "libspu/mpc/cheetah/arith/common.h"

#include "yacl/crypto/utils/rand.h"

#include "libspu/mpc/cheetah/rlwe/utils.h"
#include "libspu/mpc/utils/ring_ops.h"
namespace spu::mpc::cheetah {

EnableCPRNG::EnableCPRNG()
    : seed_(yacl::crypto::RandSeed(/*drbg*/ true)), prng_counter_(0) {}

// Uniform random on prime field
void EnableCPRNG::UniformPrime(const seal::Modulus &prime,
                               absl::Span<uint64_t> dst) {
  SPU_ENFORCE(dst.size() > 0);
  constexpr auto max_random = static_cast<uint64_t>(0xFFFFFFFFFFFFFFFFULL);

  // Sample from [0, n*p) such that n*p ~ 2^64
  auto max_multiple =
      max_random - BarrettReduce<uint64_t>(max_random, prime) - 1;

  auto r = CPRNG(FieldType::FM64, dst.size());
  ArrayView<const uint64_t> xr(r);
  pforeach(0, dst.size(), [&](int64_t i) { dst[i] = xr[i]; });
  // std::copy_n(xr.data(), xr.size(), dst.data());
  std::transform(dst.data(), dst.data() + dst.size(), dst.data(),
                 [&](uint64_t u) {
                   while (u >= max_multiple) {
                     // barely hit in
                     u = CPRNG(FieldType::FM64, 1).at<uint64_t>(0);
                   }
                   return BarrettReduce<uint64_t>(u, prime);
                 });
}

void EnableCPRNG::UniformPoly(const seal::SEALContext &context, RLWEPt *poly) {
  SPU_ENFORCE(poly != nullptr);
  SPU_ENFORCE(context.parameters_set());
  auto cntxt = context.first_context_data();
  size_t N = cntxt->parms().poly_modulus_degree();
  const auto &modulus = cntxt->parms().coeff_modulus();
  poly->parms_id() = seal::parms_id_zero;
  poly->resize(N * modulus.size());
  auto *dst_ptr = poly->data();
  for (const auto &prime : modulus) {
    UniformPrime(prime, {dst_ptr, N});
    dst_ptr += N;
  }
  poly->parms_id() = cntxt->parms_id();
}

// Uniform random on ring 2^k
ArrayRef EnableCPRNG::CPRNG(FieldType field, size_t size) {
  constexpr uint64_t kPRNG_THREASHOLD = 1ULL << 50;
  // Lock prng_counter_
  std::scoped_lock guard(counter_lock_);
  if (prng_counter_ > kPRNG_THREASHOLD) {
    seed_ = yacl::crypto::RandSeed(true);
    prng_counter_ = 0;
  }
  return ring_rand(field, size, seed_, &prng_counter_);
}

}  // namespace spu::mpc::cheetah
