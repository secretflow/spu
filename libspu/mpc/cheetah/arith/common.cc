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

#include "yacl/crypto/rand/rand.h"

#include "libspu/core/memref.h"
#include "libspu/mpc/cheetah/rlwe/utils.h"
#include "libspu/mpc/utils/ring_ops.h"
namespace spu::mpc::cheetah {

EnableCPRNG::EnableCPRNG()
    : seed_(yacl::crypto::SecureRandSeed()), prng_counter_(0) {}

// Uniform random on prime field
void EnableCPRNG::UniformPrime(const seal::Modulus &prime,
                               absl::Span<uint64_t> dst) {
  SPU_ENFORCE(dst.size() > 0);
  constexpr auto max_random = static_cast<uint64_t>(0xFFFFFFFFFFFFFFFFULL);

  // Sample from [0, n*p) such that n*p ~ 2^64
  auto max_multiple =
      max_random - BarrettReduce<uint64_t>(max_random, prime) - 1;

  auto r = CPRNG(64, dst.size());
  MemRefView<const uint64_t> xr(r);
  pforeach(0, dst.size(), [&](int64_t i) { dst[i] = xr[i]; });
  // std::copy_n(xr.data(), xr.size(), dst.data());
  std::transform(dst.data(), dst.data() + dst.size(), dst.data(),
                 [&](uint64_t u) {
                   while (u >= max_multiple) {
                     // barely hit in
                     u = CPRNG(64, 1).at<uint64_t>(0);
                   }
                   return BarrettReduce<uint64_t>(u, prime);
                 });
}

void EnableCPRNG::UniformPoly(const seal::SEALContext &context, RLWEPt *poly,
                              seal::parms_id_type pid) {
  SPU_ENFORCE(poly != nullptr);
  SPU_ENFORCE(context.parameters_set());
  if (pid == seal::parms_id_zero) {
    pid = context.first_parms_id();
  }
  auto cntxt = context.get_context_data(pid);
  SPU_ENFORCE(cntxt != nullptr);
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
MemRef EnableCPRNG::CPRNG(size_t field, size_t size) {
  constexpr uint64_t kPRNG_THREASHOLD = 1ULL << 50;
  // Lock prng_counter_
  std::scoped_lock guard(counter_lock_);
  if (prng_counter_ > kPRNG_THREASHOLD) {
    seed_ = yacl::crypto::SecureRandSeed();
    prng_counter_ = 0;
  }
  MemRef ret(makeType<RingTy>(SE_INVALID, field), {static_cast<int64_t>(size)});
  ring_rand(ret, seed_, &prng_counter_);
  return ret;
}

MemRef ring_conv2d(const MemRef &memref, const MemRef &filter,
                   int64_t num_memrefs, Shape3D memref_shape,
                   int64_t num_filters, Shape3D filter_shape,
                   Shape2D window_strides) {
  auto field = SizeOf(memref.eltype().storage_type()) * 8;
  Shape4D result_shape;
  result_shape[0] = num_memrefs;
  for (int s : {0, 1}) {
    result_shape[s + 1] =
        (memref_shape[s] - filter_shape[s] + window_strides[s]) /
        window_strides[s];
  }
  result_shape[3] = num_filters;

  Shape ts = {num_memrefs, memref_shape[0], memref_shape[1], memref_shape[2]};
  Shape fs = {filter_shape[0], filter_shape[1], filter_shape[2], num_filters};

  MemRef _memref = memref.reshape(ts);
  MemRef _filter = filter.reshape(fs);
  MemRef _ret(makeType<RingTy>(SE_INVALID, field), result_shape);
  ring_zeros(_ret);

  DISPATCH_ALL_STORAGE_TYPES(GetStorageType(field), [&]() {
    // NOTE(juhou): valid padding so offset are always 0.
    constexpr int64_t padh = 0;
    constexpr int64_t padw = 0;

    for (int64_t ib = 0; ib < ts[0]; ++ib) {
      for (int64_t oc = 0; oc < fs[3]; ++oc) {
        for (int64_t ih = -padh, oh = 0; oh < result_shape[1];
             ih += window_strides[0], ++oh) {
          for (int64_t iw = -padw, ow = 0; ow < result_shape[2];
               iw += window_strides[1], ++ow) {
            ScalarT sum{0};

            for (int64_t ic = 0; ic < filter_shape[2]; ++ic) {
              for (int64_t fh = 0; fh < filter_shape[0]; ++fh) {
                for (int64_t fw = 0; fw < filter_shape[1]; ++fw) {
                  auto f = _filter.at<ScalarT>({fh, fw, ic, oc});
                  auto t = _memref.at<ScalarT>({ib, ih + fh, iw + fw, ic});
                  sum += f * t;
                }
              }
            }
            _ret.at<ScalarT>({ib, oh, ow, oc}) = sum;
          }
        }
      }
    }
  });

  return _ret;
}
}  // namespace spu::mpc::cheetah
