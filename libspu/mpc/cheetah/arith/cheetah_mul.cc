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
#include "libspu/mpc/cheetah/arith/cheetah_mul.h"

#include <future>
#include <mutex>
#include <unordered_map>
#include <utility>

#include "absl/types/span.h"
#include "seal/batchencoder.h"
#include "seal/context.h"
#include "seal/decryptor.h"
#include "seal/encryptor.h"
#include "seal/evaluator.h"
#include "seal/keygenerator.h"
#include "seal/publickey.h"
#include "seal/secretkey.h"
#include "seal/util/locks.h"
#include "seal/util/polyarithsmallmod.h"
#include "seal/valcheck.h"
#include "spdlog/spdlog.h"
#include "xtensor/xview.hpp"
#include "yacl/link/link.h"
#include "yacl/utils/parallel.h"

#include "libspu/core/xt_helper.h"
#include "libspu/mpc/cheetah/arith/common.h"
#include "libspu/mpc/cheetah/rlwe/modswitch_helper.h"
#include "libspu/mpc/cheetah/rlwe/utils.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {

struct CheetahMul::Impl : public EnableCPRNG {
 public:
  // RLWE parameters: N = 8192, Q \approx 2^{109}, t \approx 2^{40}
  // NOTE(juhou): Under this parameters, the Mul() might introduce 1-bit error
  // within a small chance Pr = 2^{-32}.
  static constexpr size_t kPolyDegree = 8192;
  static constexpr size_t kCipherModulusBits = 109;
  static constexpr uint32_t kSmallPrimeBitLen = 40;

  static constexpr int kNoiseFloodRandomBits = 50;

  static constexpr size_t kParallelGrain = 1;
  static constexpr size_t kCtAsyncParallel = 8;

  explicit Impl(std::shared_ptr<yacl::link::Context> lctx)
      : lctx_(std::move(lctx)) {
    parms_ = DecideSEALParameters(kSmallPrimeBitLen);
  }

  ~Impl() = default;

  constexpr size_t OLEBatchSize() const { return kPolyDegree; }

  int Rank() const { return lctx_->Rank(); }
  static seal::EncryptionParameters DecideSEALParameters(uint32_t ring_bitlen) {
    size_t poly_deg = kPolyDegree;
    auto scheme_type = seal::scheme_type::bfv;
    auto parms = seal::EncryptionParameters(scheme_type);
    std::vector<int> modulus_bits;
    // NOTE(juhou): We set the 2nd modulus a bit larger than
    // `kSmallPrimeBitLen`. We will drop the 2nd modulus during the H2A step.
    // Also, it helps reducing the noise in the BFV ciphertext.
    modulus_bits = {60, 49};
    parms.set_use_special_prime(false);
    parms.set_poly_modulus_degree(poly_deg);
    parms.set_coeff_modulus(seal::CoeffModulus::Create(poly_deg, modulus_bits));
    return parms;
  }

  size_t num_slots() const { return parms_.poly_modulus_degree(); }

  void LazyExpandSEALContexts(uint32_t field_bitlen,
                              yacl::link::Context *conn = nullptr);

  ArrayRef MulOLE(const ArrayRef &shr, yacl::link::Context *conn,
                  bool evaluator);

 protected:
  void LocalExpandSEALContexts(size_t target);

  static inline uint32_t FieldBitLen(FieldType f) { return 8 * SizeOf(f); }

  static inline uint32_t TotalCRTBitLen(uint32_t field_bitlen) {
    if (field_bitlen < 26) {
      // 1 <= k < 26 uses P = 80bit
      return 2 * kSmallPrimeBitLen;
    } else if (field_bitlen < 64) {
      // 26 <= k < 64 uses P = 120bit
      return 3 * kSmallPrimeBitLen;
    } else if (field_bitlen < 128) {
      // 64 <= k < 128 uses P = 160bit
      // The Pr(1bit error) < 2^{-32}
      return 4 * kSmallPrimeBitLen;
    }
    // k == 128 uses P = 280bit
    SPU_ENFORCE_EQ(field_bitlen, 128U);
    return 7 * kSmallPrimeBitLen;
  }

  void LazyInitModSwitchHelper(uint32_t field_bitlen);

  inline uint32_t WorkingContextSize(uint32_t field_bitlen) const {
    uint32_t target_bitlen = TotalCRTBitLen(field_bitlen);
    SPU_ENFORCE(target_bitlen <= current_crt_plain_bitlen_,
                "Call ExpandSEALContexts first");
    return CeilDiv(target_bitlen, kSmallPrimeBitLen);
  }

  struct Options {
    size_t max_pack = 0;
    bool scale_delta = false;
  };

  // The array will be partitioned into sub-array of `options.max_pack` length.
  //   If `options.max_pack = 0`, set it to `num_slots`.
  //   If `options.scale_delta = true`, scale up it by Delta.
  void EncodeArray(const ArrayRef &array, const Options &options,
                   absl::Span<RLWEPt> out);

  void EncodeArray(const ArrayRef &array, const Options &options,
                   std::vector<RLWEPt> *out) {
    size_t num_elts = array.numel();
    auto eltype = array.eltype();
    SPU_ENFORCE(num_elts > 0, "empty array");
    SPU_ENFORCE(eltype.isa<RingTy>(), "array must be ring_type, got={}",
                eltype);

    auto field = eltype.as<Ring2k>()->field();
    size_t max_pack = options.max_pack > 0 ? options.max_pack : num_slots();
    size_t num_splits = CeilDiv(num_elts, max_pack);
    size_t num_seal_ctx = WorkingContextSize(FieldBitLen(field));
    size_t num_polys = num_seal_ctx * num_splits;
    out->resize(num_polys);
    absl::Span<RLWEPt> wrap(out->data(), out->size());
    EncodeArray(array, options, wrap);
  }

  // out = EncodeArray(array) if out is not null
  // return the payload size (absl::Buffer)
  size_t EncryptArrayThenSend(const ArrayRef &array,
                              std::vector<RLWEPt> *out = nullptr,
                              yacl::link::Context *conn = nullptr);

  // Sample random array `r` of `size` elements in the field.
  // Then compute ciphers*plains + r and response the result to the peer.
  // Return teh sampled array `r`.
  ArrayRef MuThenResponse(FieldType field, size_t num_elts,
                          absl::Span<const yacl::Buffer> ciphers,
                          absl::Span<const RLWEPt> plains,
                          yacl::link::Context *conn = nullptr);

  ArrayRef PrepareRandomMask(FieldType field, size_t size,
                             const Options &options,
                             std::vector<RLWEPt> *encoded_mask);

  ArrayRef PrepareRandomMask(FieldType field, size_t size,
                             std::vector<RLWEPt> *encoded_mask) {
    Options options;
    options.max_pack = num_slots();
    return PrepareRandomMask(field, size, options, encoded_mask);
  }

  ArrayRef DecryptArray(FieldType field, size_t size,
                        const std::vector<yacl::Buffer> &ct_array);

  void NoiseFloodCiphertext(RLWECt &ct, const seal::SEALContext &context);

  void RandomizeCipherForDecryption(RLWECt &ct, size_t cidx);

 private:
  std::shared_ptr<yacl::link::Context> lctx_;

  seal::EncryptionParameters parms_;

  uint32_t current_crt_plain_bitlen_{0};

  // SEAL's contexts for ZZ_{2^k}
  mutable std::mutex context_lock_;
  std::vector<seal::SEALContext> seal_cntxts_;

  // own secret key
  std::shared_ptr<seal::SecretKey> secret_key_;
  // the public key received from the opposite party
  std::shared_ptr<seal::PublicKey> pair_public_key_;

  std::unordered_map<size_t, ModulusSwitchHelper> ms_helpers_;

  std::vector<std::shared_ptr<seal::Encryptor>> sym_encryptors_;
  std::vector<std::shared_ptr<seal::Decryptor>> decryptors_;
  std::vector<std::shared_ptr<seal::Encryptor>> pk_encryptors_;
  std::vector<std::shared_ptr<seal::BatchEncoder>> bfv_encoders_;
};

void CheetahMul::Impl::LazyInitModSwitchHelper(uint32_t field_bitlen) {
  std::unique_lock guard(context_lock_);
  if (ms_helpers_.count(field_bitlen) > 0) {
    return;
  }

  uint32_t target_plain_bitlen = TotalCRTBitLen(field_bitlen);
  SPU_ENFORCE(current_crt_plain_bitlen_ >= target_plain_bitlen);
  std::vector<seal::Modulus> crt_modulus;
  uint32_t accum_plain_bitlen = 0;

  for (size_t idx = 0; accum_plain_bitlen < target_plain_bitlen; ++idx) {
    auto crt_moduli =
        seal_cntxts_[idx].key_context_data()->parms().plain_modulus();
    accum_plain_bitlen += crt_moduli.bit_count();
    crt_modulus.push_back(crt_moduli);
  }

  // NOTE(juhou): we use ckks for this crt_context
  auto parms = seal::EncryptionParameters(seal::scheme_type::ckks);
  parms.set_poly_modulus_degree(parms_.poly_modulus_degree());
  parms.set_coeff_modulus(crt_modulus);

  seal::SEALContext crt_context(parms, false, seal::sec_level_type::none);
  SPU_ENFORCE(crt_context.parameters_set());
  ms_helpers_.emplace(field_bitlen,
                      ModulusSwitchHelper(crt_context, field_bitlen));
}

void CheetahMul::Impl::LocalExpandSEALContexts(size_t target) {
  // For other CRT context, we just copy the sk/pk
  SPU_ENFORCE(target > 0 && target < seal_cntxts_.size());
  SPU_ENFORCE(sym_encryptors_.size() == target);
  SPU_ENFORCE(decryptors_.size() == target);
  SPU_ENFORCE(pk_encryptors_.size() == target);

  seal::SecretKey sk;
  sk.data().resize(secret_key_->data().coeff_count());
  std::copy_n(secret_key_->data().data(), secret_key_->data().coeff_count(),
              sk.data().data());
  sk.parms_id() = seal_cntxts_[target].key_parms_id();

  size_t keysze = pair_public_key_->data().size();
  size_t numel = pair_public_key_->data().poly_modulus_degree() *
                 pair_public_key_->data().coeff_modulus_size();

  seal::PublicKey pk;
  pk.data().resize(seal_cntxts_[target], sk.parms_id(), keysze);
  std::copy_n(pair_public_key_->data().data(), keysze * numel,
              pk.data().data());
  pk.data().is_ntt_form() = pair_public_key_->data().is_ntt_form();
  pk.parms_id() = sk.parms_id();

  sym_encryptors_.push_back(
      std::make_shared<seal::Encryptor>(seal_cntxts_[target], sk));
  decryptors_.push_back(
      std::make_shared<seal::Decryptor>(seal_cntxts_[target], sk));
  pk_encryptors_.push_back(
      std::make_shared<seal::Encryptor>(seal_cntxts_[target], pk));
}

void CheetahMul::Impl::LazyExpandSEALContexts(uint32_t field_bitlen,
                                              yacl::link::Context *conn) {
  uint32_t target_plain_bitlen = TotalCRTBitLen(field_bitlen);
  std::unique_lock guard(context_lock_);
  if (current_crt_plain_bitlen_ >= target_plain_bitlen) {
    return;
  }

  uint32_t num_seal_ctx = CeilDiv(target_plain_bitlen, kSmallPrimeBitLen);
  std::vector<int> crt_moduli_bits(num_seal_ctx, kSmallPrimeBitLen);

  auto crt_modulus =
      seal::CoeffModulus::Create(parms_.poly_modulus_degree(), crt_moduli_bits);
  // NOTE(juhou): sort the primes to make sure new primes are placed in the back
  std::sort(crt_modulus.begin(), crt_modulus.end(),
            [](const seal::Modulus &p, const seal::Modulus &q) {
              return p.value() > q.value();
            });
  uint32_t current_num_ctx = seal_cntxts_.size();

  for (uint32_t i = current_num_ctx; i < num_seal_ctx; ++i) {
    uint64_t new_plain_modulus = crt_modulus[current_num_ctx].value();
    // new plain modulus should be co-prime to all the previous plain modulus
    for (uint32_t j = 0; j < current_num_ctx; ++j) {
      uint32_t prev_plain_modulus =
          seal_cntxts_[j].key_context_data()->parms().plain_modulus().value();
      SPU_ENFORCE_NE(new_plain_modulus, prev_plain_modulus);
    }
  }

  if (conn == nullptr) {
    conn = lctx_.get();
  }

  for (uint32_t idx = current_num_ctx; idx < num_seal_ctx; ++idx) {
    parms_.set_plain_modulus(crt_modulus[idx]);
    seal_cntxts_.emplace_back(parms_, true, seal::sec_level_type::tc128);

    if (idx == 0) {
      seal::KeyGenerator keygen(seal_cntxts_[0]);
      secret_key_ = std::make_shared<seal::SecretKey>(keygen.secret_key());

      auto pk = keygen.create_public_key();
      // NOTE(juhou): we patched seal/util/serializable.h
      auto pk_buf_send = EncodeSEALObject(pk.obj());
      // exchange the public key
      int nxt_rank = conn->NextRank();
      yacl::Buffer pk_buf_recv;
      if (0 == nxt_rank) {
        conn->Send(nxt_rank, pk_buf_send, "rank1 send pk");
        pk_buf_recv = conn->Recv(nxt_rank, "rank1 recv pk");
      } else {
        pk_buf_recv = conn->Recv(nxt_rank, "rank0 recv pk");
        conn->Send(nxt_rank, pk_buf_send, "rank0 send pk");
      }
      pair_public_key_ = std::make_shared<seal::PublicKey>();
      DecodeSEALObject(pk_buf_recv, seal_cntxts_[0], pair_public_key_.get());

      // create the functors
      sym_encryptors_.push_back(
          std::make_shared<seal::Encryptor>(seal_cntxts_[0], *secret_key_));
      decryptors_.push_back(
          std::make_shared<seal::Decryptor>(seal_cntxts_[0], *secret_key_));
      pk_encryptors_.push_back(std::make_shared<seal::Encryptor>(
          seal_cntxts_[0], *pair_public_key_));
    } else {
      // For other CRT context, we just copy the sk/pk
      LocalExpandSEALContexts(idx);
    }

    bfv_encoders_.push_back(
        std::make_shared<seal::BatchEncoder>(seal_cntxts_.back()));
  }
  current_crt_plain_bitlen_ = target_plain_bitlen;
  SPDLOG_INFO(
      "BeaverCheetah::Mul uses {} modulus ({} bit each) for {} bit ring",
      num_seal_ctx, kSmallPrimeBitLen, field_bitlen);
}

ArrayRef CheetahMul::Impl::MulOLE(const ArrayRef &shr,
                                  yacl::link::Context *conn, bool evaluator) {
  if (conn == nullptr) {
    conn = lctx_.get();
  }

  auto eltype = shr.eltype();
  SPU_ENFORCE(eltype.isa<RingTy>(), "must be ring_type, got={}", eltype);
  SPU_ENFORCE(shr.numel() > 0);

  auto field = eltype.as<Ring2k>()->field();
  LazyExpandSEALContexts(FieldBitLen(field), conn);
  LazyInitModSwitchHelper(FieldBitLen(field));

  size_t numel = shr.numel();
  int nxt_rank = conn->NextRank();
  std::vector<RLWEPt> encoded_shr;

  if (evaluator) {
    Options options;
    options.max_pack = num_slots();
    options.scale_delta = false;
    EncodeArray(shr, options, &encoded_shr);
    size_t payload_sze = encoded_shr.size();
    std::vector<yacl::Buffer> recv_ct(payload_sze);
    for (size_t idx = 0; idx < payload_sze; ++idx) {
      recv_ct[idx] = conn->Recv(nxt_rank, "");
    }
    return MuThenResponse(field, numel, recv_ct, encoded_shr, conn);
  }
  size_t payload_sze = EncryptArrayThenSend(shr, nullptr, conn);
  std::vector<yacl::Buffer> recv_ct(payload_sze);
  for (size_t idx = 0; idx < payload_sze; ++idx) {
    recv_ct[idx] = conn->Recv(nxt_rank, "");
  }
  return DecryptArray(field, numel, recv_ct);
}

size_t CheetahMul::Impl::EncryptArrayThenSend(const ArrayRef &array,
                                              std::vector<RLWEPt> *out,
                                              yacl::link::Context *conn) {
  size_t num_elts = array.numel();
  auto eltype = array.eltype();
  SPU_ENFORCE(num_elts > 0, "empty array");
  SPU_ENFORCE(eltype.isa<RingTy>(), "array must be ring_type, got={}", eltype);

  Options options;
  options.max_pack = num_slots();
  options.scale_delta = true;

  auto field = eltype.as<Ring2k>()->field();
  size_t field_bitlen = FieldBitLen(field);
  size_t num_splits = CeilDiv(num_elts, options.max_pack);
  size_t num_seal_ctx = WorkingContextSize(field_bitlen);
  size_t num_polys = num_seal_ctx * num_splits;

  std::vector<RLWEPt> encoded_array;
  absl::Span<RLWEPt> _wrap;
  if (out != nullptr) {
    out->resize(num_polys);
    _wrap = {out->data(), num_polys};
  } else {
    encoded_array.resize(num_polys);
    _wrap = {encoded_array.data(), num_polys};
  }
  EncodeArray(array, options, _wrap);

  std::vector<yacl::Buffer> payload(num_polys);
  yacl::parallel_for(
      0, num_seal_ctx, kParallelGrain, [&](size_t cntxt_bgn, size_t cntxt_end) {
        for (size_t c = cntxt_bgn; c < cntxt_end; ++c) {
          size_t offset = c * num_splits;
          for (size_t idx = 0; idx < num_splits; ++idx) {
            // erase the random from memory
            AutoMemGuard guard(&encoded_array[offset + idx]);
            auto ct =
                sym_encryptors_[c]->encrypt_symmetric(_wrap[offset + idx]);
            payload.at(offset + idx) = EncodeSEALObject(ct.obj());
          }
        }
      });

  if (conn == nullptr) {
    conn = lctx_.get();
  }

  int nxt_rank = conn->NextRank();
  for (size_t i = 0; i < payload.size(); i += kCtAsyncParallel) {
    size_t this_batch = std::min(payload.size() - i, kCtAsyncParallel);
    conn->Send(nxt_rank, payload[i], "");
    for (size_t j = 1; j < this_batch; ++j) {
      conn->SendAsync(nxt_rank, payload[i + j], "");
    }
  }
  return payload.size();
}

ArrayRef CheetahMul::Impl::PrepareRandomMask(
    FieldType field, size_t size, const Options &options,
    std::vector<RLWEPt> *encoded_mask) {
  const size_t max_pack = options.max_pack;
  const size_t num_splits = CeilDiv(size, max_pack);
  const size_t field_bitlen = FieldBitLen(field);
  const size_t num_seal_ctx = WorkingContextSize(field_bitlen);
  const size_t num_polys = num_seal_ctx * num_splits;
  SPU_ENFORCE(ms_helpers_.count(field_bitlen) > 0);
  encoded_mask->resize(num_polys);

  // sample r from [0, P) in the RNS format
  // Ref: the one-bit approximate re-sharing in Cheetah's paper (eprint ver).
  std::vector<uint64_t> random_rns(size * num_seal_ctx);
  for (size_t cidx = 0; cidx < num_seal_ctx; ++cidx) {
    const auto &plain_mod =
        seal_cntxts_[cidx].key_context_data()->parms().plain_modulus();
    size_t offset = cidx * num_splits;
    std::vector<uint64_t> u64tmp(num_slots(), 0);

    for (size_t j = 0; j < num_splits; ++j) {
      size_t bgn = j * max_pack;
      size_t end = std::min(bgn + max_pack, size);
      size_t len = end - bgn;

      // sample the RNS component of r from [0, p_i)
      uint64_t *dst_ptr = random_rns.data() + cidx * size + bgn;
      absl::Span<uint64_t> dst_wrap(dst_ptr, len);
      UniformPrime(plain_mod, dst_wrap);

      std::copy_n(dst_wrap.data(), len, u64tmp.data());
      std::fill_n(u64tmp.data() + len, u64tmp.size() - len, 0);

      bfv_encoders_[cidx]->encode(u64tmp, encoded_mask->at(offset + j));
    }
  }

  // convert x \in [0, P) to [0, 2^k) by round(2^k*x/P)
  auto &ms_helper = ms_helpers_.find(field_bitlen)->second;
  absl::Span<const uint64_t> inp(random_rns.data(), random_rns.size());
  return ms_helper.ModulusDownRNS(field, inp);
}

void CheetahMul::Impl::EncodeArray(const ArrayRef &array,
                                   const Options &options,
                                   absl::Span<RLWEPt> out) {
  size_t num_elts = array.numel();
  auto eltype = array.eltype();
  SPU_ENFORCE(num_elts > 0, "empty array");
  SPU_ENFORCE(eltype.isa<RingTy>(), "array must be ring_type, got={}", eltype);

  auto field = eltype.as<Ring2k>()->field();
  size_t max_pack = options.max_pack > 0 ? options.max_pack : num_slots();
  size_t num_splits = CeilDiv(num_elts, max_pack);
  size_t field_bitlen = FieldBitLen(field);
  size_t num_seal_ctx = WorkingContextSize(field_bitlen);
  size_t num_polys = num_seal_ctx * num_splits;
  SPU_ENFORCE_EQ(out.size(), num_polys,
                 "out size mismatch, expect={}, got={}size", num_polys,
                 out.size());
  SPU_ENFORCE(ms_helpers_.count(field_bitlen) > 0);

  auto &ms_helper = ms_helpers_.find(field_bitlen)->second;

  yacl::parallel_for(
      0, num_seal_ctx, kParallelGrain, [&](size_t cntxt_bgn, size_t cntxt_end) {
        std::vector<uint64_t> u64tmp(num_slots());

        for (size_t cidx = cntxt_bgn; cidx < cntxt_end; ++cidx) {
          const size_t offset = cidx * num_splits;

          for (size_t idx = 0; idx < num_splits; ++idx) {
            auto slice = array.slice(idx * max_pack,
                                     std::min(num_elts, (idx + 1) * max_pack));
            absl::Span<uint64_t> dst(u64tmp.data(), slice.numel());

            if (options.scale_delta) {
              ms_helper.ModulusUpAt(slice, cidx, dst);
            } else {
              ms_helper.CenteralizeAt(slice, cidx, dst);
            }

            // zero-padding the rest
            std::fill_n(u64tmp.data() + slice.numel(),
                        u64tmp.size() - slice.numel(), 0);

            CATCH_SEAL_ERROR(
                bfv_encoders_[cidx]->encode(u64tmp, out[offset + idx]));
          }
        }
      });
}

ArrayRef CheetahMul::Impl::MuThenResponse(
    FieldType field, size_t num_elts, absl::Span<const yacl::Buffer> ciphers,
    absl::Span<const RLWEPt> plains, yacl::link::Context *conn) {
  SPU_ENFORCE(!ciphers.empty(), "BeaverCheetah: empty cipher");
  SPU_ENFORCE(plains.size() == ciphers.size(),
              "BeaverCheetah: ct/pt size mismatch");

  const size_t max_pack = num_slots();
  const size_t num_splits = CeilDiv(num_elts, max_pack);
  const size_t field_bitlen = FieldBitLen(field);
  const size_t num_seal_ctx = WorkingContextSize(field_bitlen);
  const size_t num_ciphers = num_seal_ctx * num_splits;
  SPU_ENFORCE(ciphers.size() == num_ciphers,
              fmt::format("MuThenResponse: expect {} != {}", num_ciphers,
                          ciphers.size()));

  std::vector<RLWEPt> ecd_random;
  auto rnd_mask = PrepareRandomMask(field, num_elts, &ecd_random);
  SPU_ENFORCE(ecd_random.size() == num_ciphers,
              "BeaverCheetah: encoded poly size mismatch");

  std::vector<yacl::Buffer> response(num_ciphers);
  yacl::parallel_for(
      0, num_seal_ctx, kParallelGrain, [&](size_t cntxt_bgn, size_t cntxt_end) {
        RLWECt ct;
        for (size_t cidx = cntxt_bgn; cidx < cntxt_end; ++cidx) {
          const size_t offset = cidx * num_splits;
          const auto &seal_cntxt = seal_cntxts_[cidx];
          seal::Evaluator evaluator(seal_cntxt);

          std::vector<uint64_t> u64tmp(max_pack, 0);
          // Multiply-then-H2A
          for (size_t idx = 0; idx < num_splits; ++idx) {
            DecodeSEALObject(ciphers.at(offset + idx), seal_cntxt, &ct);
            // Multiply step
            CATCH_SEAL_ERROR(
                evaluator.multiply_plain_inplace(ct, plains[offset + idx]));
            // re-randomize the ciphertext (e.g., noise flood)
            RandomizeCipherForDecryption(ct, cidx);
            // H2A
            CATCH_SEAL_ERROR(
                evaluator.sub_plain_inplace(ct, ecd_random[offset + idx]));
            // Truncate for a smaller communication
            TruncateBFVForDecryption(ct, seal_cntxt);
            response[offset + idx] = EncodeSEALObject(ct);
          }
        }
      });

  if (conn == nullptr) {
    conn = lctx_.get();
  }

  int nxt_rank = conn->NextRank();
  for (size_t i = 0; i < response.size(); i += kCtAsyncParallel) {
    size_t this_batch = std::min(response.size() - i, kCtAsyncParallel);
    conn->Send(nxt_rank, response[i], "");
    for (size_t j = 1; j < this_batch; ++j) {
      conn->SendAsync(nxt_rank, response[i + j], "");
    }
  }

  for (auto &pt : ecd_random) {
    AutoMemGuard{&pt};
  }

  return rnd_mask;
}

ArrayRef CheetahMul::Impl::DecryptArray(
    FieldType field, size_t size, const std::vector<yacl::Buffer> &ct_array) {
  const size_t max_pack = num_slots();
  const size_t num_splits = CeilDiv(size, max_pack);
  const size_t field_bitlen = FieldBitLen(field);
  const size_t num_seal_ctx = WorkingContextSize(field_bitlen);
  const size_t num_ciphers = num_seal_ctx * num_splits;
  SPU_ENFORCE(ct_array.size() == num_ciphers,
              "BeaverCheetah: cipher size mismatch");
  SPU_ENFORCE(ms_helpers_.count(field_bitlen) > 0);

  auto rns_temp = ring_zeros(FieldType::FM64, size * num_seal_ctx);
  auto xrns_temp = xt_mutable_adapt<uint64_t>(rns_temp);

  yacl::parallel_for(
      0, num_seal_ctx, kParallelGrain, [&](size_t cntxt_bgn, size_t cntxt_end) {
        // Loop each SEALContext
        // For each context, we obtain `size` uint64 from `num_splits` polys.
        // Each poly will decode to `max_pack` uint64, i.e., `max_pack *
        // num_splits
        // >= size`.
        for (size_t cidx = cntxt_bgn; cidx < cntxt_end; ++cidx) {
          const size_t offset = cidx * num_splits;
          auto ctx_slice =
              xt::view(xrns_temp, xt::range(cidx * size, cidx * size + size));

          RLWEPt pt;
          RLWECt ct;
          std::vector<uint64_t> subarray(max_pack, 0);

          for (size_t idx = 0; idx < num_splits; ++idx) {
            DecodeSEALObject(ct_array.at(offset + idx), seal_cntxts_[cidx],
                             &ct);
            CATCH_SEAL_ERROR(decryptors_[cidx]->decrypt(ct, pt));
            CATCH_SEAL_ERROR(bfv_encoders_[cidx]->decode(pt, subarray));

            size_t bgn = idx * max_pack;
            size_t end = std::min(size, bgn + max_pack);
            size_t len = end - bgn;
            std::copy_n(subarray.data(), len, ctx_slice.begin() + bgn);
          }
        }
      });

  auto &ms_helper = ms_helpers_.find(field_bitlen)->second;
  absl::Span<const uint64_t> inp(xrns_temp.data(), xrns_temp.size());
  return ms_helper.ModulusDownRNS(field, inp);
}

void CheetahMul::Impl::NoiseFloodCiphertext(RLWECt &ct,
                                            const seal::SEALContext &context) {
  SPU_ENFORCE(seal::is_metadata_valid_for(ct, context));
  SPU_ENFORCE(ct.size() == 2);
  auto context_data = context.get_context_data(ct.parms_id());
  yacl::CheckNotNull(context_data.get());
  size_t num_coeffs = ct.poly_modulus_degree();
  size_t num_modulus = ct.coeff_modulus_size();
  const auto &modulus = context_data->parms().coeff_modulus();

  constexpr uint64_t range_mask = (1ULL << kNoiseFloodRandomBits) - 1;
  int logQ = context_data->total_coeff_modulus_bit_count();
  int logt = context_data->parms().plain_modulus().bit_count();
  SPU_ENFORCE_GT(logQ - logt - 1, kNoiseFloodRandomBits);

  // sample random from [0, 2^{kStatRandom})
  auto random = CPRNG(FieldType::FM64, num_coeffs);
  auto xrandom = xt_mutable_adapt<uint64_t>(random);
  std::transform(xrandom.data(), xrandom.data() + xrandom.size(),
                 xrandom.data(), [](uint64_t x) { return x & range_mask; });
  AutoMemGuard guard(&random);

  // add random to each modulus of ct.data(0)
  uint64_t *dst_ptr = ct.data(0);
  for (size_t l = 0; l < num_modulus; ++l) {
    using namespace seal::util;
    if (modulus[l].bit_count() > kNoiseFloodRandomBits) {
      // When prime[l] > 2^{kNoiseFloodRandomBits} then we add directly add
      // the random
      add_poly_coeffmod(xrandom.data(), dst_ptr, num_coeffs, modulus[l],
                        dst_ptr);
    } else {
      // When prime[l] < 2^{kNoiseFloodRandomBits} we need to compute mod
      // prime[l] first
      std::vector<uint64_t> tmp(num_coeffs);
      modulo_poly_coeffs(xrandom.data(), num_coeffs, modulus[l], tmp.data());
      add_poly_coeffmod(tmp.data(), dst_ptr, num_coeffs, modulus[l], dst_ptr);
    }
    dst_ptr += num_coeffs;
  }
}

void CheetahMul::Impl::RandomizeCipherForDecryption(RLWECt &ct, size_t cidx) {
  auto &seal_cntxt = seal_cntxts_.at(cidx);
  auto context_data = seal_cntxt.last_context_data();
  yacl::CheckNotNull(context_data.get());
  seal::Evaluator evaluator(seal_cntxt);
  // 1. Add statistical independent randomness
  NoiseFloodCiphertext(ct, seal_cntxt);

  // 2. Drop all but keep one moduli
  if (ct.coeff_modulus_size() > 1) {
    evaluator.mod_switch_to_inplace(ct, context_data->parms_id());
  }

  // 3. Add zero-encryption for re-randomization
  RLWECt zero_enc;
  CATCH_SEAL_ERROR(pk_encryptors_[cidx]->encrypt_zero(ct.parms_id(), zero_enc));
  CATCH_SEAL_ERROR(evaluator.add_inplace(ct, zero_enc));
}

CheetahMul::CheetahMul(std::shared_ptr<yacl::link::Context> lctx) {
  impl_ = std::make_unique<Impl>(lctx);
}

CheetahMul::~CheetahMul() = default;

int CheetahMul::Rank() const { return impl_->Rank(); }

size_t CheetahMul::OLEBatchSize() const {
  SPU_ENFORCE(impl_ != nullptr);
  return impl_->OLEBatchSize();
}

ArrayRef CheetahMul::MulOLE(const ArrayRef &inp, yacl::link::Context *conn,
                            bool evaluator) {
  SPU_ENFORCE(impl_ != nullptr);
  SPU_ENFORCE(conn != nullptr);
  return impl_->MulOLE(inp, conn, evaluator);
}

ArrayRef CheetahMul::MulOLE(const ArrayRef &inp, bool evaluator) {
  SPU_ENFORCE(impl_ != nullptr);
  return impl_->MulOLE(inp, nullptr, evaluator);
}

}  // namespace spu::mpc::cheetah
