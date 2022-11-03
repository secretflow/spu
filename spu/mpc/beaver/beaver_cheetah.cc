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

#include <future>
#include <mutex>
#include <unordered_map>

#include "absl/types/span.h"
#include "seal/batchencoder.h"
#include "seal/context.h"
#include "seal/decryptor.h"
#include "seal/encryptor.h"
#include "seal/evaluator.h"
#include "seal/galoiskeys.h"
#include "seal/keygenerator.h"
#include "seal/publickey.h"
#include "seal/secretkey.h"
#include "seal/util/locks.h"
#include "seal/util/polyarithsmallmod.h"
#include "seal/valcheck.h"
#include "spdlog/spdlog.h"
#include "xtensor/xvectorize.hpp"
#include "xtensor/xview.hpp"
#include "yasl/link/link.h"
#include "yasl/utils/parallel.h"

#include "spu/core/xt_helper.h"
#include "spu/mpc/beaver/cheetah/lwe_decryptor.h"
#include "spu/mpc/beaver/cheetah/matvec.h"
#include "spu/mpc/beaver/cheetah/modswitch_helper.h"
#include "spu/mpc/beaver/cheetah/util.h"
#include "spu/mpc/beaver/prg_tensor.h"
#include "spu/mpc/util/ring_ops.h"
#include "spu/mpc/util/seal_help.h"

namespace spu::mpc {

template <typename T>
static T CeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

static PrgSeed GetHardwareRandom128() {
  // NOTE(juhou) can we use thr rdseed instruction ?
  std::random_device rd;
  uint64_t lhs = static_cast<uint64_t>(rd()) << 32 | rd();
  uint64_t rhs = static_cast<uint64_t>(rd()) << 32 | rd();
  return yasl::MakeUint128(lhs, rhs);
}

static void TransposeInplace(ArrayRef mat, size_t nrows, size_t ncols) {
  YASL_ENFORCE_EQ((size_t)mat.numel(), nrows * ncols);
  const auto field = mat.eltype().as<Ring2k>()->field();
  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    auto xmat = xt_mutable_adapt<ring2k_t>(mat);
    xmat.reshape({nrows, ncols});
    auto xmatT = xt::eval(xt::transpose(xmat));
    std::copy_n(xmatT.begin(), xmatT.size(), xmat.data());
  });
}

struct EnablePRNG {
  explicit EnablePRNG() : seed_(GetHardwareRandom128()), prng_counter_(0) {}

  // uniform random on prime field
  void CPRNGPrime(const seal::Modulus &prime, absl::Span<uint64_t> dst) {
    YASL_ENFORCE(dst.size() > 0);
    using namespace seal::util;
    constexpr uint64_t max_random =
        static_cast<uint64_t>(0xFFFFFFFFFFFFFFFFULL);

    // sample from [0, n*p) such that n*p ~ 2^64
    auto max_multiple = max_random - barrett_reduce_64(max_random, prime) - 1;

    auto r = CPRNG(FieldType::FM64, dst.size());
    auto xr = xt_adapt<uint64_t>(r);
    std::copy_n(xr.data(), xr.size(), dst.data());
    std::transform(dst.data(), dst.data() + dst.size(), dst.data(),
                   [&](uint64_t u) {
                     while (u >= max_multiple) {
                       // barely hit in
                       u = CPRNG(FieldType::FM64, 1).at<uint64_t>(0);
                     }
                     return barrett_reduce_64(u, prime);
                   });
  }
  // uniform random on ring 2^k
  ArrayRef CPRNG(FieldType field, size_t size) {
    // TODO(juhou) tune this
    constexpr uint64_t kPRNG_THREASHOLD = 1ULL << 40;
    std::scoped_lock guard(counter_lock_);
    // TODO(juhou): PrgCounter type might incompatible with uint64_t.
    if (prng_counter_ > kPRNG_THREASHOLD) {
      seed_ = GetHardwareRandom128();
      prng_counter_ = 0;
    }
    // NOTE(juhou): do we need to replay the PRNG ?
    PrgArrayDesc prg_desc;
    return prgCreateArray(field, size, seed_, &prng_counter_, &prg_desc);
  }

 protected:
  PrgSeed seed_;
  mutable std::mutex counter_lock_;
  PrgCounter prng_counter_;
};

struct BeaverCheetah::MulImpl : public EnablePRNG {
 public:
  static constexpr uint32_t kSmallPrimeBitLen = 36;
  static constexpr size_t kPolyDegree = 8192;
  static constexpr int kNoiseFloodRandomBits = 50;
  static constexpr size_t kParallelGrain = 1;

  MulImpl(std::shared_ptr<yasl::link::Context> lctx)
      : EnablePRNG(), lctx_(lctx) {
    parms_ = DecideSEALParameters(kSmallPrimeBitLen);
  }

  seal::EncryptionParameters DecideSEALParameters(uint32_t ring_bitlen) {
    size_t poly_deg = kPolyDegree;
    auto scheme_type = seal::scheme_type::bfv;
    auto parms = seal::EncryptionParameters(scheme_type);
    std::vector<int> modulus_bits;
    // NOTE(juhou): 109bit should be enough for one multiplication under the
    // `kSmallPrimeBitLen`
    modulus_bits = {60, 49};
    parms.set_use_special_prime(false);
    parms.set_poly_modulus_degree(poly_deg);
    parms.set_coeff_modulus(seal::CoeffModulus::Create(poly_deg, modulus_bits));
    return parms;
  }

  size_t num_slots() const { return parms_.poly_modulus_degree(); }

  void LazyExpandSEALContexts(uint32_t field_bitlen,
                              yasl::link::Context *conn = nullptr);

  Beaver::Triple Mul(FieldType field, size_t size);

  ArrayRef MulAShr(const ArrayRef &shr, yasl::link::Context *conn,
                   bool evaluator);

 protected:
  inline uint32_t FieldBitLen(FieldType f) { return 8 * SizeOf(f); }

  inline uint32_t TotalCRTBitLen(uint32_t field_bitlen) {
    const int margins_for_full_random = 15;
    return 2 * field_bitlen + margins_for_full_random;
  }

  void LazyInitModSwitchHelper(uint32_t field_bitlen);

  inline uint32_t WorkingContextSize(uint32_t field_bitlen) {
    uint32_t target_bitlen = TotalCRTBitLen(field_bitlen);
    YASL_ENFORCE(target_bitlen <= current_crt_plain_bitlen_,
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
    YASL_ENFORCE(num_elts > 0, "empty array");
    YASL_ENFORCE(eltype.isa<RingTy>(), "array must be ring_type, got={}",
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
                              yasl::link::Context *conn = nullptr);

  // Sample random array `r` of `size` elements in the field.
  // Then compute ciphers*plains + r and response the result to the peer.
  // Return teh sampled array `r`.
  ArrayRef ElementMulThenResponse(FieldType field, size_t size,
                                  absl::Span<const yasl::Buffer> ciphers,
                                  absl::Span<const RLWEPt> plains,
                                  yasl::link::Context *conn = nullptr);

  ArrayRef _PrepareRandomMask(FieldType field, size_t size,
                              const Options &options,
                              std::vector<RLWEPt> *encoded_mask);

  ArrayRef PrepareRandomMask(FieldType field, size_t size,
                             std::vector<RLWEPt> *encoded_mask) {
    Options options;
    options.max_pack = num_slots();
    return _PrepareRandomMask(field, size, options, encoded_mask);
  }

  ArrayRef DecryptArray(FieldType field, size_t size,
                        const std::vector<yasl::Buffer> &ct_array);

  void _NoiseFloodCiphertext(RLWECt &ct, const seal::SEALContext &context);

  void RandomizeCipherForDecryption(RLWECt &ct, size_t cidx);

 private:
  std::shared_ptr<yasl::link::Context> lctx_;

  seal::EncryptionParameters parms_;

  uint32_t current_crt_plain_bitlen_{0};

  // SEAL's contexts for ZZ_{2^k}
  mutable std::shared_mutex context_lock_;
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

void BeaverCheetah::MulImpl::LazyInitModSwitchHelper(uint32_t field_bitlen) {
  // TODO(juhou): multi-thread safe for ModulusSwitchHelper ?
  if (ms_helpers_.count(field_bitlen) > 0) return;

  uint32_t target_plain_bitlen = TotalCRTBitLen(field_bitlen);
  YASL_ENFORCE(current_crt_plain_bitlen_ >= target_plain_bitlen);
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
  ms_helpers_.emplace(std::make_pair(
      field_bitlen, ModulusSwitchHelper(crt_context, field_bitlen)));
}

void BeaverCheetah::MulImpl::LazyExpandSEALContexts(uint32_t field_bitlen,
                                                    yasl::link::Context *conn) {
  uint32_t target_plain_bitlen = TotalCRTBitLen(field_bitlen);
  {
    std::shared_lock<std::shared_mutex> guard(context_lock_);
    if (current_crt_plain_bitlen_ >= target_plain_bitlen) return;
  }
  std::unique_lock<std::shared_mutex> guard(context_lock_);
  if (current_crt_plain_bitlen_ >= target_plain_bitlen) return;

  uint32_t num_seal_ctx = CeilDiv(target_plain_bitlen, kSmallPrimeBitLen);
  std::vector<int> crt_moduli_bits(num_seal_ctx, kSmallPrimeBitLen);
  int last_plain = std::max<int>(
      20, target_plain_bitlen - (num_seal_ctx - 1) * kSmallPrimeBitLen);
  crt_moduli_bits.back() = last_plain;
  SPDLOG_INFO(
      "BeaverCheetah::Mul uses {} modulus ({} bit each) for {} bit ring",
      num_seal_ctx, kSmallPrimeBitLen, field_bitlen);

  auto crt_modulus =
      seal::CoeffModulus::Create(parms_.poly_modulus_degree(), crt_moduli_bits);
  uint32_t current_num_ctx = seal_cntxts_.size();

  for (uint32_t i = current_num_ctx; i < num_seal_ctx; ++i) {
    uint64_t new_plain_modulus = crt_modulus[current_num_ctx].value();
    // new plain modulus should be co-prime to all the previous plain modulus
    for (uint32_t j = 0; j < current_num_ctx; ++j) {
      uint32_t prev_plain_modulus =
          seal_cntxts_[j].key_context_data()->parms().plain_modulus().value();
      YASL_ENFORCE_NE(new_plain_modulus, prev_plain_modulus);
    }
  }

  if (!conn) conn = lctx_.get();
  for (uint32_t idx = current_num_ctx; idx < num_seal_ctx; ++idx) {
    parms_.set_plain_modulus(crt_modulus[idx]);
    seal_cntxts_.emplace_back(parms_, true, seal::sec_level_type::tc128);

    if (pair_public_key_ == nullptr) {
      seal::KeyGenerator keygen(seal_cntxts_[0]);
      secret_key_ = std::make_shared<seal::SecretKey>(keygen.secret_key());

      auto pk = keygen.create_public_key();
      // NOTE(juhou): we patched seal/util/serializable.h
      auto pk_buf = EncodeSEALObject(pk.obj());
      // exchange the public key
      int nxt_rank = conn->NextRank();
      conn->SendAsync(nxt_rank, pk_buf, "send Pk");
      pk_buf = conn->Recv(nxt_rank, "recv pk");
      pair_public_key_ = std::make_shared<seal::PublicKey>();
      DecodeSEALObject(pk_buf, seal_cntxts_[0], pair_public_key_.get());

      // create the functors
      sym_encryptors_.push_back(
          std::make_shared<seal::Encryptor>(seal_cntxts_[0], *secret_key_));
      decryptors_.push_back(
          std::make_shared<seal::Decryptor>(seal_cntxts_[0], *secret_key_));
      pk_encryptors_.push_back(std::make_shared<seal::Encryptor>(
          seal_cntxts_[0], *pair_public_key_));
    } else {
      // For other CRT context, we just copy the sk/pk
      seal::SecretKey sk;
      sk.data().resize(secret_key_->data().coeff_count());
      std::copy_n(secret_key_->data().data(), secret_key_->data().coeff_count(),
                  sk.data().data());
      sk.parms_id() = seal_cntxts_[idx].key_parms_id();

      size_t keysze = pair_public_key_->data().size();
      size_t numel = pair_public_key_->data().poly_modulus_degree() *
                     pair_public_key_->data().coeff_modulus_size();

      seal::PublicKey pk;
      pk.data().resize(seal_cntxts_[idx], sk.parms_id(), keysze);
      std::copy_n(pair_public_key_->data().data(), keysze * numel,
                  pk.data().data());
      pk.data().is_ntt_form() = pair_public_key_->data().is_ntt_form();
      pk.parms_id() = sk.parms_id();

      sym_encryptors_.push_back(
          std::make_shared<seal::Encryptor>(seal_cntxts_[idx], sk));
      decryptors_.push_back(
          std::make_shared<seal::Decryptor>(seal_cntxts_[idx], sk));
      pk_encryptors_.push_back(
          std::make_shared<seal::Encryptor>(seal_cntxts_[idx], pk));
    }

    bfv_encoders_.push_back(
        std::make_shared<seal::BatchEncoder>(seal_cntxts_.back()));
  }
  current_crt_plain_bitlen_ = target_plain_bitlen;
}

ArrayRef BeaverCheetah::MulImpl::MulAShr(const ArrayRef &shr,
                                         yasl::link::Context *conn,
                                         bool evaluator) {
  yasl::CheckNotNull(conn);
  auto eltype = shr.eltype();
  YASL_ENFORCE(eltype.isa<RingTy>(), "must be ring_type, got={}", eltype);
  YASL_ENFORCE(shr.numel() > 0);

  auto field = eltype.as<Ring2k>()->field();
  LazyExpandSEALContexts(FieldBitLen(field), conn);
  LazyInitModSwitchHelper(FieldBitLen(field));

  size_t numel = shr.numel();
  int nxt_rank = conn->NextRank();
  std::vector<RLWEPt> encoded_shr;
  if (!evaluator) {
    size_t payload_sze = EncryptArrayThenSend(shr, nullptr, conn);
    std::vector<yasl::Buffer> recv_ct(payload_sze);
    for (size_t idx = 0; idx < payload_sze; ++idx) {
      recv_ct[idx] = conn->Recv(nxt_rank, "");
    }
    return DecryptArray(field, numel, recv_ct);
  } else {
    // NOTE(juhou): rank=0 take more computation
    Options options;
    options.max_pack = num_slots();
    options.scale_delta = false;
    EncodeArray(shr, options, &encoded_shr);
    size_t payload_sze = encoded_shr.size();
    std::vector<yasl::Buffer> recv_ct(payload_sze);
    for (size_t idx = 0; idx < payload_sze; ++idx) {
      recv_ct[idx] = conn->Recv(nxt_rank, "");
    }
    return ElementMulThenResponse(field, numel, recv_ct, encoded_shr, conn);
  }
}

Beaver::Triple BeaverCheetah::MulImpl::Mul(FieldType field, size_t size) {
  LazyExpandSEALContexts(FieldBitLen(field));
  LazyInitModSwitchHelper(FieldBitLen(field));
  YASL_ENFORCE(size > 0);

  auto a = CPRNG(field, size);
  auto b = CPRNG(field, size);

  int rank = lctx_->Rank();
  auto dupx = lctx_->Spawn();
  //  (a0 + a1) * (b0 + b1)
  // = a0*b0 + <a0*b1> + <a1*b0> + a1*b1

  std::future<ArrayRef> task = std::async(std::launch::async, [&] {
    if (rank == 0) {
      return MulAShr(a, dupx.get(), true);
    } else {
      return MulAShr(b, dupx.get(), false);
    }
  });

  ArrayRef b0a1;
  if (rank == 0) {
    b0a1 = MulAShr(b, lctx_.get(), false);
  } else {
    b0a1 = MulAShr(a, lctx_.get(), true);
  }
  ArrayRef a0b1 = task.get();

  auto c = ring_add(ring_add(ring_mul(a, b), a0b1), b0a1);
  return {a, b, c};
}

size_t BeaverCheetah::MulImpl::EncryptArrayThenSend(const ArrayRef &array,
                                                    std::vector<RLWEPt> *out,
                                                    yasl::link::Context *conn) {
  size_t num_elts = array.numel();
  auto eltype = array.eltype();
  YASL_ENFORCE(num_elts > 0, "empty array");
  YASL_ENFORCE(eltype.isa<RingTy>(), "array must be ring_type, got={}", eltype);

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
  if (out) {
    out->resize(num_polys);
    _wrap = {out->data(), num_polys};
  } else {
    encoded_array.resize(num_polys);
    _wrap = {encoded_array.data(), num_polys};
  }
  EncodeArray(array, options, _wrap);

  std::vector<yasl::Buffer> payload(num_polys);
  yasl::parallel_for(
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

  if (!conn) conn = lctx_.get();
  int nxt_rank = conn->NextRank();
  for (auto &ct : payload) {
    conn->SendAsync(nxt_rank, ct, "");
  }

  return payload.size();
}

ArrayRef BeaverCheetah::MulImpl::_PrepareRandomMask(
    FieldType field, size_t size, const Options &options,
    std::vector<RLWEPt> *encoded_mask) {
  const size_t max_pack = options.max_pack;
  const size_t num_splits = CeilDiv(size, max_pack);
  const size_t field_bitlen = FieldBitLen(field);
  const size_t num_seal_ctx = WorkingContextSize(field_bitlen);
  const size_t num_polys = num_seal_ctx * num_splits;
  YASL_ENFORCE(ms_helpers_.count(field_bitlen) > 0);
  encoded_mask->resize(num_polys);

  // sample r from [0, P) in the RNS format
  // Ref: the one-bit approximate re-sharing in Cheetah's paper (eprint ver).
  std::vector<uint64_t> random_rns(size * num_seal_ctx);
  for (size_t cidx = 0; cidx < num_seal_ctx; ++cidx) {
    auto &plain_mod =
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
      CPRNGPrime(plain_mod, dst_wrap);

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

void BeaverCheetah::MulImpl::EncodeArray(const ArrayRef &array,
                                         const Options &options,
                                         absl::Span<RLWEPt> out) {
  size_t num_elts = array.numel();
  auto eltype = array.eltype();
  YASL_ENFORCE(num_elts > 0, "empty array");
  YASL_ENFORCE(eltype.isa<RingTy>(), "array must be ring_type, got={}", eltype);

  auto field = eltype.as<Ring2k>()->field();
  size_t max_pack = options.max_pack > 0 ? options.max_pack : num_slots();
  size_t num_splits = CeilDiv(num_elts, max_pack);
  size_t field_bitlen = FieldBitLen(field);
  size_t num_seal_ctx = WorkingContextSize(field_bitlen);
  size_t num_polys = num_seal_ctx * num_splits;
  YASL_ENFORCE_EQ(out.size(), num_polys,
                  "out size mismatch, expect={}, got={}size", num_polys,
                  out.size());
  YASL_ENFORCE(ms_helpers_.count(field_bitlen) > 0);

  auto &ms_helper = ms_helpers_.find(field_bitlen)->second;

  yasl::parallel_for(
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

ArrayRef BeaverCheetah::MulImpl::ElementMulThenResponse(
    FieldType field, size_t num_elts, absl::Span<const yasl::Buffer> ciphers,
    absl::Span<const RLWEPt> plains, yasl::link::Context *conn) {
  YASL_ENFORCE(ciphers.size() > 0, "BeaverCheetah: empty cipher");
  YASL_ENFORCE(plains.size() == ciphers.size(),
               "BeaverCheetah: ct/pt size mismatch");

  const size_t max_pack = num_slots();
  const size_t num_splits = CeilDiv(num_elts, max_pack);
  const size_t field_bitlen = FieldBitLen(field);
  const size_t num_seal_ctx = WorkingContextSize(field_bitlen);
  const size_t num_ciphers = num_seal_ctx * num_splits;
  YASL_ENFORCE(ciphers.size() == num_ciphers,
               fmt::format("ElementMulThenResponse: expect {} != {}",
                           num_ciphers, ciphers.size()));

  std::vector<RLWEPt> ecd_random;
  auto rnd_mask = PrepareRandomMask(field, num_elts, &ecd_random);
  YASL_ENFORCE(ecd_random.size() == num_ciphers,
               "BeaverCheetah: encoded poly size mismatch");

  std::vector<yasl::Buffer> response(num_ciphers);
  yasl::parallel_for(
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
            // H2A
            CATCH_SEAL_ERROR(
                evaluator.sub_plain_inplace(ct, ecd_random[offset + idx]));
            // re-randomize the ciphertext (e.g., noise flood)
            RandomizeCipherForDecryption(ct, cidx);
            response[offset + idx] = EncodeSEALObject(ct);
          }
        }
      });

  if (!conn) conn = lctx_.get();
  int nxt_rank = conn->NextRank();
  for (auto &ct : response) {
    conn->SendAsync(nxt_rank, ct, fmt::format("Send to P{}", nxt_rank));
  }

  for (auto &pt : ecd_random) {
    AutoMemGuard{&pt};
  }

  return rnd_mask;
}

ArrayRef BeaverCheetah::MulImpl::DecryptArray(
    FieldType field, size_t size, const std::vector<yasl::Buffer> &ct_array) {
  const size_t max_pack = num_slots();
  const size_t num_splits = CeilDiv(size, max_pack);
  const size_t field_bitlen = FieldBitLen(field);
  const size_t num_seal_ctx = WorkingContextSize(field_bitlen);
  const size_t num_ciphers = num_seal_ctx * num_splits;
  YASL_ENFORCE(ct_array.size() == num_ciphers,
               "BeaverCheetah: cipher size mismatch");
  YASL_ENFORCE(ms_helpers_.count(field_bitlen) > 0);

  auto rns_temp = ring_zeros(FieldType::FM64, size * num_seal_ctx);
  auto xrns_temp = xt_mutable_adapt<uint64_t>(rns_temp);

  yasl::parallel_for(
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

void BeaverCheetah::MulImpl::_NoiseFloodCiphertext(
    RLWECt &ct, const seal::SEALContext &context) {
  YASL_ENFORCE(seal::is_metadata_valid_for(ct, context));
  YASL_ENFORCE(ct.size() == 2);
  auto context_data = context.get_context_data(ct.parms_id());
  yasl::CheckNotNull(context_data.get());
  size_t num_coeffs = ct.poly_modulus_degree();
  size_t num_modulus = ct.coeff_modulus_size();
  auto &modulus = context_data->parms().coeff_modulus();

  constexpr uint64_t range_mask = (1ULL << kNoiseFloodRandomBits) - 1;
  int logQ = context_data->total_coeff_modulus_bit_count();
  int logt = context_data->parms().plain_modulus().bit_count();
  YASL_ENFORCE_GT(logQ - logt - 1, kNoiseFloodRandomBits);

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

void BeaverCheetah::MulImpl::RandomizeCipherForDecryption(RLWECt &ct,
                                                          size_t cidx) {
  auto &seal_cntxt = seal_cntxts_.at(cidx);
  auto context_data = seal_cntxt.last_context_data();
  yasl::CheckNotNull(context_data.get());
  seal::Evaluator evaluator(seal_cntxt);
  // 1. Add statistical independent randomness
  _NoiseFloodCiphertext(ct, seal_cntxt);

  // 2. Drop all but keep one moduli
  if (ct.coeff_modulus_size() > 1) {
    evaluator.mod_switch_to_inplace(ct, context_data->parms_id());
  }

  // 3. Add zero-encryption for re-randomization
  RLWECt zero_enc;
  CATCH_SEAL_ERROR(pk_encryptors_[cidx]->encrypt_zero(ct.parms_id(), zero_enc));
  CATCH_SEAL_ERROR(evaluator.add_inplace(ct, zero_enc));

  // 4. Truncate for smaller communication
  TruncateBFVForDecryption(ct, seal_cntxt);
}

struct BeaverCheetah::DotImpl : public EnablePRNG {
 public:
  DotImpl(std::shared_ptr<yasl::link::Context> lctx)
      : EnablePRNG(), lctx_(lctx) {}

  // Compute C = A*B where |A|=M*K, |B|=K*N
  Beaver::Triple Dot(FieldType field, size_t M, size_t N, size_t K);

  seal::EncryptionParameters DecideSEALParameters(uint32_t ring_bitlen) {
    size_t poly_deg;
    std::vector<int> modulus_bits;
    if (ring_bitlen <= 32) {
      poly_deg = 4096;
      modulus_bits = {55, 39};
    } else if (ring_bitlen <= 64) {
      poly_deg = 8192;
      modulus_bits = {55, 55, 48};
    } else {
      poly_deg = 16384;
      modulus_bits = {59, 59, 59, 59, 50};
    }

    auto scheme_type = seal::scheme_type::ckks;
    auto parms = seal::EncryptionParameters(scheme_type);

    parms.set_use_special_prime(false);
    parms.set_poly_modulus_degree(poly_deg);
    parms.set_coeff_modulus(seal::CoeffModulus::Create(poly_deg, modulus_bits));
    return parms;
  }

  inline uint32_t FieldBitLen(FieldType f) const { return 8 * SizeOf(f); }

  void LazyInit(size_t field_bitlen);

  void AddPlainInplace(RLWECt &ct, const RLWEPt &pt,
                       const seal::SEALContext &context) const {
    YASL_ENFORCE(ct.parms_id() == pt.parms_id());
    auto cntxt_dat = context.get_context_data(ct.parms_id());
    YASL_ENFORCE(cntxt_dat != nullptr);
    const auto &parms = cntxt_dat->parms();
    const auto &modulus = parms.coeff_modulus();
    size_t num_coeff = ct.poly_modulus_degree();
    size_t num_modulus = ct.coeff_modulus_size();

    for (size_t l = 0; l < num_modulus; ++l) {
      auto op0 = ct.data(0) + l * num_coeff;
      auto op1 = pt.data() + l * num_coeff;
      seal::util::add_poly_coeffmod(op0, op1, num_coeff, modulus[l], op0);
    }
  }

  void UniformPoly(RLWEPt &out, const seal::SEALContext &context) {
    auto &parms = context.first_context_data()->parms();
    auto &modulus = parms.coeff_modulus();
    out.parms_id() = seal::parms_id_zero;
    size_t num_coeffs = parms.poly_modulus_degree();
    out.resize(num_coeffs * modulus.size());
    for (size_t l = 0; l < modulus.size(); ++l) {
      absl::Span<uint64_t> wrap(out.data() + l * num_coeffs, num_coeffs);
      CPRNGPrime(modulus[l], wrap);
    }
    out.parms_id() = context.first_parms_id();
  }

  // ct -> ct + r, r where r is sampled from the full modulus range.
  void H2A(std::vector<RLWECt> &ct, const seal::SEALContext &context,
           std::vector<RLWEPt> *rnd_mask) {
    size_t num_poly = ct.size();
    YASL_ENFORCE(num_poly > 0);
    YASL_ENFORCE(rnd_mask != nullptr);

    rnd_mask->resize(num_poly);
    for (size_t idx = 0; idx < num_poly; ++idx) {
      auto &rnd = rnd_mask->at(idx);
      UniformPoly(rnd, context);
      if (ct[idx].is_ntt_form()) {
        NttInplace(rnd, context);
      }
      AddPlainInplace(ct[idx], rnd, context);
    }
  }

  void RandomizeCipherForDecryption(std::vector<RLWECt> &ct_array,
                                    const seal::Encryptor &pk_encryptor,
                                    const seal::Evaluator &evaluator) const {
    RLWECt zero_ct;
    for (auto &ct : ct_array) {
      pk_encryptor.encrypt_zero(ct.parms_id(), zero_ct);
      if (zero_ct.is_ntt_form()) {
        evaluator.transform_from_ntt_inplace(zero_ct);
      }
      evaluator.add_inplace(ct, zero_ct);
    }
  }

 private:
  std::shared_ptr<yasl::link::Context> lctx_;

  mutable std::shared_mutex context_lock_;
  // field_bitlen -> functor mapping
  std::unordered_map<size_t, std::shared_ptr<seal::SEALContext>> seal_cntxts_;
  std::unordered_map<size_t, std::shared_ptr<seal::SecretKey>> secret_keys_;
  std::unordered_map<size_t, std::shared_ptr<seal::PublicKey>> pair_pub_keys_;
  std::unordered_map<size_t, std::shared_ptr<ModulusSwitchHelper>> ms_helpers_;
  std::unordered_map<size_t, std::shared_ptr<seal::Encryptor>> sym_encryptors_;
  std::unordered_map<size_t, std::shared_ptr<seal::Encryptor>> pk_encryptors_;
  std::unordered_map<size_t, std::shared_ptr<seal::Decryptor>> decryptors_;
};

void BeaverCheetah::DotImpl::LazyInit(size_t field_bitlen) {
  {
    std::shared_lock<std::shared_mutex> guard(context_lock_);
    if (seal_cntxts_.find(field_bitlen) != seal_cntxts_.end()) return;
  }
  // double-checking
  std::unique_lock<std::shared_mutex> guard(context_lock_);
  if (seal_cntxts_.find(field_bitlen) != seal_cntxts_.end()) return;

  auto parms = DecideSEALParameters(field_bitlen);
  auto this_context =
      new seal::SEALContext(parms, true, seal::sec_level_type::none);
  seal::KeyGenerator keygen(*this_context);
  auto rlwe_sk = new seal::SecretKey(keygen.secret_key());

  auto pk = keygen.create_public_key();
  // NOTE(juhou): we patched seal/util/serializable.h
  auto pk_buf = EncodeSEALObject(pk.obj());
  // exchange the public key
  int nxt_rank = lctx_->NextRank();
  lctx_->SendAsync(nxt_rank, pk_buf, "send Pk");
  pk_buf = lctx_->Recv(nxt_rank, "recv pk");
  auto pair_public_key = std::make_shared<seal::PublicKey>();
  DecodeSEALObject(pk_buf, *this_context, pair_public_key.get());

  auto modulus = parms.coeff_modulus();
  if (parms.use_special_prime()) {
    modulus.pop_back();
  }
  parms.set_coeff_modulus(modulus);
  seal::SEALContext ms_context(parms, false, seal::sec_level_type::none);

  seal_cntxts_.emplace(field_bitlen, this_context);
  secret_keys_.emplace(field_bitlen, rlwe_sk);
  pair_pub_keys_.emplace(field_bitlen, pair_public_key);
  ms_helpers_.emplace(field_bitlen,
                      new ModulusSwitchHelper(ms_context, field_bitlen));
  sym_encryptors_.emplace(field_bitlen,
                          new seal::Encryptor(*this_context, *rlwe_sk));
  pk_encryptors_.emplace(field_bitlen,
                         new seal::Encryptor(*this_context, *pair_public_key));
  decryptors_.emplace(field_bitlen,
                      new seal::Decryptor(*this_context, *rlwe_sk));

  SPDLOG_INFO("BeaverCheetah::Dot uses {} modulus {} degree for {} bit ring",
              modulus.size(), parms.poly_modulus_degree(), field_bitlen);
}

// Compute C = A*B where |A|=M*K, |B|=K*N
Beaver::Triple BeaverCheetah::DotImpl::Dot(FieldType field, size_t M, size_t N,
                                           size_t K) {
  const size_t field_bitlen = FieldBitLen(field);
  LazyInit(field_bitlen);
  const auto &this_context = *seal_cntxts_.find(field_bitlen)->second;
  auto &this_encryptor = sym_encryptors_.find(field_bitlen)->second;
  auto &this_pk_encryptor = pk_encryptors_.find(field_bitlen)->second;
  auto &this_decryptor = decryptors_.find(field_bitlen)->second;
  auto &this_ms = ms_helpers_.find(field_bitlen)->second;
  seal::Evaluator evaluator(this_context);

  MatVecProtocol matvec_prot(this_context, *this_ms);
  YASL_ENFORCE_EQ(this_ms->base_mod_bitlen(), field_bitlen);

  const size_t lhs_nrows = std::max(M, N);
  const size_t loop_dim = std::min(M, N);
  const int nxt_rank = lctx_->NextRank();

  // To compute lhs_mat * rhs_mat = ans_mat + mask_mat
  auto lhs_mat = CPRNG(field, lhs_nrows * K);
  auto rhs_mat = CPRNG(field, loop_dim * K);
  auto ans_mat = ring_zeros(field, loop_dim * lhs_nrows);
  auto mask_mat = ring_zeros(field, loop_dim * lhs_nrows);

  MatVecProtocol::Meta meta;
  meta.nrows = lhs_nrows;
  meta.ncols = K;

  std::vector<RLWEPt> ecd_lhs_mat;
  matvec_prot.EncodeMatrix(meta, lhs_mat, &ecd_lhs_mat);

  std::vector<RLWEPt> ecd_vec;
  std::vector<RLWECt> enc_vec;

  // FIXME: sendAsync may blocking when concurrent sending task exceed
  // `ThrottleWindowSize`, so temporary disable the window, but there's no API
  // to recover it back.
  lctx_->SetThrottleWindowSize(0);

  for (size_t n = 0; n < loop_dim; ++n) {
    auto rhs_slice = rhs_mat.slice(n * K, n * K + K);
    matvec_prot.EncodeVector(meta, rhs_slice, &ecd_vec);
    enc_vec.resize(ecd_vec.size());

    // send encrypted to the peer
    for (size_t idx = 0; idx < ecd_vec.size(); ++idx) {
      NttInplace(ecd_vec[idx], this_context);
      auto ct = this_encryptor->encrypt_symmetric(ecd_vec[idx]).obj();
      lctx_->SendAsync(nxt_rank, EncodeSEALObject(ct), "");
    }

    // recv encrypted vector from the peer
    for (size_t idx = 0; idx < ecd_vec.size(); ++idx) {
      auto payload = lctx_->Recv(nxt_rank, "");
      DecodeSEALObject(payload, this_context, enc_vec.data() + idx);
    }

    // M_a, [v_b] -> [M_a * v_b]
    std::vector<RLWECt> prod;
    matvec_prot.MatVecNoExtract(meta, ecd_lhs_mat, enc_vec, &prod);

    // Re-sharing the matvec product homomorphically
    std::vector<RLWEPt> rnd_masks(prod.size());
    H2A(prod, this_context, &rnd_masks);
    // NOTE(juhou): the random mask is sampled from the whole ciphertext
    // modulus We need to cast it down to mod 2^k using `ParseMatVecResult`
    DISPATCH_ALL_FIELDS(field, "Dot-1", [&]() {
      auto xmask_mat = xt_mutable_adapt<ring2k_t>(mask_mat);
      auto xans_mat = xt_mutable_adapt<ring2k_t>(ans_mat);
      xmask_mat = xmask_mat.reshape({loop_dim, lhs_nrows});
      xans_mat = xans_mat.reshape({loop_dim, lhs_nrows});

      xt::row(xmask_mat, n) = xt_adapt<ring2k_t>(
          matvec_prot.ParseMatVecResult(field, meta, rnd_masks));
    });

    // Before sending the masked matvec product, we need to re-randomize the
    // ciphertext via adding fresh encryption of zero.
    RandomizeCipherForDecryption(prod, *this_pk_encryptor, evaluator);
    // Also, we need to clean up unused coefficients.
    // `ExtractLWEsInplace` should be placed **after**
    // `RandomizeCipherForDecryption` for a smaller communication cost.
    matvec_prot.ExtractLWEsInplace(meta, prod);

    // send the masked product to the peer
    for (size_t idx = 0; idx < prod.size(); ++idx) {
      lctx_->SendAsync(nxt_rank, EncodeSEALObject(prod[idx]), "");
    }
    // recv RLWE vector from the peer
    for (size_t idx = 0; idx < prod.size(); ++idx) {
      auto payload = lctx_->Recv(nxt_rank, "");
      DecodeSEALObject(payload, this_context, prod.data() + idx);
    }

    // Finally, decrypt the RLWEs and parse some of the coefficients as the
    // matvec result.
    for (size_t idx = 0; idx < prod.size(); ++idx) {
      evaluator.transform_to_ntt_inplace(prod[idx]);
      // re-use the RLWEPt array
      this_decryptor->decrypt(prod[idx], rnd_masks[idx]);
      InvNttInplace(rnd_masks[idx], this_context);
    }

    // r_b := M_a * v_b + r_a
    DISPATCH_ALL_FIELDS(field, "Dot-2", [&]() {
      auto xans_mat = xt_mutable_adapt<ring2k_t>(ans_mat);
      xans_mat = xans_mat.reshape({loop_dim, lhs_nrows});
      xt::row(xans_mat, n) = xt_adapt<ring2k_t>(
          matvec_prot.ParseMatVecResult(field, meta, rnd_masks));
    });
  }  // end loop_dim

  if (M == lhs_nrows) {
    // A = lhs_mat
    // B = rhs_mat^T
    // C = ans_mat^T
    TransposeInplace(rhs_mat, loop_dim, K);
    TransposeInplace(ans_mat, loop_dim, lhs_nrows);
    TransposeInplace(mask_mat, loop_dim, lhs_nrows);
  } else {
    // A = rhs_mat
    // B = lhs_mat^T,
    // C = ans_mat
    TransposeInplace(lhs_mat, lhs_nrows, K);
    std::swap(lhs_mat, rhs_mat);
  }

  ans_mat = ring_add(ans_mat,
                     ring_sub(ring_mmul(lhs_mat, rhs_mat, M, N, K), mask_mat));

  return {lhs_mat, rhs_mat, ans_mat};
}
BeaverCheetah::BeaverCheetah(std::shared_ptr<yasl::link::Context> lctx)
    : mul_impl_(std::make_shared<MulImpl>(lctx)),
      dot_impl_(std::make_shared<DotImpl>(lctx)) {
  ot_primitives_ = std::make_shared<spu::CheetahPrimitives>(lctx);
}

Beaver::Triple BeaverCheetah::Mul(FieldType field, size_t size) {
  yasl::CheckNotNull(mul_impl_.get());
  return mul_impl_->Mul(field, size);
}

ArrayRef BeaverCheetah::MulAShr(const ArrayRef &shr, yasl::link::Context *conn,
                                bool evaluator) {
  yasl::CheckNotNull(mul_impl_.get());
  return mul_impl_->MulAShr(shr, conn, evaluator);
}

Beaver::Triple BeaverCheetah::Dot(FieldType field, size_t M, size_t N,
                                  size_t K) {
  yasl::CheckNotNull(dot_impl_.get());
  return dot_impl_->Dot(field, M, N, K);
}

Beaver::Triple BeaverCheetah::And(FieldType field, size_t size) {
  yasl::CheckNotNull(ot_primitives_.get());

  ArrayRef a(makeType<RingTy>(field), size);
  ArrayRef b(makeType<RingTy>(field), size);
  ArrayRef c(makeType<RingTy>(field), size);

  ot_primitives_->nonlinear()->beaver_triple(
      (uint8_t *)a.data(), (uint8_t *)b.data(), (uint8_t *)c.data(),
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
