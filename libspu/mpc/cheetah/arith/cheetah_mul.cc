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

#include <functional>
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
#include "yacl/link/link.h"
#include "yacl/utils/parallel.h"

#include "libspu/mpc/cheetah/arith/common.h"
#include "libspu/mpc/cheetah/rlwe/modswitch_helper.h"
#include "libspu/mpc/cheetah/rlwe/utils.h"
#include "libspu/mpc/utils/ring_ops.h"

struct Options {
  size_t ring_bitlen;
  size_t msg_bitlen;  // msg_bitlen <= ring_bitlen
};

bool operator==(const Options &lhs, const Options &rhs) {
  return lhs.ring_bitlen == rhs.ring_bitlen && lhs.msg_bitlen == rhs.msg_bitlen;
}

template <>
struct std::hash<Options> {
  size_t operator()(Options const &s) const noexcept {
    return std::hash<std::string>{}(
        fmt::format("{}_{}", s.ring_bitlen, s.msg_bitlen));
  }
};

namespace spu::mpc::cheetah {

struct CheetahMul::Impl : public EnableCPRNG {
 public:
  // RLWE parameters: N = 4096, Q \approx 2^{109}, t \approx 2^{40}
  // NOTE(lwj): Under this parameters, the Mul() might introduce 1-bit error
  // within a small chance Pr = 2^{-32}.
  static constexpr size_t kPolyDegree = 4096;
  static constexpr size_t kCipherModulusBits = 109;
  static constexpr int kNoiseFloodRandomBits = 50;
  static constexpr int64_t kCtAsyncParallel = 16;

  const uint32_t small_crt_prime_len_;

  explicit Impl(std::shared_ptr<yacl::link::Context> lctx,
                bool allow_high_prob_one_bit_error)
      : small_crt_prime_len_(allow_high_prob_one_bit_error ? 45 : 42),
        lctx_(std::move(lctx)),
        allow_high_prob_one_bit_error_(allow_high_prob_one_bit_error) {
    parms_ = DecideSEALParameters();
  }

  ~Impl() = default;

  constexpr size_t OLEBatchSize() const { return kPolyDegree; }

  int Rank() const { return lctx_->Rank(); }

  static seal::EncryptionParameters DecideSEALParameters() {
    size_t poly_deg = kPolyDegree;
    auto scheme_type = seal::scheme_type::bfv;
    auto parms = seal::EncryptionParameters(scheme_type);
    std::vector<int> modulus_bits;
    // NOTE(lwj): We set the 2nd modulus a bit larger than
    // `kSmallPrimeBitLen`. We will drop the 2nd modulus during the H2A step.
    // Also, it helps reducing the noise in the BFV ciphertext.
    // Slightly larger than the recommand 109bit modulus in SEAL.
    modulus_bits = {60, 52};
    parms.set_use_special_prime(false);
    parms.set_poly_modulus_degree(poly_deg);
    parms.set_coeff_modulus(seal::CoeffModulus::Create(poly_deg, modulus_bits));
    return parms;
  }

  int64_t num_slots() const { return parms_.poly_modulus_degree(); }

  void LazyExpandSEALContexts(const Options &options,
                              yacl::link::Context *conn = nullptr);

  NdArrayRef MulOLE(const NdArrayRef &shr, yacl::link::Context *conn,
                    bool evaluator, uint32_t msg_width_hint);

 protected:
  void LocalExpandSEALContexts(size_t target);

  inline uint32_t TotalCRTBitLen(const Options &options) const {
    // Let P be the product of small CRT primes.
    // When P > 2^{2k - 1}, the Pr(1-bit error) is about 2^{2k-4}/P.
    // If allowing high prob of 1-bit error, we just set P ~ 2^{2k}
    // Otherwise we set P ~ 2^{2k + 37} so that Pr(1-bit error) is about
    // 2^{-40}.
    auto bits = options.msg_bitlen + options.ring_bitlen +
                (allow_high_prob_one_bit_error_ ? 4UL : 37UL);
    auto nprimes = CeilDiv<size_t>(bits, small_crt_prime_len_);
    nprimes = std::min(7UL, nprimes);  // Slightly reduce the margin for FM128
    return nprimes * small_crt_prime_len_;
  }

  void LazyInitModSwitchHelper(const Options &options);

  inline uint32_t WorkingContextSize(const Options &options) const {
    uint32_t target_bitlen = TotalCRTBitLen(options);
    SPU_ENFORCE(target_bitlen <= current_crt_plain_bitlen_,
                "Call LazyExpandSEALContexts first");
    return CeilDiv(target_bitlen, small_crt_prime_len_);
  }

  void EncodeArray(const NdArrayRef &array, bool need_encrypt,
                   const Options &options, absl::Span<RLWEPt> out);

  void EncodeArray(const NdArrayRef &array, bool need_encrypt,
                   const Options &options, std::vector<RLWEPt> *out) {
    int64_t num_elts = array.numel();
    auto eltype = array.eltype();
    SPU_ENFORCE(num_elts > 0, "empty array");
    SPU_ENFORCE(eltype.isa<RingTy>(), "array must be ring_type, got={}",
                eltype);

    int64_t num_splits = CeilDiv(num_elts, num_slots());
    int64_t num_seal_ctx = WorkingContextSize(options);
    int64_t num_polys = num_seal_ctx * num_splits;
    out->resize(num_polys);
    absl::Span<RLWEPt> wrap(out->data(), out->size());
    EncodeArray(array, need_encrypt, options, wrap);
  }

  // out = EncodeArray(array) if out is not null
  // return the payload size (absl::Buffer)
  size_t EncryptArrayThenSend(const NdArrayRef &array, const Options &options,
                              yacl::link::Context *conn = nullptr);

  // Sample random array `r` of `size` elements in the field.
  // Then compute ciphers*plains + r and response the result to the peer.
  // Return teh sampled array `r`.
  void MulThenResponse(FieldType field, int64_t num_elts,
                       const Options &options,
                       absl::Span<const yacl::Buffer> ciphers,
                       absl::Span<const RLWEPt> plains,
                       absl::Span<const RLWEPt> rnd_mask,
                       yacl::link::Context *conn = nullptr);

  NdArrayRef PrepareRandomMask(FieldType field, int64_t size,
                               const Options &options,
                               std::vector<RLWEPt> *encoded_mask);

  NdArrayRef DecryptArray(FieldType field, int64_t size, const Options &options,
                          const std::vector<yacl::Buffer> &ct_array);

  void NoiseFloodCiphertext(RLWECt &ct, const seal::SEALContext &context);

  void RandomizeCipherForDecryption(RLWECt &ct, size_t context_id);

 private:
  std::shared_ptr<yacl::link::Context> lctx_;

  bool allow_high_prob_one_bit_error_ = false;

  seal::EncryptionParameters parms_;

  uint32_t current_crt_plain_bitlen_{0};

  // SEAL's contexts for ZZ_{2^k}
  mutable std::shared_mutex context_lock_;
  std::vector<seal::SEALContext> seal_cntxts_;

  // own secret key
  std::shared_ptr<seal::SecretKey> secret_key_;
  // the public key received from the opposite party
  std::shared_ptr<seal::PublicKey> pair_public_key_;

  std::unordered_map<Options, ModulusSwitchHelper> ms_helpers_;

  std::vector<std::shared_ptr<seal::Encryptor>> sym_encryptors_;
  std::vector<std::shared_ptr<seal::Decryptor>> decryptors_;
  std::vector<std::shared_ptr<seal::Encryptor>> pk_encryptors_;
  std::vector<std::shared_ptr<seal::BatchEncoder>> bfv_encoders_;
};

void CheetahMul::Impl::LazyInitModSwitchHelper(const Options &options) {
  std::unique_lock<std::shared_mutex> guard(context_lock_);
  if (ms_helpers_.count(options) > 0) {
    return;
  }

  uint32_t target_plain_bitlen = TotalCRTBitLen(options);
  SPU_ENFORCE(current_crt_plain_bitlen_ >= target_plain_bitlen);
  std::vector<seal::Modulus> crt_modulus;

  uint32_t accum_plain_bitlen = 0;
  for (size_t idx = 0; accum_plain_bitlen < target_plain_bitlen; ++idx) {
    auto crt_moduli =
        seal_cntxts_[idx].key_context_data()->parms().plain_modulus();
    accum_plain_bitlen += crt_moduli.bit_count();
    crt_modulus.push_back(crt_moduli);
  }

  // NOTE(lwj): we use ckks for this crt_context
  auto parms = seal::EncryptionParameters(seal::scheme_type::ckks);
  parms.set_poly_modulus_degree(parms_.poly_modulus_degree());
  parms.set_coeff_modulus(crt_modulus);

  seal::SEALContext crt_context(parms, false, seal::sec_level_type::none);

  ms_helpers_.emplace(options,
                      ModulusSwitchHelper(crt_context, options.ring_bitlen));
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

void CheetahMul::Impl::LazyExpandSEALContexts(const Options &options,
                                              yacl::link::Context *conn) {
  uint32_t target_plain_bitlen = TotalCRTBitLen(options);
  std::unique_lock<std::shared_mutex> guard(context_lock_);
  if (current_crt_plain_bitlen_ >= target_plain_bitlen) {
    return;
  }

  uint32_t num_seal_ctx = CeilDiv(target_plain_bitlen, small_crt_prime_len_);
  std::vector<int> crt_moduli_bits(num_seal_ctx, small_crt_prime_len_);
  std::vector<seal::Modulus> crt_modulus =
      seal::CoeffModulus::Create(parms_.poly_modulus_degree(), crt_moduli_bits);
  // Sort the primes to make sure new primes are placed in the back
  std::sort(crt_modulus.begin(), crt_modulus.end(),
            [](const seal::Modulus &p, const seal::Modulus &q) {
              return p.value() > q.value();
            });

  uint32_t current_num_ctx = seal_cntxts_.size();
  for (uint32_t i = current_num_ctx; i < num_seal_ctx; ++i) {
    uint64_t new_plain_modulus = crt_modulus.at(i).value();
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
    seal_cntxts_.emplace_back(parms_, true, seal::sec_level_type::none);

    if (idx == 0) {
      seal::KeyGenerator keygen(seal_cntxts_[0]);
      secret_key_ = std::make_shared<seal::SecretKey>(keygen.secret_key());

      auto pk = keygen.create_public_key();
      // NOTE(lwj): we patched seal/util/serializable.h
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
  SPDLOG_INFO("CheetahMul uses {} modulus for {} bit input over {} bit ring",
              num_seal_ctx, options.msg_bitlen, options.ring_bitlen);
}

NdArrayRef CheetahMul::Impl::MulOLE(const NdArrayRef &shr,
                                    yacl::link::Context *conn, bool evaluator,
                                    uint32_t msg_width_hint) {
  if (conn == nullptr) {
    conn = lctx_.get();
  }

  auto eltype = shr.eltype();
  SPU_ENFORCE(eltype.isa<RingTy>(), "must be ring_type, got={}", eltype);
  SPU_ENFORCE(shr.shape().size() == 1, "need 1D Array");
  SPU_ENFORCE(shr.numel() > 0);

  auto field = eltype.as<Ring2k>()->field();
  Options options;
  options.ring_bitlen = SizeOf(field) * 8;
  options.msg_bitlen =
      msg_width_hint == 0 ? options.ring_bitlen : msg_width_hint;
  SPU_ENFORCE(options.msg_bitlen > 0 &&
              options.msg_bitlen <= options.ring_bitlen);
  LazyExpandSEALContexts(options, conn);
  LazyInitModSwitchHelper(options);

  size_t numel = shr.numel();
  int nxt_rank = conn->NextRank();
  std::vector<RLWEPt> encoded_shr;

  if (evaluator) {
    // NOTE(lwj):
    // 1. Alice & Bob enode local share to polynomials
    // 2. Alice send ciphertexts to Bob
    // 3. Bob multiply
    // 4. Bob samples polys for random masking
    // 5. Bob response the masked ciphertext to Alice
    // We can overlap Step 2 (IO) and Step 4 (local computation)
    EncodeArray(shr, false, options, &encoded_shr);
    size_t payload_sze = encoded_shr.size();
    std::vector<yacl::Buffer> recv_ct(payload_sze);
    auto io_task = std::async(std::launch::async, [&]() {
      for (size_t idx = 0; idx < payload_sze; ++idx) {
        recv_ct[idx] = conn->Recv(nxt_rank, "");
      }
    });

    std::vector<RLWEPt> encoded_mask;
    auto random_mask =
        PrepareRandomMask(field, shr.numel(), options, &encoded_mask);
    SPU_ENFORCE(encoded_mask.size() == payload_sze,
                "BeaverCheetah: random mask poly size mismatch");
    // wait for IO
    io_task.get();
    MulThenResponse(field, numel, options, recv_ct, encoded_shr,
                    absl::MakeConstSpan(encoded_mask), conn);
    return random_mask;
  }

  size_t payload_sze = EncryptArrayThenSend(shr, options, conn);
  std::vector<yacl::Buffer> recv_ct(payload_sze);
  for (size_t idx = 0; idx < payload_sze; ++idx) {
    recv_ct[idx] = conn->Recv(nxt_rank, "");
  }
  return DecryptArray(field, numel, options, recv_ct);
}

size_t CheetahMul::Impl::EncryptArrayThenSend(const NdArrayRef &array,
                                              const Options &options,
                                              yacl::link::Context *conn) {
  int64_t num_elts = array.numel();
  auto eltype = array.eltype();
  SPU_ENFORCE(num_elts > 0, "empty array");
  SPU_ENFORCE(eltype.isa<RingTy>(), "array must be ring_type, got={}", eltype);

  int64_t num_splits = CeilDiv(num_elts, num_slots());
  int64_t num_seal_ctx = WorkingContextSize(options);
  int64_t num_polys = num_seal_ctx * num_splits;

  std::vector<RLWEPt> encoded_array(num_polys);
  EncodeArray(array, /*scale_up*/ true, options, absl::MakeSpan(encoded_array));

  std::vector<yacl::Buffer> payload(num_polys);

  yacl::parallel_for(0, num_polys, CalculateWorkLoad(num_polys),
                     [&](int64_t job_bgn, int64_t job_end) {
                       for (int64_t job_id = job_bgn; job_id < job_end;
                            ++job_id) {
                         int64_t cntxt_id = job_id / num_splits;
                         auto ct = sym_encryptors_[cntxt_id]->encrypt_symmetric(
                             encoded_array.at(job_id));
                         payload.at(job_id) = EncodeSEALObject(ct.obj());
                       }
                     });

  if (conn == nullptr) {
    conn = lctx_.get();
  }

  int nxt_rank = conn->NextRank();
  for (int64_t i = 0; i < num_polys; i += kCtAsyncParallel) {
    int64_t this_batch = std::min(num_polys - i, kCtAsyncParallel);
    conn->Send(nxt_rank, payload[i],
               fmt::format("CheetahMul::Send ct[{}] to rank={}", i, nxt_rank));
    for (int64_t j = 1; j < this_batch; ++j) {
      conn->SendAsync(
          nxt_rank, payload[i + j],
          fmt::format("CheetahMul::Send ct[{}] to rank={}", i + j, nxt_rank));
    }
  }
  return payload.size();
}

NdArrayRef CheetahMul::Impl::PrepareRandomMask(
    FieldType field, int64_t size, const Options &options,
    std::vector<RLWEPt> *encoded_mask) {
  const int64_t num_splits = CeilDiv(size, num_slots());
  const int64_t num_seal_ctx = WorkingContextSize(options);
  const int64_t num_polys = num_seal_ctx * num_splits;
  SPU_ENFORCE(ms_helpers_.count(options) > 0);
  encoded_mask->resize(num_polys);

  // sample r from [0, P) in the RNS format
  // Ref: the one-bit approximate re-sharing in Cheetah's paper (eprint ver).
  std::vector<uint64_t> random_rns(size * num_seal_ctx);
  for (int64_t cidx = 0; cidx < num_seal_ctx; ++cidx) {
    const auto &plain_mod =
        seal_cntxts_[cidx].key_context_data()->parms().plain_modulus();
    int64_t offset = cidx * num_splits;
    std::vector<uint64_t> u64tmp(num_slots(), 0);

    for (int64_t j = 0; j < num_splits; ++j) {
      int64_t bgn = j * num_slots();
      int64_t end = std::min(bgn + num_slots(), size);
      int64_t len = end - bgn;

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
  auto &ms_helper = ms_helpers_.find(options)->second;
  absl::Span<const uint64_t> inp(random_rns.data(), random_rns.size());
  return ms_helper.ModulusDownRNS(field, {size}, inp);
}

void CheetahMul::Impl::EncodeArray(const NdArrayRef &array, bool need_encrypt,
                                   const Options &options,
                                   absl::Span<RLWEPt> out) {
  int64_t num_elts = array.numel();
  auto eltype = array.eltype();
  SPU_ENFORCE(num_elts > 0, "empty array");
  SPU_ENFORCE(array.shape().size() == 1, "need 1D array");
  SPU_ENFORCE(eltype.isa<RingTy>(), "array must be ring_type, got={}", eltype);

  int64_t num_splits = CeilDiv(num_elts, num_slots());
  int64_t num_seal_ctx = WorkingContextSize(options);
  int64_t num_polys = num_seal_ctx * num_splits;
  SPU_ENFORCE_EQ(out.size(), (size_t)num_polys,
                 "out size mismatch, expect={}, got={}size", num_polys,
                 out.size());
  SPU_ENFORCE(ms_helpers_.count(options) > 0);

  auto &ms_helper = ms_helpers_.find(options)->second;

  yacl::parallel_for(
      0, num_polys, CalculateWorkLoad(num_polys),
      [&](int64_t job_bgn, int64_t job_end) {
        std::vector<uint64_t> _u64tmp(num_slots());
        auto u64tmp = absl::MakeSpan(_u64tmp);

        for (int64_t job_id = job_bgn; job_id < job_end; ++job_id) {
          int64_t cntxt_id = job_id / num_splits;
          int64_t split_id = job_id % num_splits;
          int64_t slice_bgn = split_id * num_slots();
          int64_t slice_end =
              std::min(num_elts, slice_bgn + static_cast<int64_t>(num_slots()));

          auto slice = array.slice({slice_bgn}, {slice_end}, {1});
          auto dst = u64tmp.subspan(0, slice_end - slice_bgn);
          if (need_encrypt) {
            ms_helper.ModulusUpAt(slice, cntxt_id, dst);
          } else {
            ms_helper.CenteralizeAt(slice, cntxt_id, dst);
          }
          // zero-padding the rest
          std::fill_n(u64tmp.data() + slice.numel(),
                      u64tmp.size() - slice.numel(), 0);

          CATCH_SEAL_ERROR(
              bfv_encoders_[cntxt_id]->encode(_u64tmp, out[job_id]));
        }
      });
}

void CheetahMul::Impl::MulThenResponse(FieldType, int64_t num_elts,
                                       const Options &options,
                                       absl::Span<const yacl::Buffer> ciphers,
                                       absl::Span<const RLWEPt> plains,
                                       absl::Span<const RLWEPt> ecd_random,
                                       yacl::link::Context *conn) {
  SPU_ENFORCE(!ciphers.empty(), "CheetahMul: empty cipher");
  SPU_ENFORCE(plains.size() == ciphers.size(),
              "CheetahMul: ct/pt size mismatch");

  const int64_t num_splits = CeilDiv(num_elts, num_slots());
  const int64_t num_seal_ctx = WorkingContextSize(options);
  const int64_t num_ciphers = num_seal_ctx * num_splits;
  SPU_ENFORCE(ciphers.size() == (size_t)num_ciphers,
              "CheetahMul : expect {} != {}", num_ciphers, ciphers.size());
  SPU_ENFORCE(ecd_random.size() == (size_t)num_ciphers,
              "CheetahMul: encoded rnaomd size mismatch");

  std::vector<yacl::Buffer> response(num_ciphers);
  yacl::parallel_for(
      0, num_ciphers, CalculateWorkLoad(num_ciphers),
      [&](int64_t job_bgn, int64_t job_end) {
        RLWECt ct;
        std::vector<uint64_t> u64tmp(num_slots(), 0);
        for (int64_t job_id = job_bgn; job_id < job_end; ++job_id) {
          int64_t cntxt_id = job_id / num_splits;
          // int64_t offset = cntxt_id * num_splits;
          const auto &seal_cntxt = seal_cntxts_[cntxt_id];
          seal::Evaluator evaluator(seal_cntxt);
          // Multiply-then-H2A
          DecodeSEALObject(ciphers.at(job_id), seal_cntxt, &ct);
          // Multiply step
          CATCH_SEAL_ERROR(
              evaluator.multiply_plain_inplace(ct, plains[job_id]));
          // H2A
          CATCH_SEAL_ERROR(evaluator.sub_plain_inplace(ct, ecd_random[job_id]));
          // re-randomize the ciphertext (e.g., noise flood)
          RandomizeCipherForDecryption(ct, cntxt_id);
          response[job_id] = EncodeSEALObject(ct);
        }
      });

  if (conn == nullptr) {
    conn = lctx_.get();
  }

  int nxt_rank = conn->NextRank();
  for (int64_t i = 0; i < num_ciphers; i += kCtAsyncParallel) {
    int64_t this_batch = std::min(num_ciphers - i, kCtAsyncParallel);
    conn->Send(nxt_rank, response[i],
               fmt::format("MulThenResponse ct[{}] to rank{}", i, nxt_rank));
    for (int64_t j = 1; j < this_batch; ++j) {
      conn->SendAsync(
          nxt_rank, response[i + j],
          fmt::format("MulThenResponse ct[{}] to rank{}", i + j, nxt_rank));
    }
  }
}

NdArrayRef CheetahMul::Impl::DecryptArray(
    FieldType field, int64_t size, const Options &options,
    const std::vector<yacl::Buffer> &ct_array) {
  const int64_t num_splits = CeilDiv(size, num_slots());
  const int64_t num_seal_ctx = WorkingContextSize(options);
  const int64_t num_ciphers = num_seal_ctx * num_splits;
  SPU_ENFORCE(ct_array.size() == (size_t)num_ciphers,
              "CheetahMul: cipher size mismatch");
  SPU_ENFORCE(ms_helpers_.count(options) > 0);

  // Decrypt ciphertexts into size x num_modulus
  // Then apply the ModulusDown to get value in Z_{2^k}.
  std::vector<uint64_t> rns_temp(size * num_seal_ctx, 0);
  yacl::parallel_for(
      0, num_ciphers, CalculateWorkLoad(num_ciphers),
      [&](int64_t job_bgn, int64_t job_end) {
        RLWEPt pt;
        RLWECt ct;
        std::vector<uint64_t> subarray(num_slots(), 0);
        for (int64_t job_id = job_bgn; job_id < job_end; ++job_id) {
          int64_t cntxt_id = job_id / num_splits;
          int64_t split_id = job_id % num_splits;

          DecodeSEALObject(ct_array.at(job_id), seal_cntxts_[cntxt_id], &ct);
          CATCH_SEAL_ERROR(decryptors_[cntxt_id]->decrypt(ct, pt));
          CATCH_SEAL_ERROR(bfv_encoders_[cntxt_id]->decode(pt, subarray));

          int64_t slice_bgn = split_id * num_slots();
          int64_t slice_end = std::min(size, slice_bgn + num_slots());
          int64_t slice_modulus_bgn = cntxt_id * size + slice_bgn;
          std::copy_n(subarray.data(), slice_end - slice_bgn,
                      rns_temp.data() + slice_modulus_bgn);
        }
      });

  auto &ms_helper = ms_helpers_.find(options)->second;
  return ms_helper.ModulusDownRNS(field, {size}, absl::MakeSpan(rns_temp));
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
  NdArrayView<uint64_t> xrandom(random);
  pforeach(0, xrandom.numel(), [&](int64_t i) { xrandom[i] &= range_mask; });
  // add random to each modulus of ct.data(0)
  uint64_t *dst_ptr = ct.data(0);
  for (size_t l = 0; l < num_modulus; ++l) {
    using namespace seal::util;
    if (modulus[l].bit_count() > kNoiseFloodRandomBits) {
      // When prime[l] > 2^{kNoiseFloodRandomBits} then we add directly add
      // the random
      add_poly_coeffmod(&xrandom[0], dst_ptr, num_coeffs, modulus[l], dst_ptr);
    } else {
      // When prime[l] < 2^{kNoiseFloodRandomBits} we need to compute modulo
      // first.
      std::vector<uint64_t> tmp(num_coeffs);
      modulo_poly_coeffs(&xrandom[0], num_coeffs, modulus[l], tmp.data());
      add_poly_coeffmod(tmp.data(), dst_ptr, num_coeffs, modulus[l], dst_ptr);
    }
    dst_ptr += num_coeffs;
  }
}

void CheetahMul::Impl::RandomizeCipherForDecryption(RLWECt &ct,
                                                    size_t context_id) {
  auto &seal_cntxt = seal_cntxts_.at(context_id);
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
  CATCH_SEAL_ERROR(
      pk_encryptors_[context_id]->encrypt_zero(ct.parms_id(), zero_enc));
  CATCH_SEAL_ERROR(evaluator.add_inplace(ct, zero_enc));

  // 4. Truncate for smaller communication
  TruncateBFVForDecryption(ct, seal_cntxt);
}

CheetahMul::CheetahMul(std::shared_ptr<yacl::link::Context> lctx,
                       bool allow_high_prob_one_bit_error) {
  impl_ = std::make_unique<Impl>(lctx, allow_high_prob_one_bit_error);
}

CheetahMul::~CheetahMul() = default;

int CheetahMul::Rank() const { return impl_->Rank(); }

size_t CheetahMul::OLEBatchSize() const {
  SPU_ENFORCE(impl_ != nullptr);
  return impl_->OLEBatchSize();
}

NdArrayRef CheetahMul::MulOLE(const NdArrayRef &inp, yacl::link::Context *conn,
                              bool is_evaluator, uint32_t msg_width_hint) {
  SPU_ENFORCE(impl_ != nullptr);
  SPU_ENFORCE(conn != nullptr);
  return impl_->MulOLE(inp, conn, is_evaluator, msg_width_hint);
}

NdArrayRef CheetahMul::MulOLE(const NdArrayRef &inp, bool is_evaluator,
                              uint32_t msg_width_hint) {
  SPU_ENFORCE(impl_ != nullptr);
  return impl_->MulOLE(inp, nullptr, is_evaluator, msg_width_hint);
}

}  // namespace spu::mpc::cheetah