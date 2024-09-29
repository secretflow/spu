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
#include "seal/util/polyarithsmallmod.h"
#include "seal/valcheck.h"
#include "spdlog/spdlog.h"
#include "yacl/link/link.h"
#include "yacl/utils/parallel.h"

#include "libspu/mpc/cheetah/arith/common.h"
#include "libspu/mpc/cheetah/arith/simd_mul_prot.h"
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
  static constexpr size_t kPolyDegree = 8192;
  static constexpr size_t kCipherModulusBits = 145;
  static constexpr int64_t kCtAsyncParallel = 16;

  const uint32_t small_crt_prime_len_;

  explicit Impl(std::shared_ptr<yacl::link::Context> lctx,
                bool allow_high_prob_one_bit_error)
      : small_crt_prime_len_(allow_high_prob_one_bit_error ? 47 : 45),
        lctx_(std::move(lctx)),
        allow_high_prob_one_bit_error_(allow_high_prob_one_bit_error) {
    parms_ = DecideSEALParameters();
  }

  ~Impl() = default;

  constexpr size_t OLEBatchSize() const { return kPolyDegree; }

  int Rank() const { return lctx_->Rank(); }

  seal::EncryptionParameters DecideSEALParameters() {
    size_t poly_deg = kPolyDegree;
    auto scheme_type = seal::scheme_type::bfv;
    auto parms = seal::EncryptionParameters(scheme_type);
    std::vector<int> modulus_bits;
    if (allow_high_prob_one_bit_error_) {
      modulus_bits = {60, 32, 56};
    } else {
      modulus_bits = {60, 32, 52};
    }
    parms.set_use_special_prime(false);
    parms.set_poly_modulus_degree(poly_deg);
    parms.set_coeff_modulus(seal::CoeffModulus::Create(poly_deg, modulus_bits));
    return parms;
  }

  int64_t num_slots() const { return parms_.poly_modulus_degree(); }

  void LazyInit(size_t field, uint32_t msg_width_hint) {
    Options options;
    options.ring_bitlen = SizeOf(field) * 8;
    options.msg_bitlen =
        msg_width_hint == 0 ? options.ring_bitlen : msg_width_hint;
    LazyExpandSEALContexts(options);
    LazyInitModSwitchHelper(options);
  }

  void LazyExpandSEALContexts(const Options &options,
                              yacl::link::Context *conn = nullptr);

  MemRef MulOLE(const MemRef &shr, yacl::link::Context *conn, bool evaluator,
                uint32_t msg_width_hint);

  MemRef MulShare(const MemRef &xshr, const MemRef &yshr,
                  yacl::link::Context *conn, bool evaluator,
                  uint32_t msg_width_hint);

 protected:
  void LocalExpandSEALContexts(size_t target);

  inline uint32_t TotalCRTBitLen(const Options &options) const {
    auto bits = options.msg_bitlen + options.ring_bitlen +
                (allow_high_prob_one_bit_error_ ? 4UL : 32UL);
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

  void EncodeArray(const MemRef &array, bool need_encrypt,
                   const Options &options, absl::Span<RLWEPt> out);

  void EncodeArray(const MemRef &array, bool need_encrypt,
                   const Options &options, std::vector<RLWEPt> *out) {
    int64_t num_elts = array.numel();
    auto eltype = array.eltype();
    SPU_ENFORCE(num_elts > 0, "empty array");
    SPU_ENFORCE(eltype.isa<BaseRingType>(), "array must be ring_type, got={}",
                eltype);

    int64_t num_splits = CeilDiv(num_elts, num_slots());
    int64_t num_seal_ctx = WorkingContextSize(options);
    int64_t num_polys = num_seal_ctx * num_splits;
    out->resize(num_polys);
    absl::Span<RLWEPt> wrap(out->data(), out->size());
    EncodeArray(array, need_encrypt, options, wrap);
  }

  // return the payload size (absl::Buffer)
  size_t EncryptArrayThenSend(const MemRef &array, const Options &options,
                              yacl::link::Context *conn = nullptr);

  // Sample random array `r` of `size` elements in the field.
  // Then compute ciphers*plains + r and response the result to the peer.
  // Return teh sampled array `r`.
  void MulThenResponse(size_t field, int64_t num_elts, const Options &options,
                       absl::Span<const yacl::Buffer> ciphers,
                       absl::Span<const RLWEPt> plains,
                       absl::Span<const uint64_t> rnd_mask,
                       yacl::link::Context *conn = nullptr);

  // Enc(x0) * y1 + Enc(y0) * x1 + rand_mask
  void FMAThenResponse(size_t field, int64_t num_elts, const Options &options,
                       absl::Span<const yacl::Buffer> ciphers_x0,
                       absl::Span<const yacl::Buffer> ciphers_y0,
                       absl::Span<const RLWEPt> plains_x1,
                       absl::Span<const RLWEPt> plains_y1,
                       absl::Span<const uint64_t> rnd_mask,
                       yacl::link::Context *conn = nullptr);

  void PrepareRandomMask(size_t field, int64_t size, const Options &options,
                         std::vector<uint64_t> &mask);

  MemRef DecryptArray(size_t field, int64_t size, const Options &options,
                      const std::vector<yacl::Buffer> &ct_array);

 private:
  std::shared_ptr<yacl::link::Context> lctx_;

  bool allow_high_prob_one_bit_error_ = false;

  seal::EncryptionParameters parms_;

  uint32_t current_crt_plain_bitlen_{0};

  // SEAL's contexts for ZZ_{2^k}
  std::vector<seal::SEALContext> seal_cntxts_;

  std::vector<std::shared_ptr<SIMDMulProt>> simd_mul_instances_;

  // own secret key
  std::shared_ptr<seal::SecretKey> secret_key_;
  // the public key received from the opposite party
  std::shared_ptr<seal::PublicKey> peer_pub_key_;

  std::unordered_map<Options, ModulusSwitchHelper> ms_helpers_;

  std::vector<std::shared_ptr<seal::Decryptor>> decryptors_;
};

void CheetahMul::Impl::LazyInitModSwitchHelper(const Options &options) {
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
  SPU_ENFORCE(decryptors_.size() == target);

  seal::SecretKey sk;
  sk.data().resize(secret_key_->data().coeff_count());
  std::copy_n(secret_key_->data().data(), secret_key_->data().coeff_count(),
              sk.data().data());
  sk.parms_id() = seal_cntxts_[target].key_parms_id();

  size_t keysze = peer_pub_key_->data().size();
  size_t numel = peer_pub_key_->data().poly_modulus_degree() *
                 peer_pub_key_->data().coeff_modulus_size();

  seal::PublicKey pk;
  pk.data().resize(seal_cntxts_[target], sk.parms_id(), keysze);
  std::copy_n(peer_pub_key_->data().data(), keysze * numel, pk.data().data());
  pk.data().is_ntt_form() = peer_pub_key_->data().is_ntt_form();
  pk.parms_id() = sk.parms_id();

  decryptors_.push_back(
      std::make_shared<seal::Decryptor>(seal_cntxts_[target], sk));
}

void CheetahMul::Impl::LazyExpandSEALContexts(const Options &options,
                                              yacl::link::Context *conn) {
  uint32_t target_plain_bitlen = TotalCRTBitLen(options);
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
      peer_pub_key_ = std::make_shared<seal::PublicKey>();
      DecodeSEALObject(pk_buf_recv, seal_cntxts_[0], peer_pub_key_.get());

      // create the functors
      decryptors_.push_back(
          std::make_shared<seal::Decryptor>(seal_cntxts_[0], *secret_key_));
    } else {
      // For other CRT context, we just copy the sk/pk
      LocalExpandSEALContexts(idx);
    }
    simd_mul_instances_.push_back(
        std::make_shared<SIMDMulProt>(kPolyDegree, crt_modulus[idx].value()));
  }

  current_crt_plain_bitlen_ = target_plain_bitlen;
  SPDLOG_INFO("CheetahMul uses {} modulus for {} bit input over {} bit ring",
              num_seal_ctx, options.msg_bitlen, options.ring_bitlen);
}

MemRef CheetahMul::Impl::MulOLE(const MemRef &shr, yacl::link::Context *conn,
                                bool evaluator, uint32_t msg_width_hint) {
  if (conn == nullptr) {
    conn = lctx_.get();
  }

  auto eltype = shr.eltype();
  SPU_ENFORCE(eltype.isa<BaseRingType>(), "must be ring_type, got={}", eltype);
  SPU_ENFORCE(shr.numel() > 0);

  auto field = SizeOf(eltype.storage_type()) * 8;
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
    // 5. Bob responses the masked ciphertext to Alice
    // We can overlap Step 2 (IO) and Step 4 (local computation)
    EncodeArray(shr, false, options, &encoded_shr);

    size_t payload_sze = encoded_shr.size();
    std::vector<yacl::Buffer> recv_ct(payload_sze);
    auto io_task = std::async(std::launch::async, [&]() {
      for (size_t idx = 0; idx < payload_sze; ++idx) {
        recv_ct[idx] = conn->Recv(nxt_rank, "");
      }
    });

    std::vector<uint64_t> random_share_mask;
    PrepareRandomMask(field, shr.numel(), options, random_share_mask);

    // wait for IO
    io_task.get();
    MulThenResponse(field, numel, options, recv_ct, encoded_shr,
                    absl::MakeConstSpan(random_share_mask), conn);
    // convert x \in [0, P) to [0, 2^k) by round(2^k*x/P)
    auto &ms_helper = ms_helpers_.find(options)->second;
    return ms_helper.ModulusDownRNS(field, shr.shape(), random_share_mask)
        .reshape(shr.shape());
  }

  size_t payload_sze = EncryptArrayThenSend(shr, options, conn);
  std::vector<yacl::Buffer> recv_ct(payload_sze);
  for (size_t idx = 0; idx < payload_sze; ++idx) {
    recv_ct[idx] = conn->Recv(nxt_rank, "");
  }
  return DecryptArray(field, numel, options, recv_ct).reshape(shr.shape());
}

MemRef CheetahMul::Impl::MulShare(const MemRef &xshr, const MemRef &yshr,
                                  yacl::link::Context *conn, bool evaluator,
                                  uint32_t msg_width_hint) {
  if (conn == nullptr) {
    conn = lctx_.get();
  }

  auto eltype = xshr.eltype();
  SPU_ENFORCE(eltype.isa<RingTy>(), "must be ring_type, got={}", eltype);
  SPU_ENFORCE(yshr.eltype().isa<RingTy>(), "must be ring_type, got={}",
              yshr.eltype());
  SPU_ENFORCE(xshr.numel() > 0);
  SPU_ENFORCE_EQ(xshr.shape(), yshr.shape());

  auto field = SizeOf(eltype.storage_type()) * 8;
  Options options;
  options.ring_bitlen = SizeOf(field) * 8;
  options.msg_bitlen =
      msg_width_hint == 0 ? options.ring_bitlen : msg_width_hint;
  SPU_ENFORCE(options.msg_bitlen > 0 &&
              options.msg_bitlen <= options.ring_bitlen);
  LazyExpandSEALContexts(options, conn);
  LazyInitModSwitchHelper(options);

  size_t numel = xshr.numel();
  int nxt_rank = conn->NextRank();

  // x0*y0 + <x0 + y1 + x1 * y0> + x1 * y1
  if (evaluator) {
    std::vector<RLWEPt> encoded_x0;
    std::vector<RLWEPt> encoded_y0;
    EncodeArray(xshr, false, options, &encoded_x0);
    EncodeArray(yshr, false, options, &encoded_y0);

    size_t payload_sze = encoded_x0.size();
    std::vector<yacl::Buffer> recv_ct_x1(payload_sze);
    std::vector<yacl::Buffer> recv_ct_y1(payload_sze);
    auto io_task = std::async(std::launch::async, [&]() {
      for (size_t idx = 0; idx < payload_sze; ++idx) {
        recv_ct_x1[idx] = conn->Recv(nxt_rank, "");
      }
      for (size_t idx = 0; idx < payload_sze; ++idx) {
        recv_ct_y1[idx] = conn->Recv(nxt_rank, "");
      }
    });

    std::vector<uint64_t> random_share_mask;
    PrepareRandomMask(field, xshr.numel(), options, random_share_mask);

    // wait for IO
    io_task.get();
    FMAThenResponse(field, numel, options, recv_ct_x1, recv_ct_y1, encoded_x0,
                    encoded_y0, absl::MakeConstSpan(random_share_mask), conn);
    // convert x \in [0, P) to [0, 2^k) by round(2^k*x/P)
    auto &ms_helper = ms_helpers_.find(options)->second;
    auto out = ms_helper.ModulusDownRNS(field, xshr.shape(), random_share_mask)
                   .reshape(xshr.shape());
    ring_add_(out, ring_mul(xshr, yshr));
    return out;
  }

  size_t payload_sze = EncryptArrayThenSend(xshr, options, conn);
  (void)EncryptArrayThenSend(yshr, options, conn);
  std::vector<yacl::Buffer> recv_ct(payload_sze);
  for (size_t idx = 0; idx < payload_sze; ++idx) {
    recv_ct[idx] = conn->Recv(nxt_rank, "");
  }
  auto out = DecryptArray(field, numel, options, recv_ct).reshape(xshr.shape());
  ring_add_(out, ring_mul(xshr, yshr));
  return out;
}

size_t CheetahMul::Impl::EncryptArrayThenSend(const MemRef &array,
                                              const Options &options,
                                              yacl::link::Context *conn) {
  int64_t num_elts = array.numel();
  auto eltype = array.eltype();
  SPU_ENFORCE(num_elts > 0, "empty array");
  SPU_ENFORCE(eltype.isa<BaseRingType>(), "array must be ring_type, got={}",
              eltype);

  int64_t num_splits = CeilDiv(num_elts, num_slots());
  int64_t num_seal_ctx = WorkingContextSize(options);
  int64_t num_polys = num_seal_ctx * num_splits;

  std::vector<RLWEPt> encoded_array(num_polys);
  EncodeArray(array, /*scale_up*/ true, options, absl::MakeSpan(encoded_array));

  std::vector<yacl::Buffer> payload(num_polys);

  yacl::parallel_for(0, num_polys, [&](int64_t job_bgn, int64_t job_end) {
    RLWECt ct;
    for (int64_t job_id = job_bgn; job_id < job_end; ++job_id) {
      int64_t cntxt_id = job_id / num_splits;

      simd_mul_instances_[cntxt_id]->SymEncrypt(
          {&encoded_array[job_id], 1}, *secret_key_, seal_cntxts_[cntxt_id],
          /*save_seed*/ true, {&ct, 1});
      payload.at(job_id) = EncodeSEALObject(ct);
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

void CheetahMul::Impl::PrepareRandomMask(size_t field, int64_t size,
                                         const Options &options,
                                         std::vector<uint64_t> &mask) {
  const int64_t num_splits = CeilDiv(size, num_slots());
  const int64_t num_seal_ctx = WorkingContextSize(options);
  SPU_ENFORCE(ms_helpers_.count(options) > 0);
  mask.resize(num_seal_ctx * size);

  // sample r from [0, P) in the RNS format
  // Ref: the one-bit approximate re-sharing in Cheetah's paper (eprint ver).
  auto _mask = absl::MakeSpan(mask);
  for (int64_t cidx = 0; cidx < num_seal_ctx; ++cidx) {
    const auto &plain_mod =
        seal_cntxts_[cidx].key_context_data()->parms().plain_modulus();
    std::vector<uint64_t> u64tmp(num_slots(), 0);

    for (int64_t j = 0; j < num_splits; ++j) {
      int64_t bgn = j * num_slots();
      int64_t len = std::min(num_slots(), size - bgn);

      // sample the RNS component of r from [0, p_i)
      UniformPrime(plain_mod, _mask.subspan(cidx * size + bgn, len));
    }
  }
}

void CheetahMul::Impl::EncodeArray(const MemRef &array, bool need_encrypt,
                                   const Options &options,
                                   absl::Span<RLWEPt> out) {
  int64_t num_elts = array.numel();
  auto eltype = array.eltype();
  SPU_ENFORCE(num_elts > 0, "empty array");
  SPU_ENFORCE(eltype.isa<BaseRingType>(), "array must be ring_type, got={}",
              eltype);

  int64_t num_splits = CeilDiv(num_elts, num_slots());
  int64_t num_seal_ctx = WorkingContextSize(options);
  int64_t num_polys = num_seal_ctx * num_splits;
  SPU_ENFORCE_EQ(out.size(), (size_t)num_polys,
                 "out size mismatch, expect={}, got={}size", num_polys,
                 out.size());
  SPU_ENFORCE(ms_helpers_.count(options) > 0);

  auto &ms_helper = ms_helpers_.find(options)->second;

  yacl::parallel_for(0, num_polys, [&](int64_t job_bgn, int64_t job_end) {
    std::vector<uint64_t> _u64tmp(num_slots());
    auto u64tmp = absl::MakeSpan(_u64tmp);

    MemRef slots(array.eltype(), {num_slots()});

    for (int64_t job_id = job_bgn; job_id < job_end; ++job_id) {
      int64_t cntxt_id = job_id / num_splits;
      int64_t split_id = job_id % num_splits;
      int64_t slice_bgn = split_id * num_slots();
      int64_t slice_end =
          std::min(num_elts, slice_bgn + static_cast<int64_t>(num_slots()));
      int64_t slice_n = slice_end - slice_bgn;

      // take a slice
      for (int64_t _i = 0; _i < slice_n; ++_i) {
        std::memcpy(&slots.at(_i), &array.at(slice_bgn + _i), array.elsize());
      }
      auto dst = u64tmp.subspan(0, slice_n);
      if (need_encrypt) {
        // Compute round(P/2^k * x) mod pi
        ms_helper.ModulusUpAt(slots.slice({0}, {slice_n}, {1}), cntxt_id, dst);
      } else {
        // view x \in [0, 2^k) as [-2^{k-1}, 2^{k-1})
        // Then compute x mod pi.
        ms_helper.CenteralizeAt(slots.slice({0}, {slice_n}, {1}), cntxt_id,
                                dst);
      }
      // zero-padding the rest
      std::fill_n(u64tmp.data() + slice_n, u64tmp.size() - slice_n, 0);

      simd_mul_instances_[cntxt_id]->EncodeSingle(_u64tmp, out[job_id]);
    }
  });
}

void CheetahMul::Impl::MulThenResponse(size_t, int64_t num_elts,
                                       const Options &options,
                                       absl::Span<const yacl::Buffer> ciphers,
                                       absl::Span<const RLWEPt> plains,
                                       absl::Span<const uint64_t> rnd_mask,
                                       yacl::link::Context *conn) {
  SPU_ENFORCE(!ciphers.empty(), "CheetahMul: empty cipher");
  SPU_ENFORCE(plains.size() == ciphers.size(),
              "CheetahMul: ct/pt size mismatch");

  const int64_t num_splits = CeilDiv(num_elts, num_slots());
  const int64_t num_seal_ctx = WorkingContextSize(options);
  const int64_t num_ciphers = num_seal_ctx * num_splits;
  SPU_ENFORCE(ciphers.size() == (size_t)num_ciphers,
              "CheetahMul : expect {} != {}", num_ciphers, ciphers.size());
  SPU_ENFORCE(rnd_mask.size() == (size_t)num_elts * num_seal_ctx,
              "CheetahMul: rnd_mask size mismatch");

  std::vector<yacl::Buffer> response(num_ciphers);
  yacl::parallel_for(0, num_ciphers, [&](int64_t job_bgn, int64_t job_end) {
    RLWECt ct;
    std::vector<uint64_t> u64tmp(num_slots(), 0);
    for (int64_t job_id = job_bgn; job_id < job_end; ++job_id) {
      int64_t cntxt_id = job_id / num_splits;
      int64_t split_id = job_id % num_splits;

      int64_t slice_bgn = split_id * num_slots();
      int64_t slice_n = std::min(num_slots(), num_elts - slice_bgn);
      // offset by context id
      slice_bgn += cntxt_id * num_elts;

      DecodeSEALObject(ciphers[job_id], seal_cntxts_[cntxt_id], &ct);

      // ct <- Re-randomize(ct * pt) - random_mask
      simd_mul_instances_[cntxt_id]->MulThenReshareInplace(
          {&ct, 1}, plains.subspan(job_id, 1),
          rnd_mask.subspan(slice_bgn, slice_n), *peer_pub_key_,
          seal_cntxts_[cntxt_id]);

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

void CheetahMul::Impl::FMAThenResponse(
    size_t, int64_t num_elts, const Options &options,
    absl::Span<const yacl::Buffer> ciphers_x0,
    absl::Span<const yacl::Buffer> ciphers_y0,
    absl::Span<const RLWEPt> plains_x1, absl::Span<const RLWEPt> plains_y1,
    absl::Span<const uint64_t> rnd_mask, yacl::link::Context *conn) {
  SPU_ENFORCE(!ciphers_x0.empty(), "CheetahMul: empty cipher");
  SPU_ENFORCE(!ciphers_y0.empty(), "CheetahMul: empty cipher");
  SPU_ENFORCE_EQ(ciphers_x0.size(), ciphers_y0.size());
  SPU_ENFORCE_EQ(plains_x1.size(), ciphers_x0.size(),
                 "CheetahMul: ct/pt size mismatch");
  SPU_ENFORCE_EQ(plains_y1.size(), ciphers_y0.size(),
                 "CheetahMul: ct/pt size mismatch");

  const int64_t num_splits = CeilDiv(num_elts, num_slots());
  const int64_t num_seal_ctx = WorkingContextSize(options);
  const int64_t num_ciphers = num_seal_ctx * num_splits;
  SPU_ENFORCE(ciphers_x0.size() == (size_t)num_ciphers,
              "CheetahMul : expect {} != {}", num_ciphers, ciphers_x0.size());
  SPU_ENFORCE(rnd_mask.size() == (size_t)num_elts * num_seal_ctx,
              "CheetahMul: rnd_mask size mismatch");

  std::vector<yacl::Buffer> response(num_ciphers);
  yacl::parallel_for(0, num_ciphers, [&](int64_t job_bgn, int64_t job_end) {
    RLWECt ct_x;
    RLWECt ct_y;
    std::vector<uint64_t> u64tmp(num_slots(), 0);
    for (int64_t job_id = job_bgn; job_id < job_end; ++job_id) {
      int64_t cntxt_id = job_id / num_splits;
      int64_t split_id = job_id % num_splits;

      int64_t slice_bgn = split_id * num_slots();
      int64_t slice_n = std::min(num_slots(), num_elts - slice_bgn);
      // offset by context id
      slice_bgn += cntxt_id * num_elts;

      DecodeSEALObject(ciphers_x0[job_id], seal_cntxts_[cntxt_id], &ct_x);
      DecodeSEALObject(ciphers_y0[job_id], seal_cntxts_[cntxt_id], &ct_y);

      // ct_x <- Re-randomize(ct_x * pt_y + ct_y * pt_x) - random_mask
      simd_mul_instances_[cntxt_id]->FMAThenReshareInplace(
          {&ct_x, 1}, {&ct_y, 1}, plains_y1.subspan(job_id, 1),
          plains_x1.subspan(job_id, 1), rnd_mask.subspan(slice_bgn, slice_n),
          *peer_pub_key_, seal_cntxts_[cntxt_id]);

      response[job_id] = EncodeSEALObject(ct_x);
    }
  });

  if (conn == nullptr) {
    conn = lctx_.get();
  }

  int nxt_rank = conn->NextRank();
  for (int64_t i = 0; i < num_ciphers; i += kCtAsyncParallel) {
    int64_t this_batch = std::min(num_ciphers - i, kCtAsyncParallel);
    conn->Send(nxt_rank, response[i],
               fmt::format("FMAThenResponse ct[{}] to rank{}", i, nxt_rank));
    for (int64_t j = 1; j < this_batch; ++j) {
      conn->SendAsync(
          nxt_rank, response[i + j],
          fmt::format("FMAThenResponse ct[{}] to rank{}", i + j, nxt_rank));
    }
  }
}

MemRef CheetahMul::Impl::DecryptArray(
    size_t field, int64_t size, const Options &options,
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
  yacl::parallel_for(0, num_ciphers, [&](int64_t job_bgn, int64_t job_end) {
    RLWEPt pt;
    RLWECt ct;
    std::vector<uint64_t> subarray(num_slots(), 0);
    for (int64_t job_id = job_bgn; job_id < job_end; ++job_id) {
      int64_t cntxt_id = job_id / num_splits;
      int64_t split_id = job_id % num_splits;

      DecodeSEALObject(ct_array.at(job_id), seal_cntxts_[cntxt_id], &ct);
      CATCH_SEAL_ERROR(decryptors_[cntxt_id]->decrypt(ct, pt));

      simd_mul_instances_[cntxt_id]->DecodeSingle(pt, absl::MakeSpan(subarray));

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

MemRef CheetahMul::MulShare(const MemRef &xshr, const MemRef &yshr,
                            yacl::link::Context *conn, bool is_evaluator,
                            uint32_t msg_width_hint) {
  SPU_ENFORCE(impl_ != nullptr);
  SPU_ENFORCE(conn != nullptr);
  return impl_->MulShare(xshr, yshr, conn, is_evaluator, msg_width_hint);
}

MemRef CheetahMul::MulShare(const MemRef &xshr, const MemRef &yshr,
                            bool is_evaluator, uint32_t msg_width_hint) {
  SPU_ENFORCE(impl_ != nullptr);
  return impl_->MulShare(xshr, yshr, nullptr, is_evaluator, msg_width_hint);
}

MemRef CheetahMul::MulOLE(const MemRef &inp, yacl::link::Context *conn,
                          bool is_evaluator, uint32_t msg_width_hint) {
  SPU_ENFORCE(impl_ != nullptr);
  SPU_ENFORCE(conn != nullptr);
  return impl_->MulOLE(inp, conn, is_evaluator, msg_width_hint);
}

MemRef CheetahMul::MulOLE(const MemRef &inp, bool is_evaluator,
                          uint32_t msg_width_hint) {
  SPU_ENFORCE(impl_ != nullptr);
  return impl_->MulOLE(inp, nullptr, is_evaluator, msg_width_hint);
}

void CheetahMul::LazyInitKeys(size_t field, uint32_t msg_width_hint) {
  SPU_ENFORCE(impl_ != nullptr);
  SPU_ENFORCE(msg_width_hint <= SizeOf(field) * 8);
  return impl_->LazyInit(field, msg_width_hint);
}

}  // namespace spu::mpc::cheetah
