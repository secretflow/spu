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
#include "libspu/mpc/cheetah/arith/batch_matmul.h"

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
#include "seal/galoiskeys.h"
#include "seal/keygenerator.h"
#include "seal/publickey.h"
#include "seal/secretkey.h"
#include "seal/util/polyarithsmallmod.h"
#include "seal/valcheck.h"
#include "spdlog/spdlog.h"
#include "yacl/link/link.h"
#include "yacl/utils/parallel.h"
#include "yacl/utils/elapsed_timer.h"

#include "libspu/mpc/cheetah/arith/common.h"
#include "libspu/mpc/cheetah/arith/simd_batchmm_prot.h"
#include "libspu/mpc/cheetah/rlwe/modswitch_helper.h"
#include "libspu/mpc/cheetah/rlwe/packlwes.h"
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

struct BatchMatMul::Impl : public EnableCPRNG {
 public:
  static constexpr size_t kPolyDegree = 8192;
  static constexpr size_t kCipherModulusBits = 145;
  static constexpr int64_t kCtAsyncParallel = 16;

  const uint32_t small_crt_prime_len_;

  explicit Impl(std::shared_ptr<yacl::link::Context> lctx,
                bool allow_high_prob_one_bit_error)
      : small_crt_prime_len_(allow_high_prob_one_bit_error ? 47 : 44),
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
      // modulus_bits = {60, 32, 56};
      modulus_bits = {59, 45, 45, 59};
    } else {
      // modulus_bits = {60, 32, 52};
      modulus_bits = {59, 45, 45, 59};
    }
    parms.set_use_special_prime(true);
    parms.set_poly_modulus_degree(poly_deg);
    parms.set_coeff_modulus(seal::CoeffModulus::Create(poly_deg, modulus_bits));
    // parms.set_coeff_modulus((seal::CoeffModulus::BFVDefault(poly_deg)));
    return parms;
  }

  int64_t num_slots() const { return parms_.poly_modulus_degree(); }

  void LazyInit(FieldType field, uint32_t msg_width_hint) {
    Options options;
    options.ring_bitlen = SizeOf(field) * 8;
    options.msg_bitlen =
        msg_width_hint == 0 ? options.ring_bitlen : msg_width_hint;
    LazyExpandSEALContexts(options);
    LazyInitModSwitchHelper(options);
    // LazyInitGaloisKey();
  }

  // void LazyInitGaloisKey();
  void InitGaloisKey(const Shape4D& dim4);

  void LazyExpandSEALContexts(const Options &options,
                              yacl::link::Context *conn = nullptr);

  NdArrayRef MatMulClient(const NdArrayRef& x, yacl::link::Context* conn, const Shape4D& dim4, 
                          uint32_t msg_width_hint);

  NdArrayRef MatMulServer(const NdArrayRef& x, const NdArrayRef& w,
                          yacl::link::Context* conn, const Shape4D& dim4,
                          uint32_t msg_width_hint); 



 protected:
  // void LocalExpandSEALContexts(size_t target);

  inline uint32_t TotalCRTBitLen(const Options &options) const {
    auto bits = options.msg_bitlen + options.ring_bitlen +
                (allow_high_prob_one_bit_error_ ? 4UL : 32UL);
    // std::cout << "TotalCRTBitLen bits: " << bits << std::endl;
    auto nprimes = CeilDiv<size_t>(bits, small_crt_prime_len_);
    if (options.ring_bitlen == 128) {
      nprimes = std::max(8UL, nprimes);
    }
    // nprimes = std::min(7UL, nprimes);  // Slightly reduce the margin for FM128
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
    SPU_ENFORCE(eltype.isa<Ring2k>(), "array must be ring_type, got={}",
                eltype);

    int64_t num_splits = CeilDiv(num_elts, num_slots());
    int64_t num_seal_ctx = WorkingContextSize(options);
    int64_t num_polys = num_seal_ctx * num_splits;
    out->resize(num_polys);
    absl::Span<RLWEPt> wrap(out->data(), out->size());
    EncodeArray(array, need_encrypt, options, wrap);
  }

  // return the payload size (absl::Buffer)
  size_t EncryptArrayThenSend(const NdArrayRef &array, const Shape4D& dim4, const Options &options,
                              yacl::link::Context *conn = nullptr);

  // Sample random array `r` of `size` elements in the field.
  // Then compute ciphers*plains + r and response the result to the peer.
  // Return teh sampled array `r`.
  void BatchMatMulThenResponse(FieldType field, Shape4D dim4,
                               const Options &options,
                               absl::Span<const RLWECt> ciphers,
                               absl::Span<const RLWEPt> plains,
                               absl::Span<const uint64_t> rnd_mask,
                               yacl::link::Context *conn = nullptr);



  void PrepareRandomMask(FieldType field, int64_t size, const Options &options,
                         std::vector<uint64_t> &mask);

  NdArrayRef DecryptArray(FieldType field, int64_t size, const Options &options,
                          const std::vector<yacl::Buffer> &ct_array);

 private:
  std::shared_ptr<yacl::link::Context> lctx_;

  bool allow_high_prob_one_bit_error_ = false;

  seal::EncryptionParameters parms_;

  uint32_t current_crt_plain_bitlen_{0};

  // SEAL's contexts for ZZ_{2^k}
  std::vector<seal::SEALContext> seal_cntxts_;

  // std::vector<std::shared_ptr<SIMDMulProt>> simd_mul_instances_;
  std::vector<std::shared_ptr<SIMDBatchMMProt>> simd_batchmm_instances_;

  // own secret key
  // std::shared_ptr<seal::SecretKey> secret_key_;
  std::vector<std::shared_ptr<seal::SecretKey>> secret_key_;
  // the public key received from the opposite party
  // std::shared_ptr<seal::PublicKey> peer_pub_key_;
  std::vector<std::shared_ptr<seal::PublicKey>> peer_pub_key_;
  std::vector<std::shared_ptr<seal::GaloisKeys>> peer_gal_key_;

  std::unordered_map<Options, ModulusSwitchHelper> ms_helpers_;

  std::vector<std::shared_ptr<seal::Decryptor>> decryptors_;
};


void BatchMatMul::Impl::LazyInitModSwitchHelper(const Options &options) {
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

void BatchMatMul::Impl::LazyExpandSEALContexts(const Options &options,
                                              yacl::link::Context *conn) {
  uint32_t target_plain_bitlen = TotalCRTBitLen(options);
  if (current_crt_plain_bitlen_ >= target_plain_bitlen) {
    return;
  }

  uint32_t num_seal_ctx = CeilDiv(target_plain_bitlen, small_crt_prime_len_);
  std::vector<int> crt_moduli_bits(num_seal_ctx, small_crt_prime_len_);
  std::vector<seal::Modulus> crt_modulus =
      // seal::CoeffModulus::Create(parms_.poly_modulus_degree(), crt_moduli_bits);
      seal::PlainModulus::Batching(parms_.poly_modulus_degree(), crt_moduli_bits);
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

    // if (idx == 0) {
    if (1) {
      int nxt_rank = conn->NextRank();
      if (nxt_rank == 0) {//server

        // receive sk for debug
        auto sk_buf_recv = conn->Recv(nxt_rank, "rank0 recv sk");
        secret_key_.push_back(std::make_shared<seal::SecretKey>());
        DecodeSEALObject(sk_buf_recv, seal_cntxts_[idx], secret_key_[idx].get());

        // receive pk
        auto pk_buf_recv = conn->Recv(nxt_rank, "rank0 recv pk");
        peer_pub_key_.push_back(std::make_shared<seal::PublicKey>());
        DecodeSEALObject(pk_buf_recv, seal_cntxts_[idx],
                         peer_pub_key_[idx].get());
      
        } else {//client

        seal::KeyGenerator keygen(seal_cntxts_[idx]);
        secret_key_.push_back(std::make_shared<seal::SecretKey>(keygen.secret_key()));

        // send sk for debug
        auto sk_buf_send = EncodeSEALObject(*secret_key_[idx]);
        conn->Send(nxt_rank, sk_buf_send, "rank1 send sk");

        // generate and send pk
        seal::PublicKey public_key;
        keygen.create_public_key(public_key);
        auto pk_buf_send = EncodeSEALObject(public_key);
        conn->Send(nxt_rank, pk_buf_send, "rank1 send pk");
        peer_pub_key_.push_back(std::make_shared<seal::PublicKey>(public_key));
      } 

      // create the functors
      decryptors_.push_back(
          std::make_shared<seal::Decryptor>(seal_cntxts_[idx], *(secret_key_[idx])));
    // } else {
      // For other CRT context, we just copy the sk/pk
      // LocalExpandSEALContexts(idx);
    }
    simd_batchmm_instances_.push_back(
        std::make_shared<SIMDBatchMMProt>(kPolyDegree, crt_modulus[idx].value()));
  }

  current_crt_plain_bitlen_ = target_plain_bitlen;
  // SPDLOG_INFO("CheetahMul uses {} modulus for {} bit input over {} bit ring",
  //             num_seal_ctx, options.msg_bitlen, options.ring_bitlen);
}

void BatchMatMul::Impl::InitGaloisKey(const Shape4D& dim4) {
  // check whether galois keys are already initialized
  if (peer_gal_key_.size() == seal_cntxts_.size()) {
    return;
  }  

  // compute basic parameters
  SIMDBatchMMProt::Meta meta;
  meta.batch = dim4[0];
  meta.dims = {dim4[1], dim4[2], dim4[3]};
  Shape2D in_shape = simd_batchmm_instances_[0]->ComputeInShape(meta);
  size_t block_size = in_shape[1];
  uint64_t baby_step = absl::bit_ceil(
        static_cast<uint64_t>(std::sqrt(block_size * meta.dims[2] / (double)meta.dims[1])));
  baby_step = std::min(baby_step, block_size);
  uint64_t step0 = in_shape[0];
  uint64_t giant_step = (block_size / baby_step);

  std::vector<int> steps;
  // {step0, 2*step0, ..., (baby_step-1)*step0} 
  for (uint64_t s = 1; s < baby_step; ++s) {
    steps.push_back(s * step0);
  }
  // U {baby_step*step0, 2*baby_step*step0, (gstep-1)*baby_step*step0}
  for (uint64_t gs = 1; gs < giant_step; ++gs) {
    steps.push_back(gs * baby_step * step0);
  }

  for (size_t idx = 0; idx < seal_cntxts_.size(); ++idx) {
    if (lctx_->NextRank() == 0) {// server
      // receive galois key
      auto gk_buf = lctx_->Recv(lctx_->NextRank(), "recv galois key");
      peer_gal_key_.push_back(std::make_shared<seal::GaloisKeys>());
      DecodeSEALObject(gk_buf, seal_cntxts_[idx], peer_gal_key_[idx].get());
    }
    else {// client
      // generate and send galois key
      seal::KeyGenerator keygen(seal_cntxts_[idx], *(secret_key_[idx]));
      seal::GaloisKeys gk;
      keygen.create_galois_keys(steps, gk);
      // keygen.create_galois_keys(gk);
      peer_gal_key_.push_back(std::make_shared<seal::GaloisKeys>(gk));
      auto gk_buf = EncodeSEALObject(gk);
      lctx_->Send(lctx_->NextRank(), gk_buf, "send galois key");
    }
  }
}

NdArrayRef BatchMatMul::Impl::MatMulClient(const NdArrayRef& x, 
                                           yacl::link::Context* conn, 
                                           const Shape4D& dim4,
                                           uint32_t msg_width_hint) {
  if (conn == nullptr) {
  conn = lctx_.get();
  }
  InitGaloisKey(dim4);

  auto eltype = x.eltype();
  SPU_ENFORCE(eltype.isa<Ring2k>(), "must be ring_type, got={}", eltype);
  SPU_ENFORCE(x.numel() > 0);

  auto field = eltype.as<Ring2k>()->field();
  Options options;
  options.ring_bitlen = SizeOf(field) * 8;
  options.msg_bitlen =
      msg_width_hint == 0 ? options.ring_bitlen : msg_width_hint;
  SPU_ENFORCE(options.msg_bitlen > 0 &&
              options.msg_bitlen <= options.ring_bitlen);
  LazyExpandSEALContexts(options, conn);
  LazyInitModSwitchHelper(options);
  const int64_t num_seal_ctx = WorkingContextSize(options);


  NdArrayRef x_vec;
  // prepare input vector into the order we want
  SIMDBatchMMProt::Meta meta;
  meta.batch = dim4[0];
  meta.dims = {dim4[1], dim4[2], dim4[3]};

  Shape2D in_shape = simd_batchmm_instances_[0]->ComputeInShape(meta);
  x_vec = simd_batchmm_instances_[0]->PrepareInputVector(meta, in_shape, x);
  // SPDLOG_INFO("Client prepared input vector shape: {}", x_vec.shape());

  auto io_task = std::async(std::launch::async, [&]() {
    size_t bytes_sent = conn->GetStats()->sent_bytes;
    size_t payload_size = EncryptArrayThenSend(x_vec, dim4, options, conn);
    bytes_sent = conn->GetStats()->sent_bytes - bytes_sent;
    SPDLOG_INFO("Client sent {} ct in {} MiB", payload_size, 
                std::roundf(bytes_sent / 1024. / 1024. * 1000) / 1000.);
  });
  size_t num_ct_to_recv =
      simd_batchmm_instances_[0]->ComputeOutputCtNum(meta, in_shape) * num_seal_ctx;

  // receive the result ciphers
  io_task.get();
  // SPDLOG_INFO("Client expecting to receive {} ciphertexts", num_ct_to_recv);
  int next_rank = conn->NextRank();
  std::vector<yacl::Buffer> recv_ct(num_ct_to_recv);
  for (size_t i = 0; i < num_ct_to_recv; i += 1) {
    // SPDLOG_INFO("Client received ciphertext {}/{}", i + 1, num_ct_to_recv);
    recv_ct[i] = conn->Recv(next_rank, "");
  }

  NdArrayRef rec_mat = DecryptArray(field, num_ct_to_recv*kPolyDegree/num_seal_ctx, options, recv_ct);

  return simd_batchmm_instances_[0]->ParseResult(meta, in_shape, rec_mat);
}

NdArrayRef BatchMatMul::Impl::MatMulServer(const NdArrayRef& x, 
                                           const NdArrayRef& w,
                                           yacl::link::Context* conn, 
                                           const Shape4D& dim4,
                                           uint32_t msg_width_hint) {
  if (conn == nullptr) {
  conn = lctx_.get();
  }
  InitGaloisKey(dim4);

  auto eltype = x.eltype();
  SPU_ENFORCE(eltype.isa<Ring2k>(), "must be ring_type, got={}", eltype);
  SPU_ENFORCE(x.numel() > 0);

  auto field = eltype.as<Ring2k>()->field();
  Options options;
  options.ring_bitlen = SizeOf(field) * 8;
  options.msg_bitlen =
      msg_width_hint == 0 ? options.ring_bitlen : msg_width_hint;
  SPU_ENFORCE(options.msg_bitlen > 0 &&
              options.msg_bitlen <= options.ring_bitlen);
  LazyExpandSEALContexts(options, conn);
  LazyInitModSwitchHelper(options);
  const int64_t num_seal_ctx = WorkingContextSize(options);

  SIMDBatchMMProt::Meta meta;
  meta.batch = dim4[0];
  meta.dims = {dim4[1], dim4[2], dim4[3]};
  Shape2D in_shape = simd_batchmm_instances_[0]->ComputeInShape(meta);

  size_t num_input = simd_batchmm_instances_[0]->ComputeInputCtNum(meta, in_shape);
  size_t num_ct_to_recv = num_input * num_seal_ctx;
  // SPDLOG_INFO("Server expecting to receive {} ciphertexts", num_ct_to_recv);

  // receive the input ciphers share
  int next_rank = conn->NextRank();
  std::vector<yacl::Buffer> recv_ct(num_ct_to_recv);
  auto io_task = std::async(std::launch::async, [&]() {
    for (size_t i = 0; i < num_ct_to_recv; i += 1) {
      recv_ct[i] = conn->Recv(next_rank,"");
      // SPDLOG_INFO("Server received ciphertext {}/{}", i + 1, num_ct_to_recv);
    }
  });

  // prepare weight vector into the order we want

  yacl::ElapsedTimer pack_timer;

  NdArrayRef w_vec;
  w_vec = simd_batchmm_instances_[0]->PrepareWeightVector(meta, in_shape, w);
  std::vector<RLWEPt> encoded_w;
  EncodeArray(w_vec, /*need_encrypt*/ false, options, &encoded_w);

  // double weight_pack_time = pack_timer.CountMs();
  // SPDLOG_INFO("Server packed weight vector time: {} ms", std::roundf(weight_pack_time * 1000) / 1000.);

  // prepare input vector into the order we want
  NdArrayRef x_vec;
  x_vec = simd_batchmm_instances_[0]->PrepareInputVector(meta, in_shape, x);
  std::vector<RLWEPt> encoded_x;
  EncodeArray(x_vec, /*need_encrypt*/ true, options, &encoded_x);

  // prepare random mask
  std::vector<uint64_t> rnd_mask;
  size_t out_num = simd_batchmm_instances_[0]->ComputeOutputCtNum(meta, in_shape);
  PrepareRandomMask(field, out_num * kPolyDegree, options, rnd_mask);

  io_task.get();
  // Decode but not decrypt the received ciphers
  std::vector<RLWECt> input_ct(num_ct_to_recv);
  yacl::parallel_for(0, num_ct_to_recv, [&](uint64_t job_bgn, uint64_t job_end) {
    for (uint64_t job_id = job_bgn; job_id < job_end; ++job_id) {
      DecodeSEALObject(recv_ct[job_id], seal_cntxts_[job_id / num_input],
                       &input_ct[job_id]);
    }
  });

  // SPDLOG_INFO("Server received and decoded {} ciphertexts", num_ct_to_recv);

  // Add input_ct with plaintext input x
  for (size_t i = 0; i < (size_t)num_seal_ctx; ++i) {
    seal::Evaluator evaluator(seal_cntxts_[i]);
    for (size_t j = 0; j < num_input; ++j) {
      evaluator.add_plain_inplace(input_ct[i * num_input + j], encoded_x[i * num_input + j]);
    }
  }

  // do the batch matmul and response
  BatchMatMulThenResponse(field, dim4, options, absl::MakeSpan(input_ct),
                          absl::MakeSpan(encoded_w), absl::MakeSpan(rnd_mask),
                          conn);

  // construct the output NdArrayRef
  // the output share of the server is random mask
  // convert num_seal_ctx of rnd_mask(uint64) to NdArrayRef(field)
  int64_t out_elts = out_num * kPolyDegree;
  NdArrayRef out = ring_zeros(field, {out_elts});
  auto &ms_helper = ms_helpers_.find(options)->second;
  auto out1 = ms_helper.ModulusDownRNS(field, {out_elts}, rnd_mask).reshape({out_elts});
  return simd_batchmm_instances_[0]->ParseResult(meta, in_shape, out1);
}

size_t BatchMatMul::Impl::EncryptArrayThenSend(const NdArrayRef &array,
                                               const Shape4D &dim4,
                                               const Options &options,
                                               yacl::link::Context *conn) {
  int64_t num_elts = array.numel();
  auto eltype = array.eltype();
  SPU_ENFORCE(num_elts > 0, "empty array");
  SPU_ENFORCE(eltype.isa<Ring2k>(), "array must be ring_type, got={}", eltype);

  int64_t num_splits = CeilDiv(num_elts, num_slots());
  int64_t num_seal_ctx = WorkingContextSize(options);
  int64_t num_polys = num_seal_ctx * num_splits;

  std::vector<RLWEPt> encoded_array(num_polys);
  EncodeArray(array, /*scale_up*/ true, options, absl::MakeSpan(encoded_array));

  std::vector<yacl::Buffer> payload(num_polys);


  yacl::parallel_for(0, num_polys, [&](int64_t job_bgn, int64_t job_end) {
    for (int64_t job_id = job_bgn; job_id < job_end; ++job_id) {
      RLWECt ct;
      int64_t cntxt_id = job_id / num_splits;

      simd_batchmm_instances_[cntxt_id]->SymEncrypt(
          {&encoded_array[job_id], 1}, *(secret_key_[cntxt_id]), seal_cntxts_[cntxt_id],
          /*save_seed*/ true, {&ct, 1});
      payload.at(job_id) = EncodeSEALObject(ct);
      
    }
  });

  if (conn == nullptr) {
    conn = lctx_.get();
  }

  int nxt_rank = conn->NextRank();
  for (size_t i = 0; i < payload.size(); i += 1) {
    conn->Send(nxt_rank, payload[i], "");
  }
  return payload.size();
}

void BatchMatMul::Impl::PrepareRandomMask(FieldType field, int64_t size,
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

void BatchMatMul::Impl::EncodeArray(const NdArrayRef &array, bool need_encrypt,
                                   const Options &options,
                                   absl::Span<RLWEPt> out) {
  int64_t num_elts = array.numel();
  auto eltype = array.eltype();
  SPU_ENFORCE(num_elts > 0, "empty array");
  SPU_ENFORCE(eltype.isa<Ring2k>(), "array must be ring_type, got={}", eltype);

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

    NdArrayRef slots(array.eltype(), {num_slots()});

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
        ms_helper.CenteralizeAt(slots.slice({0}, {slice_n}, {1}), cntxt_id, dst);
      }
      // zero-padding the rest
      std::fill_n(u64tmp.data() + slice_n, u64tmp.size() - slice_n, 0);

      simd_batchmm_instances_[cntxt_id]->EncodeSingle(_u64tmp, out[job_id]);
    }
  });
}

void BatchMatMul::Impl::BatchMatMulThenResponse(FieldType, Shape4D dim4,
                                                const Options &options,
                                                absl::Span<const RLWECt> ciphers,
                                                absl::Span<const RLWEPt> plains,
                                                absl::Span<const uint64_t> rnd_mask,
                                                yacl::link::Context *conn) {  
  SIMDBatchMMProt::Meta meta;
  meta.batch = dim4[0];
  meta.dims = {dim4[1], dim4[2], dim4[3]};

  // different instances have different coefficient modulus, but have the same 
  Shape2D in_shape = simd_batchmm_instances_[0]->ComputeInShape(meta);

  const int64_t num_splits = simd_batchmm_instances_[0]->ComputeInputCtNum(meta, in_shape);
  const int64_t num_seal_ctx = WorkingContextSize(options);
  const int64_t num_ciphers = num_seal_ctx * num_splits;
  SPU_ENFORCE(ciphers.size() == (size_t)num_ciphers,
              "BatchMatMul: cipher size mismatch");

  const int64_t num_out_ct = simd_batchmm_instances_[0]->ComputeOutputCtNum(meta, in_shape);
  SPU_ENFORCE(rnd_mask.size() == num_seal_ctx * num_out_ct * kPolyDegree,
              "BatchMatMul: rnd_mask size mismatch");
  // SPDLOG_INFO("server: num_splits: {}, num_seal_ctx: {}, num_out_ct: {}",
  //             num_splits, num_seal_ctx, num_out_ct);

  const int64_t num_plains = simd_batchmm_instances_[0]->ComputeWeightPtNum(meta, in_shape);
  SPU_ENFORCE(plains.size() == (size_t)num_plains * num_seal_ctx,
              "BatchMatMul: plain size mismatch");

  std::vector<RLWECt> out_ct(num_seal_ctx * num_out_ct);

// {
//       auto &context = seal_cntxts_[0];
//       auto chain_index = context.get_context_data(ciphers[0].parms_id())->chain_index();
//       auto noise_budget = decryptors_[0]->invariant_noise_budget(ciphers[0]);
//       std::cout << "-----------------before BMM---------------" << std::endl;
//       std::cout << "Level (chain index): " << chain_index << ", Noise budget: " << noise_budget << " bits" << std::endl;
//       std::cout << "----------------------------------------" << std::endl;
// }
// std::cout << "server: start BatchMatMul compute ..." << std::endl;
  yacl::ElapsedTimer pack_timer;

  // yacl::parallel_for(0, num_seal_ctx, [&](size_t job_bgn, size_t job_end) {
    size_t job_bgn = 0;
    size_t job_end = num_seal_ctx;
    for (size_t cidx = job_bgn; cidx < job_end; ++cidx) {
      simd_batchmm_instances_[cidx]->BatchMatMatMul(
          meta, in_shape, 
          ciphers.subspan(cidx * num_splits, num_splits),
          plains.subspan(cidx * num_plains, num_plains), *(peer_pub_key_[cidx]), *(peer_gal_key_[cidx]),
          seal_cntxts_[cidx], 
          absl::MakeSpan(out_ct).subspan(cidx * num_out_ct, num_out_ct)
      );

  // {
  //       auto &context = seal_cntxts_[0];
  //       auto chain_index = context.get_context_data(out_ct[0].parms_id())->chain_index();
  //       auto noise_budget = decryptors_[0]->invariant_noise_budget(out_ct[0]);
  //       std::cout << "-------------------after BMM----------------[" << cidx << "]" << std::endl;
  //       std::cout << "Level (chain index): " << chain_index << ", Noise budget: " << noise_budget << " bits" << std::endl;
  //       std::cout << "----------------------------------------" << std::endl;
  // }

      // add rnd_mask 
      simd_batchmm_instances_[cidx]->ReshareOutputInplace(
          absl::MakeSpan(out_ct).subspan(cidx * num_out_ct, num_out_ct),
          rnd_mask.subspan(cidx * num_out_ct * kPolyDegree, num_out_ct * kPolyDegree),
          *(peer_pub_key_[cidx]), seal_cntxts_[cidx]
      );
    }
  // });

  double compute_time = pack_timer.CountMs();

  if (conn == nullptr) {
    conn = lctx_.get();
  }

  std::vector<yacl::Buffer> response(num_seal_ctx * num_out_ct);
  yacl::parallel_for(0, num_seal_ctx * num_out_ct, [&](int64_t job_bgn, int64_t job_end) {
    for (int64_t job_id = job_bgn; job_id < job_end; ++job_id) {
      response[job_id] = EncodeSEALObject(out_ct[job_id]);
    }
  });

  int nxt_rank = conn->NextRank();
  size_t bytes_sent = conn->GetStats()->sent_bytes;
  for (size_t i = 0; i < response.size(); i += 1) {
    // SPDLOG_INFO("Server sending ciphertext {}/{}", i + 1, response.size());
    conn->Send(nxt_rank, response[i], "");
  }
  bytes_sent = conn->GetStats()->sent_bytes - bytes_sent;
  
  SPDLOG_INFO("Server BatchMatMul compute time: {} ms, sending back {} ct in {} MiB", 
              std::roundf(compute_time * 1000) / 1000.,
              response.size(), 
              std::roundf(bytes_sent / 1024. / 1024. * 1000) / 1000.);
  
}


NdArrayRef BatchMatMul::Impl::DecryptArray(FieldType field, int64_t size, const Options &options,
                                           const std::vector<yacl::Buffer> &ct_array) {
  const int64_t num_splits = CeilDiv(size, num_slots());
  const int64_t num_seal_ctx = WorkingContextSize(options);
  const int64_t num_ciphers = num_seal_ctx * num_splits;
  SPU_ENFORCE_EQ(ct_array.size(), (size_t)num_ciphers);
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

// if (job_id == 0)
// {
//       auto &context = seal_cntxts_[cntxt_id];
//       auto chain_index = context.get_context_data(ct.parms_id())->chain_index();
//       auto noise_budget = decryptors_[cntxt_id]->invariant_noise_budget(ct);
//       std::cout << "--------------client dec------------------" << std::endl;
//       std::cout << "Level (chain index): " << chain_index << ", Noise budget: " << noise_budget << " bits" << std::endl;
//       std::cout << "----------------------------------------" << std::endl;
// }


      simd_batchmm_instances_[cntxt_id]->DecodeSingle(pt, absl::MakeSpan(subarray));

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

BatchMatMul::BatchMatMul(std::shared_ptr<yacl::link::Context> lctx,
                       bool allow_high_prob_one_bit_error) {
  impl_ = std::make_unique<Impl>(lctx, allow_high_prob_one_bit_error);
}

BatchMatMul::~BatchMatMul() = default;

int BatchMatMul::Rank() const { return impl_->Rank(); }

size_t BatchMatMul::OLEBatchSize() const {
  SPU_ENFORCE(impl_ != nullptr);
  return impl_->OLEBatchSize();
}

NdArrayRef BatchMatMul::MatMulClient(const NdArrayRef& x, 
                                     yacl::link::Context* conn,
                                     const Shape4D& dim4,
                                     uint32_t msg_width_hint) {
  SPU_ENFORCE(conn != nullptr);
  SPU_ENFORCE(impl_ != nullptr);
  return impl_->MatMulClient(x, conn, dim4, msg_width_hint);
}

NdArrayRef BatchMatMul::MatMulClient(const NdArrayRef& x, 
                                     const Shape4D& dim4,
                                     uint32_t msg_width_hint) {
  SPU_ENFORCE(impl_ != nullptr);
  return impl_->MatMulClient(x, nullptr, dim4, 0);
}

NdArrayRef BatchMatMul::MatMulServer(const NdArrayRef& x, 
                                     const NdArrayRef& w,
                                     yacl::link::Context* conn,
                                     const Shape4D& dim4,
                                     uint32_t msg_width_hint) {
  SPU_ENFORCE(conn != nullptr);
  SPU_ENFORCE(impl_ != nullptr);
  return impl_->MatMulServer(x, w, conn, dim4, msg_width_hint);
}

NdArrayRef BatchMatMul::MatMulServer(const NdArrayRef& x, 
                                     const NdArrayRef& w,
                                     const Shape4D& dim4,
                                     uint32_t msg_width_hint) {
  SPU_ENFORCE(impl_ != nullptr);
  return impl_->MatMulServer(x, w, nullptr, dim4, msg_width_hint);
}

void BatchMatMul::LazyInitKeys(FieldType field, uint32_t msg_width_hint) {
  SPU_ENFORCE(impl_ != nullptr);
  SPU_ENFORCE(msg_width_hint <= SizeOf(field) * 8);
  return impl_->LazyInit(field, msg_width_hint);
}

}  // namespace spu::mpc::cheetah
