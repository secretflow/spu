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

#include "experimental/squirrel/bin_matvec_prot.h"

#include <future>

#include "seal/seal.h"
#include "seal/util/polyarithsmallmod.h"
#include "seal/util/rlwe.h"
#include "yacl/utils/elapsed_timer.h"

#include "libspu/mpc/cheetah/arith/common.h"
#include "libspu/mpc/cheetah/arith/vector_encoder.h"
#include "libspu/mpc/cheetah/rlwe/lwe_ct.h"
#include "libspu/mpc/cheetah/rlwe/packlwes.h"
#include "libspu/mpc/cheetah/rlwe/utils.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace squirrel {

template <typename T>
inline T CeilDiv(T a, T b) {
  SPU_ENFORCE(b > 0);
  return (a + b - 1) / b;
}

template <typename MatType>
int64_t GetCols(MatType const &mat);

template <typename MatType>
int64_t GetRows(MatType const &mat);

template <>
int64_t GetCols(const StlSparseMatrix &mat) {
  return mat.cols();
}

template <>
int64_t GetRows(const StlSparseMatrix &mat) {
  return mat.rows();
}

void GenerateOneKSKey(seal::util::ConstRNSIter new_key,
                      const seal::SecretKey &secret_key,
                      const seal::SEALContext &context,
                      std::vector<seal::PublicKey> &destination,
                      bool save_seed);

// REF: Squirrel: A Scalable Secure Two-Party Computation Framework for Training
// Gradient Boosting Decision Tree
void GenerateLWEKeySwitchKey(const seal::SecretKey &src_key,
                             const seal::SecretKey &dst_key,
                             const seal::SEALContext &src_context,
                             const seal::SEALContext &dst_context,
                             seal::KSwitchKeys &out, bool save_seed = false);

// Lift the LWE ciphertext to a different dimension
// LWE_n(m) -> LWE_N(m) for N >= n
void LiftLWEDimension(const spu::mpc::cheetah::LWECt &src,
                      const seal::KSwitchKeys &lift_key,
                      const seal::SEALContext &src_context,
                      const seal::SEALContext &dst_context,
                      spu::mpc::cheetah::LWECt &dst);

struct BinMatVecProtocol::Impl : public spu::mpc::cheetah::EnableCPRNG {
 public:
  static constexpr int64_t kCtAsyncParallel = 8;  // not to send too much aysnc
                                                  //
  Impl(size_t ring_bitwidth, std::shared_ptr<yacl::link::Context> conn) {
    SPU_ENFORCE(ring_bitwidth <= 128U and ring_bitwidth >= 32U,
                "require 32 <= ring_bitwidth <= 128 but got={}", ring_bitwidth);
    SPU_ENFORCE(conn != nullptr);
    conn_ = conn;
    ring_bitwidth_ = ring_bitwidth;

    InitKeys(ring_bitwidth);
  }

  spu::NdArrayRef Send(const spu::NdArrayRef &ashare, int64_t dim_out,
                       int64_t dim_in) {
    int64_t n = ashare.numel();
    SPU_ENFORCE_EQ(n, dim_in);
    if (dim_in == 0 || dim_out == 0) {
      return spu::NdArrayRef(ashare.eltype(), {0});
    }
    auto field = ashare.eltype().as<spu::RingTy>()->field();

    // convert from AShare to HE ciphertext
    A2hSend(ashare);

    // Wait result
    return H2aRecv(field, dim_out).as(ashare.eltype());
  }

  template <class SparsMatType>
  spu::NdArrayRef Recv(const spu::NdArrayRef &ashare, int64_t dim_out,
                       int64_t dim_in, const SparsMatType &prv_bin_mat,
                       absl::Span<const uint8_t> indicator) {
    int64_t n = ashare.numel();
    SPU_ENFORCE_EQ(n, dim_in);
    if (dim_in == 0 || dim_out == 0) {
      return spu::NdArrayRef(ashare.eltype(), {0});
    }
    int64_t cols = GetCols(prv_bin_mat);
    int64_t rows = GetRows(prv_bin_mat);
    SPU_ENFORCE_EQ(cols, dim_in, "dim_in mismatch, expected={}, got = {}", cols,
                   dim_in);
    SPU_ENFORCE_EQ(rows, dim_out, "dim_out mismatch, expected={}, got={}", rows,
                   dim_out);
    auto field = ashare.eltype().as<spu::RingTy>()->field();
    // Step 1: A2H
    std::vector<seal::Ciphertext> rlwe_cipers(
        CeilDiv<size_t>(dim_in, poly_degree_in(ring_bitwidth_)));
    A2hRecv(ashare, absl::MakeSpan(rlwe_cipers));

#if ENABLE_CACHED_RLWE
    // TODO(lwj): implement the AVX acceleration
    using CtType = spu::mpc::cheetah::CachedRLWECt;
    const auto &in_context = *(in_cntxts_.find(ring_)->second);
    std::vector<CtType> cached_rlwe_ciphers(rlwe_cipers.size());
    yacl::parallel_for(
        0, rlwe_cipers.size(), kParallelStride, [&](size_t bgn, size_t end) {
          for (size_t i = bgn; i < end; ++i) {
            cached_rlwe_ciphers[i].CacheIt(rlwe_cipers[i], in_context);
          }
        });
    absl::Span<const CtType> hshare = absl::MakeSpan(cached_rlwe_ciphers);
#else
    using CtType = seal::Ciphertext;
    absl::Span<const CtType> hshare = absl::MakeSpan(rlwe_cipers);
#endif

    // Step 2: Extract-then-Add
    size_t num_lwes = 0;
    if (dim_out <= static_cast<int64_t>(poly_degree_out(ring_bitwidth_))) {
      num_lwes = absl::bit_ceil(static_cast<size_t>(dim_out));
    } else {
      // NOTE: we find the smallest `d` such that
      //       dim_out <= k*poly_degree_out + 2^d
      size_t floor = absl::bit_floor(static_cast<size_t>(dim_out));
      size_t margin = dim_out - floor;
      num_lwes = floor;
      if (margin > 0) {
        num_lwes += absl::bit_ceil(margin);
      }
    }
    std::vector<spu::mpc::cheetah::LWECt> lwes(num_lwes);
    doBinMatVec<CtType>(field, prv_bin_mat, hshare, indicator,
                        {lwes.data(), static_cast<size_t>(dim_out)});
    // Step 3: Pack LWEs as RLWEs
    std::vector<seal::Ciphertext> packed_lwes =
        doPackLWEs(field, absl::MakeSpan(lwes));

    // Step 4: reshare via H2A
    return H2aSend(field, dim_out, absl::MakeSpan(packed_lwes))
        .as(ashare.eltype());
  }

 private:
  constexpr size_t poly_degree_in(size_t ring_bitlen) const {
    if (ring_bitlen <= 32) {
      return 4096;
    } else if (ring_bitlen <= 64) {
      return 4096;
    }
    return 8192;
  }

  constexpr size_t poly_degree_out(size_t ring_bitlen) const {
    if (ring_bitlen <= 32) {
      return 4096;
    } else if (ring_bitlen <= 64) {
      return 8192;
    }
    return 8192;
  }

  void EncodeVectorToPoly(const spu::NdArrayRef &vec,
                          const spu::mpc::cheetah::VectorEncoder &encoder,
                          seal::Plaintext &poly) const;

  spu::NdArrayRef DecodePolyToVector(
      spu::FieldType field, int64_t numel, const seal::Plaintext &poly,
      const spu::mpc::cheetah::VectorEncoder &encoder) const;

  std::vector<seal::Ciphertext> doPackLWEs(
      spu::FieldType field, absl::Span<spu::mpc::cheetah::LWECt> lwes) const;

  template <typename RLWEType, typename SparsMatType>
  void doBinMatVec(spu::FieldType field, const SparsMatType &matrix,
                   absl::Span<const RLWEType> vec,
                   absl::Span<const uint8_t> indicator,
                   absl::Span<spu::mpc::cheetah::LWECt> outp) const {
    const auto &context = *in_context_;

    size_t dim_in = GetCols(matrix);
    size_t dim_out = GetRows(matrix);
    size_t num_lwes = outp.size();
    SPU_ENFORCE_EQ(num_lwes, dim_out, "expected num_lwes={}, got={}", dim_out,
                   outp.size());

    // Partition `n` samples into batches.
    // Each batch contains at most `2^B` samples.
    // When `n < 2^B`, insert 0s between two samples to make it power-2-align.
    // That is n -> bit_ceil(n).
    struct IndexMapper {
      IndexMapper(size_t num_samples, size_t sample_batch_size) {
        // sample_batch_size needs to be a 2-power value.
        SPU_ENFORCE(num_samples > 0 && sample_batch_size > 0 &&
                    absl::has_single_bit(sample_batch_size));
        num_samples_ = num_samples;
        div_ = absl::bit_width(sample_batch_size) - 1;
        mod_ = sample_batch_size - 1;

        size_t num_batch = CeilDiv(num_samples, sample_batch_size);
        gaps_.resize(num_batch, 1);
        for (size_t i = 0; i < gaps_.size(); ++i) {
          size_t bgn = i * sample_batch_size;
          size_t end = std::min(bgn + sample_batch_size, num_samples);
          gaps_[i] = sample_batch_size / absl::bit_ceil(end - bgn);
        }
      }

      std::array<size_t, 2> MapToLoc(size_t sample_index) {
        SPU_ENFORCE(sample_index < num_samples_);
        size_t rlwe_idx = sample_index >> div_;
        size_t cidx = (sample_index & mod_) * gaps_.at(rlwe_idx);
        return {rlwe_idx, cidx};
      }

     private:
      size_t num_samples_;
      size_t div_, mod_;
      std::vector<size_t> gaps_;
    };
    IndexMapper mapper(dim_in, poly_degree_in(ring_bitwidth_));

    auto pick_then_sum = [&](size_t row_start, size_t row_util) {
      for (size_t row = row_start; row < row_util; ++row) {
        for (auto col_iter = matrix.iterate_row_begin(row);
             col_iter != matrix.iterate_row_end(row); ++col_iter) {
          SPU_ENFORCE(*col_iter < dim_in);
          if (not indicator.empty() and 0 == indicator[*col_iter]) {
            continue;
          }
          auto loc = mapper.MapToLoc(*col_iter);
          outp[row].AddLazyInplace(vec[loc[0]], loc[1], context);
        }

        if (!outp[row].IsValid()) {
          // empty LWE
        }
        outp[row].Reduce(context);
      }
    };

    yacl::parallel_for(0, dim_out, pick_then_sum);
  }

  void A2hSend(const spu::NdArrayRef &ashare);

  void A2hRecv(const spu::NdArrayRef &ashare,
               absl::Span<seal::Ciphertext> hshare);

  spu::NdArrayRef H2aSend(spu::FieldType field, int64_t len,
                          absl::Span<seal::Ciphertext> hshare);

  spu::NdArrayRef H2aRecv(spu::FieldType field, int64_t len);

  std::tuple<seal::EncryptionParameters, seal::EncryptionParameters>
  DecideSEALParameters(size_t ring_bitlen);

  void InitKeys(size_t ring_bitwidth);

  size_t ring_bitwidth_;
  std::shared_ptr<yacl::link::Context> conn_;

  // RLWE_n
  std::unique_ptr<seal::SEALContext> in_context_;
  std::unique_ptr<seal::SecretKey> in_skey_;
  std::unique_ptr<spu::mpc::cheetah::VectorEncoder> in_vencoder_;
  // RLWE_N
  std::unique_ptr<seal::SEALContext> out_context_;
  std::unique_ptr<seal::SecretKey> out_skey_;
  std::unique_ptr<spu::mpc::cheetah::VectorEncoder> out_vencoder_;

  std::unique_ptr<seal::PublicKey> out_peer_pkey_;
  std::unique_ptr<seal::GaloisKeys> peer_galoiskey_;
  // LWE_n -> LWE_N
  std::unique_ptr<seal::KSwitchKeys> dim_lift_key_;
};

std::tuple<seal::EncryptionParameters, seal::EncryptionParameters>
BinMatVecProtocol::Impl::DecideSEALParameters(size_t ring_bitlen) {
  size_t poly_deg_n = poly_degree_in(ring_bitlen);
  size_t poly_deg_N = poly_degree_out(ring_bitlen);

  std::vector<int> modulus_bits;
  if (ring_bitlen <= 32) {
    // 74bit modulus for 32bit accumulator
    modulus_bits = {45, 29, 35};
  } else if (ring_bitlen <= 64) {
    // 109bit modulus for 64bit accumulator
    modulus_bits = {55, 54, 60};
  } else {
    // 160bit modulus for 128bit accumulator
    modulus_bits = {53, 53, 54, 58};
  }

  auto scheme_type = seal::scheme_type::ckks;
  auto in_parms = seal::EncryptionParameters(scheme_type);
  auto out_parms = seal::EncryptionParameters(scheme_type);

  // NOTE p = 1 mod 2N should also p = 1 mod 2n given N >= n
  auto modulus = seal::CoeffModulus::Create(poly_deg_N, modulus_bits);

  out_parms.set_use_special_prime(true);
  out_parms.set_poly_modulus_degree(poly_deg_N);
  out_parms.set_coeff_modulus(modulus);

  in_parms.set_use_special_prime(false);
  in_parms.set_poly_modulus_degree(poly_deg_n);
  modulus.pop_back();
  in_parms.set_coeff_modulus(modulus);
  return {in_parms, out_parms};
}

void BinMatVecProtocol::Impl::InitKeys(size_t ring_bitwidth) {
  using namespace spu::mpc::cheetah;
  auto [in_parms, out_parms] = DecideSEALParameters(ring_bitwidth);
  SPU_ENFORCE(not in_parms.use_special_prime());

  in_context_ = std::make_unique<seal::SEALContext>(in_parms, true,
                                                    seal::sec_level_type::none);
  out_context_ = std::make_unique<seal::SEALContext>(
      out_parms, true, seal::sec_level_type::none);

  seal::KeyGenerator in_keygen(*in_context_);
  seal::KeyGenerator out_keygen(*out_context_);
  in_skey_ = std::make_unique<seal::SecretKey>(in_keygen.secret_key());
  out_skey_ = std::make_unique<seal::SecretKey>(out_keygen.secret_key());

  // GaloisKeys for PackLWEs
  std::vector<uint32_t> galois_elt;
  size_t logN = absl::bit_width(out_parms.poly_modulus_degree()) - 1;
  for (uint32_t i = 1; i <= logN; i++) {
    galois_elt.push_back((1UL << i) + 1);
  }

  // public key
  auto pk = out_keygen.create_public_key();
  // galois for PackLWEs
  auto galois_keys = out_keygen.create_galois_keys(galois_elt);
  // switching key for LWE dimension lifting
  auto dim_lift_key = std::make_shared<seal::KSwitchKeys>();
  GenerateLWEKeySwitchKey(*in_skey_, *out_skey_, *in_context_, *out_context_,
                          *dim_lift_key, /*save_seed*/ true);

  // Exchange the public key materials
  std::vector<yacl::Buffer> keys(3);
  keys[0] = EncodeSEALObject(pk.obj());
  keys[1] = EncodeSEALObject(galois_keys);
  keys[2] = EncodeSEALObject(*dim_lift_key);
  [[maybe_unused]] size_t key_nbytes = 0;
  for (size_t i = 0; i < 3; ++i) {
    conn_->SendAsync(conn_->NextRank(), keys[i], "send keys");
    key_nbytes += keys[i].size();
  }

  std::vector<yacl::Buffer> recv_keys(3);
  for (size_t i = 0; i < 3; ++i) {
    recv_keys[i] = conn_->Recv(conn_->NextRank(), "recv keys");
  }

  out_peer_pkey_ = std::make_unique<seal::PublicKey>();
  peer_galoiskey_ = std::make_unique<seal::GaloisKeys>();
  dim_lift_key_ = std::make_unique<seal::KSwitchKeys>();

  DecodeSEALObject(recv_keys[0], *out_context_, out_peer_pkey_.get());
  DecodeSEALObject(recv_keys[1], *out_context_, peer_galoiskey_.get());
  DecodeSEALObject(recv_keys[2], *out_context_, dim_lift_key_.get());

  ModulusSwitchHelper in_msh(*in_context_, ring_bitwidth);
  in_vencoder_ = std::make_unique<VectorEncoder>(*in_context_, in_msh);

  if (out_parms.use_special_prime()) {
    // NOTE(lwj): Current ModulusSwitchHelper uses key_context_data() for
    // defining Q. Thus, we need to drop the last special prime first.
    std::vector<seal::Modulus> modulus = out_parms.coeff_modulus();
    modulus.pop_back();
    out_parms.set_coeff_modulus(modulus);
    seal::SEALContext out_ms_context(out_parms, false,
                                     seal::sec_level_type::none);
    ModulusSwitchHelper out_msh(out_ms_context, ring_bitwidth);
    out_vencoder_ = std::make_unique<VectorEncoder>(*out_context_, out_msh);
  } else {
    ModulusSwitchHelper out_msh(*out_context_, ring_bitwidth);
    out_vencoder_ = std::make_unique<VectorEncoder>(*out_context_, out_msh);
  }

  SPDLOG_DEBUG("BinMatVecProtocol sent {:4f} MB keys",
               static_cast<double>(key_nbytes) / 1024. / 1024.);
}

spu::NdArrayRef BinMatVecProtocol::Impl::H2aSend(
    spu::FieldType field, int64_t len, absl::Span<seal::Ciphertext> hshare) {
  using namespace spu::mpc;
  if (len == 0) {
    return ring_zeros(field, {len});
  }

  const auto &context = *out_context_;
  const auto &vencoder = *out_vencoder_;

  int64_t poly_deg_N = poly_degree_out(ring_bitwidth_);
  int64_t num_rlwes = hshare.size();
  SPU_ENFORCE_EQ(CeilDiv(len, poly_deg_N), num_rlwes);
  for (int64_t i = 0; i < num_rlwes; ++i) {
    SPU_ENFORCE_EQ(hshare[i].poly_modulus_degree(), (size_t)poly_deg_N);
    SPU_ENFORCE(not hshare[i].is_ntt_form());
  }

  spu::NdArrayRef out = ring_zeros(field, {len});
  auto reshare_callback = [&, this](int64_t bgn, int64_t end) {
    // sample r from Rq
    // remask hshare[i] <- hshare[i] + r
    // out share = -(2^k*r/q) mod 2^k
    seal::Evaluator evaluator(context);
    seal::Plaintext rand;
    for (int64_t i = bgn; i < end; ++i) {
      // sample r from Rq
      UniformPoly(context, &rand, hshare[i].parms_id());
      hshare[i].is_ntt_form() = true;  // foo SEAL
      evaluator.add_plain_inplace(hshare[i], rand);
      hshare[i].is_ntt_form() = false;

      int64_t slice_bgn = i * poly_deg_N;
      int64_t slice_end = std::min(slice_bgn + poly_deg_N, len);
      int64_t numel = slice_end - slice_bgn;

      // round(2^k*r/q)
      auto out_slice = DecodePolyToVector(field, numel, rand, vencoder);

      // -round(2^k*r/q)
      spu::mpc::ring_neg_(out_slice);

      std::memcpy(&out.at(slice_bgn), &out_slice.at(0), numel * out.elsize());
    }
  };

  yacl::parallel_for(0, num_rlwes, reshare_callback);

  [[maybe_unused]] size_t bytes_sent = 0;
  std::vector<yacl::Buffer> payloads(hshare.size());
  for (int64_t i = 0; i < num_rlwes; ++i) {
    payloads[i] = cheetah::EncodeSEALObject(hshare[i]);
    bytes_sent += payloads[i].size();
  }

  int next = conn_->NextRank();
  for (int64_t i = 0; i < num_rlwes; i += kCtAsyncParallel) {
    int64_t this_batch = std::min<int64_t>(num_rlwes - i, kCtAsyncParallel);
    auto tag = fmt::format("H2aSend_Batch{}", i / kCtAsyncParallel);
    for (int64_t j = 1; j < this_batch; ++j) {
      conn_->SendAsync(next, payloads[i + j - 1], tag);
    }
    conn_->Send(next, payloads[i + this_batch - 1], tag);
  }

  SPDLOG_DEBUG("H2A ({}) on {} shares sent {:4f} MB ", field, len,
               static_cast<double>(bytes_sent) / 1024. / 1024.);

  return out;
}

std::vector<seal::Ciphertext> BinMatVecProtocol::Impl::doPackLWEs(
    spu::FieldType field, absl::Span<spu::mpc::cheetah::LWECt> lwes) const {
  const size_t poly_deg_out = poly_degree_out(ring_bitwidth_);
  const auto &in_context = *in_context_;
  const auto &out_context = *out_context_;
  const auto &dim_lift_key = *dim_lift_key_;
  const auto &peer_galois = *peer_galoiskey_;

  // Step 3-1: LWEDimLift and re-randomize
  auto lift_callback = [&](int64_t start, int64_t util) {
    // TODO(lwj)
    // re-randomize the lifted LWE_N by adding a random LWE_N(0)
    seal::Ciphertext rlwe_N_zero;
    seal::util::encrypt_zero_asymmetric(*out_peer_pkey_, out_context,
                                        out_context.first_parms_id(),
                                        /*ntt*/ false, rlwe_N_zero);

    std::default_random_engine eng(std::time(0));
    std::uniform_int_distribution<size_t> uniform(0, poly_deg_out - 1);
    for (int64_t i = start; i < util; ++i) {
      // LWE_n -> LWE_N
      if (lwes[i].IsValid()) {
        spu::mpc::cheetah::LWECt copy{lwes[i]};
        LiftLWEDimension(copy, dim_lift_key, in_context, out_context, lwes[i]);
      }
      // randomize the lifted LWE_N by adding a random LWE_N(0)
      // This also make sure all LWEs to the PackLWEs are valid.
      lwes[i].AddLazyInplace(rlwe_N_zero, uniform(eng), out_context);
    }
  };

  const size_t num_lwes = lwes.size();
  yacl::ElapsedTimer timer;
  timer.Restart();
  yacl::parallel_for(0, num_lwes, lift_callback);
  [[maybe_unused]] double lift_time = timer.CountMs();

  // Step 3-2: PackLWEs on larger RLWE dimension
  // Each `DimOut` LWEs are packed into one RLWE_N
  timer.Restart();
  std::vector<seal::Ciphertext> packed(CeilDiv(num_lwes, poly_deg_out));
  size_t used = spu::mpc::cheetah::PackLWEs(lwes, peer_galois, out_context,
                                            absl::MakeSpan(packed));
  [[maybe_unused]] double pack_time = timer.CountMs();

  SPU_ENFORCE_EQ(used, packed.size());
  for (auto &rlwe : packed) {
    SPU_ENFORCE(rlwe.size() == 2 && not rlwe.is_ntt_form());
  }

  SPDLOG_DEBUG(
      "H2aSend: Lift {} LWEs took {:4f} ms. Pack them into {} RLWEs took "
      "{:4f} ms",
      num_lwes, lift_time, packed.size(), pack_time);
  return packed;
}

spu::NdArrayRef BinMatVecProtocol::Impl::H2aRecv(spu::FieldType field,
                                                 int64_t len) {
  using namespace spu::mpc;
  if (len == 0) {
    return ring_zeros(field, {len});
  }

  int64_t poly_deg_N = poly_degree_out(ring_bitwidth_);
  int64_t num_rlwes = CeilDiv(len, poly_deg_N);
  int next_rank = conn_->NextRank();

  const auto &context = *out_context_;
  const auto &rlwe_sk = *out_skey_;
  const auto &vencoder = *out_vencoder_;

  std::vector<seal::Ciphertext> rlwe_ciphers(num_rlwes);
  for (int64_t i = 0; i < num_rlwes; ++i) {
    auto ct_recv = conn_->Recv(next_rank, "H2aRecv");
    cheetah::DecodeSEALObject(ct_recv, context, &rlwe_ciphers[i],
                              /*do_check*/ true);
  }

  seal::Evaluator evaluator(context);
  seal::Decryptor decryptor(context, rlwe_sk);

  spu::NdArrayRef out = ring_zeros(field, {len});
  auto dec_callback = [&](int64_t start, int64_t util) {
    seal::Plaintext decrypted;
    for (int64_t i = start; i < util; ++i) {
      bool is_ntt = rlwe_ciphers[i].is_ntt_form();
      // NOTE: ckks decrypt to NTT form
      if (!is_ntt) {
        evaluator.transform_to_ntt_inplace(rlwe_ciphers[i]);
      }
      decryptor.decrypt(rlwe_ciphers[i], decrypted);
      cheetah::InvNttInplace(decrypted, context);

      int64_t bgn = i * poly_deg_N;
      int64_t end = std::min(bgn + poly_deg_N, len);
      int64_t numel = end - bgn;
      auto out_slice = DecodePolyToVector(field, numel, decrypted, vencoder);

      std::memcpy(&out.at(bgn), &out_slice.at(0), numel * out.elsize());
    }
  };

  yacl::parallel_for(0, num_rlwes, dec_callback);

  return out;
}

void BinMatVecProtocol::Impl::EncodeVectorToPoly(
    const spu::NdArrayRef &vec, const spu::mpc::cheetah::VectorEncoder &encoder,
    seal::Plaintext &poly) const {
  int64_t N = encoder.poly_degree();
  int64_t n = vec.numel();
  SPU_ENFORCE(n > 0 && n <= N, "invalid vector size={}", n);

  if (n == N) {
    encoder.Forward(vec, &poly, /*scale*/ true);
  } else {
    // NOTE(lwj): To uniform the A2H and H2A logic,
    // we insert zeros between two elements.
    // Because for input to H2A, it is computed from PackLWEs.
    spu::NdArrayRef tmp(vec.eltype(), {N});
    std::memset(&tmp.at(0), 0, vec.elsize() * N);
    int64_t gap = N / absl::bit_ceil(static_cast<size_t>(n));
    for (int64_t i = 0, j = 0; i < n; ++i, j += gap) {
      std::memcpy(/*dst*/ &tmp.at(j), /*src*/ &vec.at(i), vec.elsize());
    }
    encoder.Forward(tmp, &poly, /*scale*/ true);
  }
}

spu::NdArrayRef BinMatVecProtocol::Impl::DecodePolyToVector(
    spu::FieldType field, int64_t numel, const seal::Plaintext &poly,
    const spu::mpc::cheetah::VectorEncoder &encoder) const {
  int64_t N = encoder.poly_degree();
  SPU_ENFORCE(numel > 0 && numel <= N, "invalid size n={}, N={}", numel, N);
  size_t num_modulus = encoder.ms_helper().coeff_modulus_size();
  SPU_ENFORCE_EQ(poly.coeff_count(), num_modulus * N);
  auto poly_wrap = absl::MakeConstSpan(poly.data(), poly.coeff_count());
  if (N == numel) {
    return encoder.ms_helper().ModulusDownRNS(field, {numel}, poly_wrap);
  }

  // Skip the 0s between two elements
  spu::NdArrayRef ret = spu::mpc::ring_zeros(field, {numel});
  int64_t gap = N / absl::bit_ceil(static_cast<size_t>(numel));
  spu::NdArrayRef tmp =
      encoder.ms_helper().ModulusDownRNS(field, {N}, poly_wrap);
  for (int64_t i = 0, j = 0; i < numel; ++i, j += gap) {
    std::memcpy(&ret.at(i), &tmp.at(j), ret.elsize());
  }
  return ret;
}

void BinMatVecProtocol::Impl::A2hSend(const spu::NdArrayRef &ashare) {
  int64_t n = ashare.numel();
  if (n == 0) {
    return;
  }

  int64_t poly_deg_n = poly_degree_in(ring_bitwidth_);
  int64_t num_rlwes = CeilDiv(n, poly_deg_n);

  const auto &context = *in_context_;
  const auto &rlwe_sk = *in_skey_;
  const auto &vencoder = *in_vencoder_;

  std::vector<yacl::Buffer> ciphers(num_rlwes);
  auto enc_callback = [&](int64_t start, int64_t until) {
    using namespace spu::mpc;
    constexpr bool ntt = false;
    constexpr bool save_seed = true;
    seal::Plaintext pt;
    seal::Ciphertext ct;
    for (int64_t i = start; i < until; ++i) {
      int64_t bgn = i * poly_deg_n;
      int64_t end = std::min(bgn + poly_deg_n, n);
      EncodeVectorToPoly(ashare.slice({bgn}, {end}, {1}), vencoder, pt);
      spu::mpc::cheetah::SymmetricRLWEEncrypt(rlwe_sk, context, {&pt, 1}, ntt,
                                              save_seed, {&ct, 1});
      ciphers[i] = spu::mpc::cheetah::EncodeSEALObject(ct);
    }
    seal::util::seal_memzero(pt.data(), sizeof(uint64_t) * pt.coeff_count());
  };

  yacl::parallel_for(0, num_rlwes, enc_callback);

  [[maybe_unused]] size_t bytes_sent = 0;
  for (auto &c : ciphers) {
    bytes_sent += c.size();
  }

  int next = conn_->NextRank();
  for (int64_t i = 0; i < num_rlwes; i += kCtAsyncParallel) {
    auto tag = fmt::format("A2hSend_Batch{}", i / kCtAsyncParallel);
    int64_t this_batch = std::min<int64_t>(num_rlwes - i, kCtAsyncParallel);
    for (int64_t j = 1; j < this_batch; ++j) {
      conn_->SendAsync(next, ciphers[i + j - 1], tag);
    }
    conn_->Send(next, ciphers[i + this_batch - 1], tag);
  }

  SPDLOG_DEBUG("A2H ({} bit ring) on {} shares sent {:4f} MB ", ring_bitwidth_,
               n, static_cast<double>(bytes_sent) / 1024. / 1024.);
}

void BinMatVecProtocol::Impl::A2hRecv(const spu::NdArrayRef &ashare,
                                      absl::Span<seal::Ciphertext> hshare) {
  int64_t n = ashare.numel();
  if (n == 0) {
    return;
  }

  int64_t poly_deg_n = poly_degree_in(ring_bitwidth_);
  int64_t num_rlwes = CeilDiv(n, poly_deg_n);
  SPU_ENFORCE_EQ(num_rlwes, (int64_t)hshare.size(), "expected={} got={}",
                 num_rlwes, hshare.size());

  const auto &context = *in_context_;
  const auto &vencoder = *in_vencoder_;

  // Launch an IO thread
  std::future<void> io_task = std::async([&, this] {
    int next = conn_->NextRank();
    for (int64_t i = 0; i < num_rlwes; ++i) {
      auto recv = conn_->Recv(next, "A2hRecv");
      spu::mpc::cheetah::DecodeSEALObject(recv, context, hshare.data() + i);
    }
  });

  std::vector<seal::Plaintext> polys(num_rlwes);
  auto ecd_callback = [&](int64_t start, int64_t until) {
    using namespace spu::mpc;
    for (int64_t i = start; i < until; ++i) {
      int64_t bgn = i * poly_deg_n;
      int64_t end = std::min(bgn + poly_deg_n, n);
      EncodeVectorToPoly(ashare.slice({bgn}, {end}, {1}), vencoder, polys[i]);
    }
  };
  yacl::parallel_for(0, num_rlwes, ecd_callback);

  // Wait all RLWEs from the peer
  io_task.get();

  seal::Evaluator evaluator(context);
  for (int64_t i = 0; i < num_rlwes; ++i) {
    SPU_ENFORCE(not hshare[i].is_ntt_form() && hshare[i].size() == 2);
    SPU_ENFORCE(hshare[i].parms_id() == polys[i].parms_id());

    hshare[i].is_ntt_form() = true;  // foo SEAL on CKKS cipher
    evaluator.add_plain_inplace(hshare[i], polys[i]);
    hshare[i].is_ntt_form() = false;
  }
}

BinMatVecProtocol::BinMatVecProtocol(size_t ring_bitwidth,
                                     std::shared_ptr<yacl::link::Context> conn)
    : ring_bitwidth_(ring_bitwidth) {
  impl_ = std::make_shared<Impl>(ring_bitwidth, conn);
}

spu::NdArrayRef BinMatVecProtocol::Send(const spu::NdArrayRef &vec_in,
                                        int64_t dim_out, int64_t dim_in) {
  SPU_ENFORCE_EQ(vec_in.shape().ndim(), 1L);
  SPU_ENFORCE_EQ(vec_in.numel(), dim_in);

  auto eltype = vec_in.eltype();
  SPU_ENFORCE(eltype.isa<spu::RingTy>());
  SPU_ENFORCE_EQ(ring_bitwidth_,
                 spu::SizeOf(eltype.as<spu::RingTy>()->field()) * 8);
  return impl_->Send(vec_in, dim_out, dim_in);
}

spu::NdArrayRef BinMatVecProtocol::Recv(const spu::NdArrayRef &vec_in,
                                        int64_t dim_out, int64_t dim_in,
                                        const StlSparseMatrix &prv_bin_mat,
                                        absl::Span<const uint8_t> indicator) {
  SPU_ENFORCE_EQ(vec_in.shape().ndim(), 1L);
  SPU_ENFORCE_EQ(vec_in.numel(), dim_in);
  if (not indicator.empty()) {
    // mat * diag(indicator) should not be all zeros.
    SPU_ENFORCE(std::any_of(indicator.begin(), indicator.end(),
                            [](uint8_t x) { return x > 0; }),
                "empty matrix is not allowed");
  }

  auto eltype = vec_in.eltype();
  SPU_ENFORCE(eltype.isa<spu::RingTy>());
  SPU_ENFORCE_EQ(ring_bitwidth_, SizeOf(eltype.as<spu::RingTy>()->field()) * 8);
  return impl_->Recv(vec_in, dim_out, dim_in, prv_bin_mat, indicator);
}

void GenerateLWEKeySwitchKey(const seal::SecretKey &src_key,
                             const seal::SecretKey &dst_key,
                             const seal::SEALContext &src_context,
                             const seal::SEALContext &dst_context,
                             seal::KSwitchKeys &out, bool save_seed) {
  using namespace seal;
  using namespace seal::util;
  SPU_ENFORCE(src_context.parameters_set());
  SPU_ENFORCE(dst_context.parameters_set());

  auto src_dat = src_context.key_context_data();
  auto dst_dat = dst_context.key_context_data();
  const auto &src_parms = src_dat->parms();
  const auto &dst_parms = dst_dat->parms();

  size_t n = src_parms.poly_modulus_degree();
  size_t N = dst_parms.poly_modulus_degree();

  SPU_ENFORCE(N >= n, "require dst dim={} >= src dim={}", N, n);

  std::vector<uint64_t> src_key_non_ntt(n);
  std::copy_n(src_key.data().data(), n, src_key_non_ntt.data());
  inverse_ntt_negacyclic_harvey(src_key_non_ntt.data(),
                                src_dat->small_ntt_tables()[0]);

  auto ntt_tables = dst_dat->small_ntt_tables();
  auto &dst_modulus = dst_parms.coeff_modulus();
  size_t num_dst_modulus = dst_modulus.size();

  // Number of modulus for ciphertexts
  size_t num_src_ct_mod = src_parms.coeff_modulus().size();
  size_t num_dst_ct_mod = dst_parms.coeff_modulus().size();
  if (src_parms.use_special_prime()) num_src_ct_mod -= 1;
  if (dst_parms.use_special_prime()) num_dst_ct_mod -= 1;
  SPU_ENFORCE(num_src_ct_mod <= num_dst_ct_mod);

  std::vector<uint64_t> src_key_extent(N * num_dst_modulus, 0);
  for (size_t l = 0; l < num_dst_modulus; ++l) {
    uint64_t factor = 1;
    for (size_t j = num_src_ct_mod; j < num_dst_ct_mod; ++j) {
      factor =
          multiply_uint_mod(factor, dst_modulus[j].value(), dst_modulus[l]);
    }

    uint64_t negative_one = dst_modulus[l].value() - 1;
    uint64_t *dst_ptr = src_key_extent.data() + l * N;
    std::fill_n(dst_ptr, N, 0);

    // keep position 0 unchanged.
    dst_ptr[0] = SEAL_COND_SELECT(src_key_non_ntt[0] <= 1UL, src_key_non_ntt[0],
                                  negative_one);

    // move position [1, n) to [N - n + 1, N)
    std::transform(src_key_non_ntt.begin() + 1, src_key_non_ntt.end(),
                   dst_ptr + N - n + 1, [negative_one](uint64_t v) {
                     return SEAL_COND_SELECT(v <= 1UL, v, negative_one);
                   });

    if (factor != 1) {
      multiply_poly_scalar_coeffmod(dst_ptr, N, factor, dst_modulus[l],
                                    dst_ptr);
    }
    ntt_negacyclic_harvey(dst_ptr, ntt_tables[l]);
  }

  ConstRNSIter new_key(src_key_extent.data(), N);
  out.data().resize(1);
  GenerateOneKSKey(new_key, dst_key, dst_context, out.data()[0], save_seed);
  out.parms_id() = dst_context.key_parms_id();

  seal_memzero(src_key_non_ntt.data(),
               sizeof(uint64_t) * src_key_non_ntt.size());
  seal_memzero(src_key_extent.data(), sizeof(uint64_t) * src_key_extent.size());
}

void GenerateOneKSKey(seal::util::ConstRNSIter new_key,
                      const seal::SecretKey &secret_key,
                      const seal::SEALContext &context,
                      std::vector<seal::PublicKey> &destination,
                      bool save_seed) {
  using namespace seal;
  using namespace seal::util;
  SPU_ENFORCE(context.using_keyswitching());
  size_t coeff_count =
      context.key_context_data()->parms().poly_modulus_degree();
  size_t decomp_mod_count =
      context.first_context_data()->parms().coeff_modulus().size();
  const auto &key_context_data = *context.key_context_data();
  const auto &key_parms = key_context_data.parms();
  const auto &key_modulus = key_parms.coeff_modulus();

  // Size check
  SPU_ENFORCE(product_fits_in(coeff_count, decomp_mod_count));

  destination.resize(decomp_mod_count);

  SEAL_ITERATE(
      iter(new_key, key_modulus, destination, size_t(0)), decomp_mod_count,
      [&](const auto &I) {
        SEAL_ALLOCATE_GET_COEFF_ITER(temp, coeff_count,
                                     MemoryManager::GetPool());
        encrypt_zero_symmetric(secret_key, context, key_context_data.parms_id(),
                               /*ntt*/ true, save_seed, get<2>(I).data());
        uint64_t factor =
            barrett_reduce_64(key_modulus.back().value(), get<1>(I));
        multiply_poly_scalar_coeffmod(get<0>(I), coeff_count, factor, get<1>(I),
                                      temp);

        CoeffIter destination_iter = (*iter(get<2>(I).data()))[get<3>(I)];
        add_poly_coeffmod(destination_iter, temp, coeff_count, get<1>(I),
                          destination_iter);
      });
}

void LiftLWEDimension(const spu::mpc::cheetah::LWECt &src,
                      const seal::KSwitchKeys &lift_key,
                      const seal::SEALContext &src_context,
                      const seal::SEALContext &dst_context,
                      spu::mpc::cheetah::LWECt &dst) {
  size_t n = src_context.key_context_data()->parms().poly_modulus_degree();
  size_t N = dst_context.key_context_data()->parms().poly_modulus_degree();

  SPU_ENFORCE_EQ(src.poly_modulus_degree(), n);
  seal::Ciphertext src_rlwe;
  src.CastAsRLWE(src_context, /*multiplier*/ 1, &src_rlwe);
  SPU_ENFORCE(!src_rlwe.is_ntt_form());

  seal::Ciphertext extented_rlwe;
  extented_rlwe.resize(dst_context, dst_context.first_parms_id(),
                       src_rlwe.size());
  extented_rlwe.is_ntt_form() = false;

  const auto src_dat = src_context.first_context_data();
  const auto dst_dat = dst_context.first_context_data();
  const auto &dst_modulus = dst_dat->parms().coeff_modulus();

  const size_t num_src_ct_mod = src_dat->parms().coeff_modulus().size();
  const size_t num_modulus = extented_rlwe.coeff_modulus_size();
  std::vector<uint64_t> target(N * num_modulus, 0);

  for (size_t l = 0; l < num_modulus; ++l) {
    uint64_t factor = 1;
    for (size_t j = num_src_ct_mod; j < num_modulus; ++j) {
      factor = seal::util::multiply_uint_mod(factor, dst_modulus[j].value(),
                                             dst_modulus[l]);
    }
    uint64_t *target_ptr = target.data() + l * N;
    if (l < num_src_ct_mod) {
      std::copy_n(src_rlwe.data(1) + l * n, n, target_ptr);
      std::fill_n(target_ptr + n, N - n, 0UL);
    } else {
      SPU_ENFORCE(factor == 0);
      std::fill_n(target_ptr, N, 0UL);
    }
    extented_rlwe.data(0)[l * N] = seal::util::multiply_uint_mod(
        src_rlwe.data(0)[l * n], factor, dst_modulus[l]);
  }

  // (src_rlwe[0], 0) + src_rlwe[1] * ksk
  seal::Evaluator evaluator(dst_context);
  seal::util::ConstRNSIter target_iter(target.data(), N);
  // NOTE (lwj): we have patched seal/evaluator.h to export `switch_key_inplace`
  // as public function.
  CATCH_SEAL_ERROR(
      evaluator.switch_key_inplace(extented_rlwe, target_iter, lift_key, 0));

  dst = spu::mpc::cheetah::LWECt(extented_rlwe, 0, dst_context);
}

}  // namespace squirrel
