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

// Author: Wen-jie Lu(juhou)

#include "spu/mpc/beaver/beaver_he.h"

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
#include "yasl/utils/parallel.h"

#include "spu/core/xt_helper.h"
#include "spu/mpc/beaver/matvec.h"
#include "spu/mpc/beaver/modswitch_helper.h"
#include "spu/mpc/beaver/prg_tensor.h"
#include "spu/mpc/util/ring_ops.h"
#include "spu/mpc/util/seal_help.h"

namespace spu::mpc {

constexpr uint32_t kSmallRingBitLen = 36;
constexpr size_t kPolyDegree = 8192;
constexpr int kNoiseFloodRandomBits = 50;

// Erase the memory automatically
struct MemGuard {
  MemGuard(ArrayRef *obj) : obj_(obj) {}

  MemGuard(seal::Plaintext *pt) : pt_(pt) {}

  ~MemGuard() {
    try {
      if (obj_ && obj_->numel() > 0 && obj_->elsize() > 0) {
        auto ptr = reinterpret_cast<char *>(obj_->data());
        size_t nbytes =
            seal::util::mul_safe<size_t>(obj_->numel(), obj_->elsize());
        seal::util::seal_memzero(ptr, nbytes);
      }

      if (pt_ && pt_->coeff_count() > 0) {
        size_t nbytes =
            seal::util::mul_safe(pt_->coeff_count(), sizeof(uint64_t));
        seal::util::seal_memzero(pt_->data(), nbytes);
      }
    } catch (const std::exception &e) {
      SPDLOG_ERROR("Error in ~MemGuard(): {}", e.what());
    }
  };

  ArrayRef *obj_{nullptr};
  seal::Plaintext *pt_{nullptr};
};

template <typename T>
static T CeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

static seal::EncryptionParameters DecideSEALParameters(uint32_t ring_bitlen) {
  size_t poly_deg = kPolyDegree;
  auto scheme_type = seal::scheme_type::bfv;
  auto parms = seal::EncryptionParameters(scheme_type);
  std::vector<int> modulus_bits;
  if (poly_deg == 8192) {
    modulus_bits = {60, 49, 60};  // <= 218bits
  } else {
    YASL_THROW_LOGIC_ERROR("N={} is not supported yet", poly_deg);
  }
  parms.set_poly_modulus_degree(poly_deg);
  parms.set_coeff_modulus(seal::CoeffModulus::Create(poly_deg, modulus_bits));
  return parms;
}

static PrgSeed GetHardwareRandom128() {
  // NOTE(juhou) can we use thr rdseed instruction ?
  std::random_device rd;
  // call random_device four times, make sure uint128 is random in 2^128 set.
  uint64_t lhs = static_cast<uint64_t>(rd()) << 32 | rd();
  uint64_t rhs = static_cast<uint64_t>(rd()) << 32 | rd();
  return yasl::MakeUint128(lhs, rhs);
}

// Given the number of slots, and the shape of matrix, partition the matrix into
// subblocks that can fit into slots.
static std::array<size_t, 2> DecideMatrixPartition(size_t num_slots,
                                                   size_t nrows, size_t ncols) {
  YASL_ENFORCE(num_slots > 0 && IsTwoPower(num_slots));
  YASL_ENFORCE(nrows > 0 && ncols > 0);

  // NOTE(juhou): we pack the as much as possible columns into one polynomial.
  // This enables us to re-use `EncryptRandomArrayThenSend` for the vector.
  size_t optimal_col_extent = std::min(num_slots, ncols);
  size_t optimal_row_extent = std::min(nrows, num_slots);
  if (optimal_col_extent == num_slots) {
    optimal_row_extent = std::min(optimal_row_extent, num_slots / 2);
  }

  return {optimal_row_extent, optimal_col_extent};
}

static size_t CalNumDiagnoals(size_t num_slots, size_t nrows, size_t ncols) {
  auto submat_dims = DecideMatrixPartition(num_slots, nrows, ncols);
  size_t n_diags = 0;
  for (size_t rstart = 0; rstart < nrows; rstart += submat_dims[0]) {
    size_t row_end = std::min(nrows, rstart + submat_dims[0]);
    size_t row_extent = row_end - rstart;
    for (size_t cstart = 0; cstart < ncols; cstart += submat_dims[1]) {
      size_t col_end = std::min(ncols, cstart + submat_dims[1]);
      size_t col_extent = col_end - cstart;
      n_diags += Next2Pow(std::min(row_extent, col_extent));
    }
  }
  return n_diags;
}

static void TransposeInplace(ArrayRef mat, size_t nrows, size_t ncols) {
  YASL_ENFORCE_EQ((size_t)mat.numel(), nrows * ncols);
  const auto field = mat.eltype().as<Ring2k>()->field();
  DISPATCH_FM3264(field, "_", [&]() {
    auto xmat = xt_mutable_adapt<ring2k_t>(mat);
    xmat.reshape({nrows, ncols});
    auto xmatT = xt::eval(xt::transpose(xmat));
    std::copy_n(xmatT.begin(), xmatT.size(), xmat.data());
  });
}

struct BeaverHE::Impl {
 public:
  Impl(std::shared_ptr<yasl::link::Context> lctx)
      : lctx_(lctx), seed_(GetHardwareRandom128()), counter_(0) {
    parms_ = DecideSEALParameters(kSmallRingBitLen);
    // NOTE(juhou): by default we set up enough information for 32bit & 64bit
    // ring.
    ExpandSEALContexts(/*field_bitlen*/ 64);
    InitModSwitchHelper(/*field_bitlen*/ 32);
    InitModSwitchHelper(/*field_bitlen*/ 64);
  }

  size_t num_slots() const { return parms_.poly_modulus_degree(); }

  void ExpandSEALContexts(uint32_t field_bitlen);

  Beaver::Triple Mul(FieldType field, size_t size) {
    YASL_ENFORCE(size > 0);
    int nxt_rank = lctx_->NextRank();
    // Exchange HE ciphers [Delta*a0], [Delta*a1]
    // where a0, a1 uniform in [0, 2^k)
    auto a = EncryptRandomArrayThenSend(field, size);
    // Sample b0, b1 uniform in [0, 2^k)
    std::vector<seal::Plaintext> encoded_b;
    Options options;
    options.max_pack = num_slots();
    options.scale_delta = false;
    options.tiling = false;
    auto b = PrepareRandomElements(field, size, options, &encoded_b);
    // Receive [Delta*a] from the peer
    size_t payload_sze = encoded_b.size();
    std::vector<yasl::Buffer> recv_ct(payload_sze);
    for (size_t idx = 0; idx < payload_sze; ++idx) {
      recv_ct[idx] =
          lctx_->Recv(nxt_rank, fmt::format("recv from P{}", nxt_rank));
    }

    // Compute [Delta*a0*b1 + Delta*r0], [Delta*a1*b0 + Delta*r1]
    // where r0, r1 uniform in [0, 2^k)
    auto r = ElementMulThenResponse(field, size, recv_ct, encoded_b);
    for (size_t idx = 0; idx < payload_sze; ++idx) {
      recv_ct[idx] = lctx_->Recv(
          nxt_rank, fmt::format("recv response from P{}", nxt_rank));
    }

    // d0 = a0 * b1 + r1 mod 2^k
    // d1 = a1 * b0 + r0 mod 2^k
    auto d = DecryptArray(field, size, recv_ct);
    // c0 = a0 * b0 + d0 - r0
    // c1 = a1 * b1 + d1 - r1
    auto c = ring_sub(ring_add(ring_mul(a, b), d), r);

    return {a, b, c};
  }

  Beaver::Triple DotNative(FieldType field, size_t M, size_t N, size_t K) {
    int nxt_rank = lctx_->NextRank();
    const size_t size = K;
    const size_t num_splits = CeilDiv(size, num_slots());
    const size_t base_bitlen = FieldBitLen(field);
    const size_t num_seal_ctx = WorkingContextSize(base_bitlen);
    const size_t payload_sze = num_seal_ctx * num_splits;

    std::vector<seal::Plaintext> encoded_b(payload_sze * N);
    std::vector<ArrayRef> bs;
    for (size_t idx = 0; idx < N; ++idx) {
      auto ecd_b = absl::Span<seal::Plaintext>(
          encoded_b.data() + idx * payload_sze, payload_sze);
      Options options;
      options.max_pack = num_slots();
      options.scale_delta = false;
      options.tiling = false;
      bs.push_back(PrepareRandomElements(field, size, options, ecd_b));
    }

    auto a = ring_zeros(field, M * K);
    auto b = ring_zeros(field, K * N);
    auto c = ring_zeros(field, M * N);

    DISPATCH_FM3264(field, "Dot", [&]() {
      auto xa = xt_mutable_adapt<ring2k_t>(a);
      auto xb = xt_mutable_adapt<ring2k_t>(b);
      auto xc = xt_mutable_adapt<ring2k_t>(c);
      xa.reshape({M, K});
      xb.reshape({K, N});
      xc.reshape({M, N});
      for (size_t inp = 0; inp < M; ++inp) {
        auto _a = EncryptRandomArrayThenSend(field, size);
        xt::row(xa, inp) = xt_adapt<ring2k_t>(_a);

        std::vector<yasl::Buffer> recv_ct(payload_sze);
        for (size_t idx = 0; idx < payload_sze; ++idx) {
          recv_ct[idx] = lctx_->Recv(nxt_rank, "recv ct from peer");
        }

        for (size_t outp = 0; outp < N; ++outp) {
          auto ecd_b = absl::Span<const seal::Plaintext>(
              encoded_b.data() + outp * payload_sze, payload_sze);

          auto r = ElementMulThenResponse(field, size, recv_ct, ecd_b);
          std::vector<yasl::Buffer> response(payload_sze);
          for (size_t idx = 0; idx < payload_sze; ++idx) {
            response[idx] = lctx_->Recv(nxt_rank, "recv response peer");
          }
          // d0 = a0 * b1 + r1 mod 2^k
          auto d = DecryptArray(field, size, response);
          // the outp-th column of b
          auto _b = bs[outp];
          // c0 = a0 * b0 + d0 - r0
          auto _c = ring_sub(ring_add(ring_mul(_a, _b), d), r);
          xc(inp, outp) = xt::sum(xt_adapt<ring2k_t>(_c))();

          // assign once
          if (inp == 0) xt::col(xb, outp) = xt_adapt<ring2k_t>(_b);
        }
      }
      return;
    });

    return {a, b, c};
  }

  Beaver::Triple Dot(FieldType field, size_t M, size_t N, size_t K) {
    YASL_ENFORCE(M > 0 && N > 0 && K > 0);

    if (M * N < 16) {
      return DotNative(field, M, N, K);
    }

    // Compute A*B = C for A.shape= M*K, B.shape=K*N
    // When M > N, iterate along N-axis.
    // When M < N, iterate along M-axis, and we compute B^t*A^t = C^t.
    const size_t loop_dim = std::min(M, N);
    const size_t lhs_nrows = std::max(M, N);
    const size_t field_bitlen = FieldBitLen(field);
    const size_t num_seal_ctx = WorkingContextSize(field_bitlen);
    const int nxt_rank = lctx_->NextRank();
    LazyInitRotationKeys(static_cast<uint32_t>(field_bitlen));

    std::vector<seal::Plaintext> encoded_mat;
    auto lhs_mat = PrepareRandomMatrix(field, lhs_nrows, K, &encoded_mat);
    auto rhs_mat = ring_zeros(field, K * loop_dim);
    auto ans_mat = ring_zeros(field, M * N);

    auto submat_dims = DecideMatrixPartition(num_slots(), lhs_nrows, K);
    // TODO(juhou) To combine these multiple send & recv into one.
    for (size_t n = 0; n < loop_dim; ++n) {
      auto vec = EncryptRandomArrayThenSend(field, K, /*tiling*/ true);

      size_t payload_sze = CeilDiv(K, submat_dims[1]) * num_seal_ctx;
      std::vector<yasl::Buffer> recv_ct(payload_sze);
      for (size_t idx = 0; idx < payload_sze; ++idx) {
        recv_ct[idx] =
            lctx_->Recv(nxt_rank, fmt::format("recv from P{}", nxt_rank));
      }

      auto r = MatVecThenResponse(field, lhs_nrows, K, recv_ct, encoded_mat);

      size_t response_sze = CeilDiv(lhs_nrows, submat_dims[0]) * num_seal_ctx;
      recv_ct.resize(response_sze);
      for (size_t idx = 0; idx < response_sze; ++idx) {
        recv_ct[idx] =
            lctx_->Recv(nxt_rank, fmt::format("recv from P{}", nxt_rank));
      }

      auto d = DecryptVector(field, lhs_nrows, K, recv_ct);
      auto res_vec =
          ring_sub(ring_add(ring_mmul(lhs_mat, vec, lhs_nrows, 1, K), d), r);

      // Assign the sampled vector and resulting vector to `rhs_mat` and
      // `ans_mat`.
      DISPATCH_FM3264(field, "_", [&]() {
        auto xrhs_mat = xt_mutable_adapt<ring2k_t>(rhs_mat);
        auto xans_mat = xt_mutable_adapt<ring2k_t>(ans_mat);

        // We store rsh_mat and ans_mat in the transposed form
        // so that we can assign the row directly.
        xrhs_mat.reshape({loop_dim, K});
        xans_mat.reshape({loop_dim, lhs_nrows});

        xt::row(xrhs_mat, n) = xt_adapt<ring2k_t>(vec);
        xt::row(xans_mat, n) = xt_adapt<ring2k_t>(res_vec);
      });
    }

    if (lhs_nrows == M) {
      // rhs_mat and ans_mat are transposed.
      TransposeInplace(rhs_mat, loop_dim, K);
      TransposeInplace(ans_mat, loop_dim, lhs_nrows);
      return {lhs_mat, rhs_mat, ans_mat};
    } else {
      // But in this case we compute B^t*A^t = C^t,
      // so we only need to transpose lhs_mat.
      TransposeInplace(lhs_mat, lhs_nrows, K);
      return {rhs_mat, lhs_mat, ans_mat};
    }
  }

 protected:
  inline uint32_t FieldBitLen(FieldType f) { return 8 * SizeOf(f); }

  inline uint32_t TotalCRTBitLen(uint32_t field_bitlen) {
    // NOTE(juhou) We can use a smaller value for approximated Beaver
    const int approximated_LSB = 0;
    const int margins_for_full_random = 15;
    return 2 * field_bitlen + margins_for_full_random - approximated_LSB;
  }

  void InitModSwitchHelper(uint32_t field_bitlen);

  void LazyInitRotationKeys(uint32_t field_bitlen);

  inline uint32_t WorkingContextSize(uint32_t field_bitlen) {
    uint32_t target_bitlen = TotalCRTBitLen(field_bitlen);
    YASL_ENFORCE(target_bitlen <= current_crt_plain_bitlen_,
                 "Call ExpandSEALContexts first");
    return CeilDiv(target_bitlen, kSmallRingBitLen);
  }

  struct Options {
    size_t max_pack = 0;
    bool scale_delta = false;
    bool tiling = false;
  };

  // Sample a random array r of `size` field elements.
  // The array will be partitioned into sub-array of `options.max_pack` length.
  //   If `options.max_pack = 0`, set it to `num_slots`.
  //   If `options.scale_delta = true`e, scale up it by Delta.
  //   If `options.tiling = false`, zero-padding is used to align to `num_slots`
  //   elements. Otherwise, repeat-tiling the sub-array to `num_slots` elements.
  ArrayRef PrepareRandomElements(FieldType field, size_t size,
                                 const Options &options,
                                 std::vector<seal::Plaintext> *encoded_rnd);

  ArrayRef PrepareRandomElements(FieldType field, size_t size,
                                 const Options &options,
                                 absl::Span<seal::Plaintext> encoded_rnd);

  ArrayRef DoPrepareRandomMask(FieldType field, size_t size,
                               const Options &options,
                               std::vector<seal::Plaintext> *encoded_mask) {
    const size_t max_pack = options.max_pack;
    const size_t num_splits = CeilDiv(size, max_pack);
    const size_t field_bitlen = FieldBitLen(field);
    const size_t num_seal_ctx = WorkingContextSize(field_bitlen);
    const size_t num_polys = num_seal_ctx * num_splits;
    YASL_ENFORCE(ms_helpers_.count(field_bitlen) > 0);
    encoded_mask->resize(num_polys);

    // sample r from [0, P) in the RNS format
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
        if (options.tiling) {
          // tiling: first zero-pad to 2-power elements then repeat
          size_t padded_sze = Next2Pow(len);
          size_t nrep = u64tmp.size() / padded_sze;
          std::fill_n(u64tmp.data() + len, padded_sze - len, 0);
          for (size_t r = 1; r < nrep; ++r) {
            std::copy_n(u64tmp.data(), padded_sze,
                        u64tmp.data() + r * padded_sze);
          }
        } else {
          std::fill_n(u64tmp.data() + len, u64tmp.size() - len, 0);
        }

        bfv_encoders_[cidx]->encode(u64tmp, encoded_mask->at(offset + j));
      }
    }

    // convert x \in [0, P) to [0, 2^k) by round(2^k*x/P)
    auto &ms_helper = ms_helpers_.find(field_bitlen)->second;

    std::vector<uint64_t> u64tmp(size);
    absl::Span<uint64_t> out_span(u64tmp.data(), u64tmp.size());
    absl::Span<const uint64_t> inp(random_rns.data(), random_rns.size());
    ms_helper.ModulusDownRNS(inp, out_span);

    return DISPATCH_FM3264(field, "PrepareRandomMask", [&]() {
      auto out = ring_zeros(field, size);
      auto out_ptr = reinterpret_cast<ring2k_t *>(out.data());
      std::transform(
          u64tmp.begin(), u64tmp.end(), out_ptr,
          [](uint64_t u) -> ring2k_t { return static_cast<ring2k_t>(u); });
      return out;
    });
  }

  ///// Mul releated methods /////
  // Sample random array of `size` elements in the field.
  // Then send the encrypted array to the peer.
  // Return the sampled array.
  ArrayRef EncryptRandomArrayThenSend(FieldType field, size_t size,
                                      bool tiling = false);

  // Sample random array `r` of `size` elements in the field.
  // Then compute ciphers*plains + r and response the result to the peer.
  // Return teh sampled array `r`.
  ArrayRef ElementMulThenResponse(FieldType field, size_t size,
                                  absl::Span<const yasl::Buffer> ciphers,
                                  absl::Span<const seal::Plaintext> plains);

  ArrayRef PrepareRandomMask(FieldType field, size_t size,
                             std::vector<seal::Plaintext> *encoded_mask) {
    Options options;
    options.max_pack = num_slots();
    options.tiling = false;
    return DoPrepareRandomMask(field, size, options, encoded_mask);
  }

  ArrayRef DecryptArray(FieldType field, size_t size,
                        const std::vector<yasl::Buffer> &ct_array);
  ///// Mul releated methods /////

  ///// Dot releated methods /////
  ArrayRef MatVecThenResponse(FieldType field, size_t nrows, size_t ncols,
                              absl::Span<const yasl::Buffer> ciphers,
                              absl::Span<const seal::Plaintext> plains);

  ArrayRef PrepareRandomMatrix(FieldType field, size_t nrows, size_t ncols,
                               std::vector<seal::Plaintext> *encoded_rnd);

  ArrayRef PrepareRandomMask(FieldType field, size_t nrows, size_t ncols,
                             std::vector<seal::Plaintext> *encoded_mask) {
    auto submat_dims = DecideMatrixPartition(num_slots(), nrows, ncols);
    Options options;
    options.max_pack = Next2Pow(submat_dims[0]);
    options.tiling = true;
    return DoPrepareRandomMask(field, nrows, options, encoded_mask);
  }

  ArrayRef DecryptVector(FieldType field, size_t nrows, size_t ncols,
                         const std::vector<yasl::Buffer> &ct_vector);
  ///// Dot releated methods /////

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
    if (counter_ > kPRNG_THREASHOLD) {
      seed_ = GetHardwareRandom128();
      counter_ = 0;
    }
    // NOTE(juhou): do we need to replay the PRNG ?
    PrgArrayDesc prg_desc;
    return prgCreateArray(field, size, seed_, &counter_, &prg_desc);
  }

  void NoiseFloodCiphertext(seal::Ciphertext &ct,
                            const seal::SEALContext &context) {
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
    MemGuard guard(&random);

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

  void RandomizeCipherForDecryption(seal::Ciphertext &ct, size_t cidx) {
    auto &seal_cntxt = seal_cntxts_.at(cidx);
    auto context_data = seal_cntxt.last_context_data();
    yasl::CheckNotNull(context_data.get());
    seal::Evaluator evaluator(seal_cntxt);
    // 1. Add statistical independent randomness
    NoiseFloodCiphertext(ct, seal_cntxt);

    // 2. Drop all but keep one moduli
    if (ct.coeff_modulus_size() > 1) {
      evaluator.mod_switch_to_inplace(ct, context_data->parms_id());
    }

    // 3. Add zero-encryption for re-randomization
    seal::Ciphertext zero_enc;
    CATCH_SEAL_ERROR(
        pk_encryptors_[cidx]->encrypt_zero(ct.parms_id(), zero_enc));
    CATCH_SEAL_ERROR(evaluator.add_inplace(ct, zero_enc));

    // 4. Truncate for smaller communication
    TruncateBFVForDecryption(ct, seal_cntxt);
  }

 private:
  std::shared_ptr<yasl::link::Context> lctx_;

  PrgSeed seed_;

  mutable std::mutex counter_lock_;

  PrgCounter counter_;

  seal::EncryptionParameters parms_;

  uint32_t current_crt_plain_bitlen_{0};

  // SEAL's contexts for ZZ_{2^k}
  mutable std::shared_mutex context_lock_;
  std::vector<seal::SEALContext> seal_cntxts_;

  // own secret key
  std::shared_ptr<seal::SecretKey> secret_key_;
  // the public key received from the opposite party
  std::shared_ptr<seal::PublicKey> pair_public_key_;
  // the galois keys received from the opposite party
  std::shared_ptr<seal::GaloisKeys> pair_galois_key_;

  std::unordered_map<size_t, ModulusSwitchHelper> ms_helpers_;

  std::vector<std::shared_ptr<seal::Encryptor>> sym_encryptors_;
  std::vector<std::shared_ptr<seal::Decryptor>> decryptors_;
  std::vector<std::shared_ptr<seal::Encryptor>> pk_encryptors_;
  std::vector<std::shared_ptr<seal::BatchEncoder>> bfv_encoders_;
};

void BeaverHE::Impl::LazyInitRotationKeys(uint32_t field_bitlen) {
  uint32_t target_plain_bitlen = TotalCRTBitLen(field_bitlen);
  uint32_t num_keys = CeilDiv(target_plain_bitlen, kSmallRingBitLen);
  YASL_ENFORCE(current_crt_plain_bitlen_ >= target_plain_bitlen);
  YASL_ENFORCE(seal_cntxts_.size() >= num_keys);

  {
    std::shared_lock<std::shared_mutex> guard(context_lock_);
    if (pair_galois_key_) return;
  }
  std::unique_lock<std::shared_mutex> guard(context_lock_);
  // double-checking
  if (pair_galois_key_) return;

  // Generate the rotation key and exchange it
  int nxt_rank = lctx_->NextRank();
  seal::KeyGenerator keygen(seal_cntxts_[0], *secret_key_);
  auto galois_key = keygen.create_galois_keys();
  pair_galois_key_ = std::make_shared<seal::GaloisKeys>(galois_key.obj());
  auto gk_buf = EncodeSEALObject(*pair_galois_key_);
  lctx_->SendAsync(nxt_rank, gk_buf, "send gk");
  gk_buf = lctx_->Recv(nxt_rank, "recv gk");
  DecodeSEALObject(gk_buf, seal_cntxts_[0], pair_galois_key_.get());
  spdlog::info("BeaverHE lazy init rotation keys for {} bit length",
               target_plain_bitlen);
}

void BeaverHE::Impl::InitModSwitchHelper(uint32_t field_bitlen) {
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

  // NOTE(juhou): we needs `upper_half_threshold()` from CKKS
  auto parms = seal::EncryptionParameters(seal::scheme_type::ckks);
  parms.set_poly_modulus_degree(parms_.poly_modulus_degree());
  parms.set_coeff_modulus(crt_modulus);

  seal::SEALContext crt_context(parms, false, seal::sec_level_type::none);
  ms_helpers_.emplace(std::make_pair(
      field_bitlen, ModulusSwitchHelper(crt_context, field_bitlen)));
}

void BeaverHE::Impl::ExpandSEALContexts(uint32_t field_bitlen) {
  uint32_t target_plain_bitlen = TotalCRTBitLen(field_bitlen);
  {
    std::shared_lock<std::shared_mutex> guard(context_lock_);
    if (current_crt_plain_bitlen_ >= target_plain_bitlen) return;
  }
  std::unique_lock<std::shared_mutex> guard(context_lock_);
  if (current_crt_plain_bitlen_ >= target_plain_bitlen) return;

  uint32_t num_seal_ctx = CeilDiv(target_plain_bitlen, kSmallRingBitLen);
  std::vector<int> crt_moduli_bits(num_seal_ctx, kSmallRingBitLen);
  int last_plain = std::max<int>(
      20, target_plain_bitlen - (num_seal_ctx - 1) * kSmallRingBitLen);
  crt_moduli_bits.back() = last_plain;
  spdlog::info("BeaverHE uses {} modulus ({} bit each) for {} bit length",
               num_seal_ctx, kSmallRingBitLen, target_plain_bitlen);

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
      int nxt_rank = lctx_->NextRank();
      lctx_->SendAsync(nxt_rank, pk_buf, "send Pk");
      pk_buf = lctx_->Recv(nxt_rank, "recv pk");
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

ArrayRef BeaverHE::Impl::EncryptRandomArrayThenSend(FieldType field,
                                                    size_t num_elts,
                                                    bool tiling) {
  YASL_ENFORCE(num_elts > 0, "BeaverHE: empty array");

  // Sample random then encrypt
  Options options;
  options.max_pack = num_slots();
  options.scale_delta = true;
  options.tiling = tiling;
  std::vector<seal::Plaintext> ecd_rnd;
  auto random = PrepareRandomElements(field, num_elts, options, &ecd_rnd);

  size_t field_bitlen = FieldBitLen(field);
  size_t num_seal_ctx = WorkingContextSize(field_bitlen);
  size_t num_ct_per_ctx = ecd_rnd.size() / num_seal_ctx;
  YASL_ENFORCE(ecd_rnd.size() % num_seal_ctx == 0, "Internal bug");

  std::vector<yasl::Buffer> payload(ecd_rnd.size());
  yasl::parallel_for(
      0, num_seal_ctx, 1, [&](size_t cntxt_bgn, size_t cntxt_end) {
        for (size_t c = cntxt_bgn; c < cntxt_end; ++c) {
          size_t offset = c * num_ct_per_ctx;
          for (size_t idx = 0; idx < num_ct_per_ctx; ++idx) {
            // erase the random from memory
            MemGuard guard(&ecd_rnd[offset + idx]);
            auto ct =
                sym_encryptors_[c]->encrypt_symmetric(ecd_rnd[offset + idx]);
            payload.at(offset + idx) = EncodeSEALObject(ct.obj());
          }
        }
      });

  int nxt_rank = lctx_->NextRank();
  for (auto &ct : payload) {
    lctx_->SendAsync(nxt_rank, ct, fmt::format("Send to P{}", nxt_rank));
  }

  return random;
}

ArrayRef BeaverHE::Impl::PrepareRandomElements(
    FieldType field, size_t num_elts, const Options &options,
    std::vector<seal::Plaintext> *ecd_random) {
  YASL_ENFORCE(num_elts > 0, "BeaverHE: empty array");
  yasl::CheckNotNull(ecd_random);

  const size_t max_pack = options.max_pack > 0 ? options.max_pack : num_slots();
  const size_t num_splits = CeilDiv(num_elts, max_pack);
  const size_t field_bitlen = FieldBitLen(field);
  const size_t num_seal_ctx = WorkingContextSize(field_bitlen);
  const size_t num_polys = num_seal_ctx * num_splits;
  ecd_random->resize(num_polys);
  return PrepareRandomElements(
      field, num_elts, options,
      absl::Span<seal::Plaintext>(ecd_random->data(), num_polys));
}

ArrayRef BeaverHE::Impl::PrepareRandomElements(
    FieldType field, size_t num_elts, const Options &options,
    absl::Span<seal::Plaintext> ecd_random) {
  YASL_ENFORCE(num_elts > 0, "BeaverHE: empty array");

  const size_t max_pack = options.max_pack > 0 ? options.max_pack : num_slots();
  const size_t num_splits = CeilDiv(num_elts, max_pack);
  const size_t field_bitlen = FieldBitLen(field);
  const size_t num_seal_ctx = WorkingContextSize(field_bitlen);
  const size_t num_polys = num_seal_ctx * num_splits;
  YASL_ENFORCE(ecd_random.size() == num_polys, "ecd_random buffer size");
  YASL_ENFORCE(ms_helpers_.count(field_bitlen) > 0);

  auto random = CPRNG(field, num_elts);

  // NOTE(juhou): DISPATCH_FM3264 will expand a ring2k_t type.
  DISPATCH_FM3264(field, "PrepareRandomElements", [&]() {
    using ring2k_u = typename std::make_unsigned<ring2k_t>::type;
    auto &ms_helper = ms_helpers_.find(field_bitlen)->second;

    yasl::parallel_for(
        0, num_seal_ctx, 1, [&](size_t cntxt_bgn, size_t cntxt_end) {
          auto seal_pool = seal::MemoryManager::GetPool();
          std::vector<uint64_t> u64tmp(num_slots());

          for (size_t cidx = cntxt_bgn; cidx < cntxt_end; ++cidx) {
            const size_t offset = cidx * num_splits;

            for (size_t idx = 0; idx < num_splits; ++idx) {
              size_t bgn = idx * max_pack;
              size_t end = std::min(num_elts, bgn + max_pack);

              auto slice = xt_adapt<ring2k_u>(random.slice(bgn, end));
              absl::Span<const ring2k_u> src(slice.data(), slice.size());
              absl::Span<uint64_t> dst(u64tmp.data(), slice.size());

              if (options.scale_delta) {
                ms_helper.ModulusUpAt(src, cidx, dst);
              } else {
                ms_helper.CenteralizeAt(src, cidx, dst);
              }

              if (options.tiling) {
                size_t padded_sze = Next2Pow(src.size());
                std::fill_n(u64tmp.data() + src.size(), padded_sze - src.size(),
                            0);
                size_t nrep = num_slots() / padded_sze;

                // repeat the vector to full fill all the slots
                for (size_t r = 1; r < nrep; ++r) {
                  std::copy_n(u64tmp.data(), padded_sze,
                              u64tmp.data() + r * padded_sze);
                }
              } else {
                // just zero-padding the rest
                std::fill_n(u64tmp.data() + src.size(),
                            u64tmp.size() - src.size(), 0);
              }

              CATCH_SEAL_ERROR(bfv_encoders_[cidx]->encode(
                  u64tmp, ecd_random[offset + idx]));
            }
          }
        });
    return;
  });

  return random;
}

ArrayRef BeaverHE::Impl::PrepareRandomMatrix(
    FieldType field, size_t nrows, size_t ncols,
    std::vector<seal::Plaintext> *encoded_rnd) {
  yasl::CheckNotNull(encoded_rnd);

  const size_t field_bitlen = FieldBitLen(field);
  const size_t num_seal_ctx = WorkingContextSize(field_bitlen);
  const size_t max_pack = num_slots();
  YASL_ENFORCE(ms_helpers_.count(field_bitlen) > 0);
  const ModulusSwitchHelper &ms_helper = ms_helpers_.find(field_bitlen)->second;

  YASL_ENFORCE(nrows > 0 && ncols > 0);
  YASL_ENFORCE(pair_public_key_ != nullptr);
  // YASL_ENFORCE(pair_galois_keys_.size() >= num_seal_ctx);

  size_t n_polys = CalNumDiagnoals(max_pack, nrows, ncols);
  encoded_rnd->resize(n_polys * num_seal_ctx);
  auto rnd_mat = CPRNG(field, nrows * ncols);

  auto submat_dims = DecideMatrixPartition(max_pack, nrows, ncols);
  yasl::parallel_for(
      0, num_seal_ctx, 1, [&](size_t cntxt_bgn, size_t cntxt_end) {
        MatVecHelper::MatViewMeta meta;
        meta.is_transposed = false;
        meta.num_rows = nrows;
        meta.num_cols = ncols;

        for (size_t cidx = cntxt_bgn; cidx < cntxt_end; ++cidx) {
          MatVecProtocol matvec_prot(*pair_galois_key_, seal_cntxts_[cidx]);
          auto dst_poly_ptr = encoded_rnd->data() + cidx * n_polys;
          meta.row_start = 0;

          for (; meta.row_start < nrows; meta.row_start += submat_dims[0]) {
            size_t row_end = std::min(nrows, meta.row_start + submat_dims[0]);
            meta.row_extent = row_end - meta.row_start;
            meta.col_start = 0;
            for (; meta.col_start < ncols; meta.col_start += submat_dims[1]) {
              size_t col_end = std::min(ncols, meta.col_start + submat_dims[1]);
              meta.col_extent = col_end - meta.col_start;

              size_t n_diags =
                  Next2Pow(std::min(meta.row_extent, meta.col_extent));
              absl::Span<seal::Plaintext> dst_wrap(dst_poly_ptr, n_diags);
              matvec_prot.EncodeSubMatrix(rnd_mat, meta, ms_helper, cidx,
                                          dst_wrap);
              dst_poly_ptr += n_diags;
            }
          }
        }
      });
  return rnd_mat;
}

ArrayRef BeaverHE::Impl::ElementMulThenResponse(
    FieldType field, size_t num_elts, absl::Span<const yasl::Buffer> ciphers,
    absl::Span<const seal::Plaintext> plains) {
  YASL_ENFORCE(ciphers.size() > 0, "BeaverHE: empty cipher");
  YASL_ENFORCE(plains.size() == ciphers.size(),
               "BeaverHE: ct/pt size mismatch");

  const size_t max_pack = num_slots();
  const size_t num_splits = CeilDiv(num_elts, max_pack);
  const size_t field_bitlen = FieldBitLen(field);
  const size_t num_seal_ctx = WorkingContextSize(field_bitlen);
  const size_t num_ciphers = num_seal_ctx * num_splits;
  YASL_ENFORCE(ciphers.size() == num_ciphers,
               fmt::format("ElementMulThenResponse: expect {} != {}",
                           num_ciphers, ciphers.size()));

  std::vector<seal::Plaintext> ecd_polys;
  auto rnd_mask = PrepareRandomMask(field, num_elts, &ecd_polys);
  YASL_ENFORCE(ecd_polys.size() == num_ciphers,
               "BeaverHE: encoded poly size mismatch");

  std::vector<yasl::Buffer> response(num_ciphers);
  // NOTE(juhou): DISPATCH_FM3264 will expand a ring2k_t type.
  DISPATCH_FM3264(field, "MulThenMask", [&]() {
    yasl::parallel_for(
        0, num_seal_ctx, 1, [&](size_t cntxt_bgn, size_t cntxt_end) {
          seal::Ciphertext ct;
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
                  evaluator.add_plain_inplace(ct, ecd_polys[offset + idx]));
              // re-randomize the ciphertext
              RandomizeCipherForDecryption(ct, cidx);
              response[offset + idx] = EncodeSEALObject(ct);
            }
          }
        });
    return;
  });

  int nxt_rank = lctx_->NextRank();
  for (auto &ct : response) {
    lctx_->SendAsync(nxt_rank, ct, fmt::format("Send to P{}", nxt_rank));
  }

  for (auto &pt : ecd_polys) {
    MemGuard{&pt};
  }

  return rnd_mask;
}

ArrayRef BeaverHE::Impl::MatVecThenResponse(
    FieldType field, size_t nrows, size_t ncols,
    absl::Span<const yasl::Buffer> ciphers,
    absl::Span<const seal::Plaintext> plains) {
  const size_t field_bitlen = FieldBitLen(field);
  const size_t num_seal_ctx = WorkingContextSize(field_bitlen);
  const size_t max_pack = num_slots();
  YASL_ENFORCE(pair_public_key_ != nullptr);

  const size_t n_polys = CalNumDiagnoals(max_pack, nrows, ncols);
  auto submat_dims = DecideMatrixPartition(max_pack, nrows, ncols);

  size_t num_inp_ct_per_cntxt = CeilDiv(ncols, submat_dims[1]);
  YASL_ENFORCE_EQ(ciphers.size(), num_inp_ct_per_cntxt * num_seal_ctx);
  YASL_ENFORCE_EQ(plains.size(), n_polys * num_seal_ctx);

  // Decode to Ciphertext and then convert to NTT form
  std::vector<seal::Ciphertext> inp_ciphers(ciphers.size());
  yasl::parallel_for(
      0, num_seal_ctx, 1, [&](size_t cntxt_bgn, size_t cntxt_end) {
        for (size_t cidx = cntxt_bgn; cidx < cntxt_end; ++cidx) {
          seal::Evaluator evaluator(seal_cntxts_[cidx]);
          size_t offset = cidx * num_inp_ct_per_cntxt;

          for (size_t idx = 0; idx < num_inp_ct_per_cntxt; ++idx) {
            seal::Ciphertext &inp_ct = inp_ciphers[offset + idx];
            DecodeSEALObject(ciphers[offset + idx], seal_cntxts_[cidx],
                             &inp_ct);
            if (!inp_ct.is_ntt_form()) {
              evaluator.transform_to_ntt_inplace(inp_ct);
            }
          }
        }
      });

  size_t num_oup_ct_per_cntxt = CeilDiv(nrows, submat_dims[0]);

  std::vector<seal::Plaintext> ecd_polys;
  auto rnd_mask = PrepareRandomMask(field, nrows, ncols, &ecd_polys);
  YASL_ENFORCE_EQ(ecd_polys.size(), num_oup_ct_per_cntxt * num_seal_ctx);

  std::vector<seal::Ciphertext> oup_ciphers(num_oup_ct_per_cntxt *
                                            num_seal_ctx);
  yasl::parallel_for(
      0, num_seal_ctx, 1, [&](size_t cntxt_bgn, size_t cntxt_end) {
        MatVecHelper::MatViewMeta meta;
        meta.is_transposed = false;
        meta.num_rows = nrows;
        meta.num_cols = ncols;

        for (size_t cidx = cntxt_bgn; cidx < cntxt_end; ++cidx) {
          seal::Evaluator evaluator(seal_cntxts_[cidx]);
          MatVecProtocol matvec_prot(*pair_galois_key_, seal_cntxts_[cidx]);

          auto mat_pt_ptr = plains.data() + cidx * n_polys;
          auto oup_ct_ptr = oup_ciphers.data() + cidx * num_oup_ct_per_cntxt;
          auto rnd_mask_ptr = ecd_polys.data() + cidx * num_oup_ct_per_cntxt;
          meta.row_start = 0;

          // loop row blocks
          for (; meta.row_start < nrows; meta.row_start += submat_dims[0]) {
            auto inp_ct_ptr = inp_ciphers.data() + cidx * num_inp_ct_per_cntxt;
            size_t row_end = std::min(nrows, meta.row_start + submat_dims[0]);
            meta.row_extent = row_end - meta.row_start;
            meta.col_start = 0;

            // loop column blocks
            for (; meta.col_start < ncols; meta.col_start += submat_dims[1]) {
              size_t col_end = std::min(ncols, meta.col_start + submat_dims[1]);
              meta.col_extent = col_end - meta.col_start;

              size_t n_diags =
                  Next2Pow(std::min(meta.row_extent, meta.col_extent));
              absl::Span<const seal::Plaintext> mat_wrap(mat_pt_ptr, n_diags);
              assert(mat_wrap.begin() < plains.end());
              assert(mat_wrap.end() <= plains.end());

              // accumulate along the column blocks
              if (oup_ct_ptr->size() == 0) {
                matvec_prot.Compute(*inp_ct_ptr++, mat_wrap, meta, oup_ct_ptr);
              } else {
                seal::Ciphertext tmp;
                matvec_prot.Compute(*inp_ct_ptr++, mat_wrap, meta, &tmp);
                evaluator.add_inplace(*oup_ct_ptr, tmp);
              }

              mat_pt_ptr += n_diags;
            }

            if (oup_ct_ptr->is_ntt_form()) {
              evaluator.transform_from_ntt_inplace(*oup_ct_ptr);
            }

            // H2A
            CATCH_SEAL_ERROR(
                evaluator.add_plain_inplace(*oup_ct_ptr, *rnd_mask_ptr));
            RandomizeCipherForDecryption(*oup_ct_ptr, cidx);

            oup_ct_ptr += 1;
            rnd_mask_ptr += 1;
          }
        }
      });

  int nxt_rank = lctx_->NextRank();
  for (auto &ct : oup_ciphers) {
    lctx_->SendAsync(nxt_rank, EncodeSEALObject(ct),
                     fmt::format("Send to P{}", nxt_rank));
  }

  return rnd_mask;
}

ArrayRef BeaverHE::Impl::DecryptVector(
    FieldType field, size_t nrows, size_t ncols,
    const std::vector<yasl::Buffer> &ct_vector) {
  const size_t field_bitlen = FieldBitLen(field);
  const size_t num_seal_ctx = WorkingContextSize(field_bitlen);
  const size_t max_pack = num_slots();
  auto submat_dims = DecideMatrixPartition(max_pack, nrows, ncols);
  size_t num_vec_per_cntxt = CeilDiv(nrows, submat_dims[0]);
  YASL_ENFORCE_EQ(ct_vector.size(), num_vec_per_cntxt * num_seal_ctx);
  YASL_ENFORCE(ms_helpers_.count(field_bitlen) > 0);

  auto rns_temp = ring_zeros(FieldType::FM64, nrows * num_seal_ctx);
  auto xrns_temp = xt_mutable_adapt<uint64_t>(rns_temp);

  yasl::parallel_for(
      0, num_seal_ctx, 1, [&](size_t cntxt_bgn, size_t cntxt_end) {
        for (size_t cidx = cntxt_bgn; cidx < cntxt_end; ++cidx) {
          const size_t offset = cidx * num_vec_per_cntxt;
          auto rns_slice = xt::view(
              xrns_temp, xt::range(cidx * nrows, cidx * nrows + nrows));

          seal::Plaintext pt;
          seal::Ciphertext ct;
          std::vector<uint64_t> subarray(max_pack, 0);

          for (size_t idx = 0; idx < num_vec_per_cntxt; ++idx) {
            DecodeSEALObject(ct_vector.at(offset + idx), seal_cntxts_[cidx],
                             &ct);
            CATCH_SEAL_ERROR(decryptors_[cidx]->decrypt(ct, pt));
            CATCH_SEAL_ERROR(bfv_encoders_[cidx]->decode(pt, subarray));

            size_t bgn = idx * submat_dims[0];
            size_t end = std::min(nrows, bgn + submat_dims[0]);
            size_t len = end - bgn;
            std::copy_n(subarray.data(), len, rns_slice.begin() + bgn);
          }
        }
      });

  auto &ms_helper = ms_helpers_.find(field_bitlen)->second;
  std::vector<uint64_t> u64tmp(nrows);
  absl::Span<uint64_t> out_span(u64tmp.data(), u64tmp.size());
  absl::Span<const uint64_t> inp(xrns_temp.data(), xrns_temp.size());
  ms_helper.ModulusDownRNS(inp, out_span);

  return DISPATCH_FM3264(field, "DecryptArray", [&]() {
    auto out = ring_zeros(field, nrows);
    auto out_ptr = reinterpret_cast<ring2k_t *>(out.data());
    std::transform(
        u64tmp.begin(), u64tmp.end(), out_ptr,
        [](uint64_t u) -> ring2k_t { return static_cast<ring2k_t>(u); });
    return out;
  });
}

ArrayRef BeaverHE::Impl::DecryptArray(
    FieldType field, size_t size, const std::vector<yasl::Buffer> &ct_array) {
  const size_t max_pack = num_slots();
  const size_t num_splits = CeilDiv(size, max_pack);
  const size_t field_bitlen = FieldBitLen(field);
  const size_t num_seal_ctx = WorkingContextSize(field_bitlen);
  const size_t num_ciphers = num_seal_ctx * num_splits;
  YASL_ENFORCE(ct_array.size() == num_ciphers,
               "BeaverHE: cipher size mismatch");
  YASL_ENFORCE(ms_helpers_.count(field_bitlen) > 0);

  auto rns_temp = ring_zeros(FieldType::FM64, size * num_seal_ctx);
  auto xrns_temp = xt_mutable_adapt<uint64_t>(rns_temp);

  yasl::parallel_for(
      0, num_seal_ctx, 1, [&](size_t cntxt_bgn, size_t cntxt_end) {
        // Loop each SEALContext
        // For each context, we obtain `size` uint64 from `num_splits` polys.
        // Each poly will decode to `max_pack` uint64, i.e., `max_pack *
        // num_splits
        // >= size`.
        for (size_t cidx = cntxt_bgn; cidx < cntxt_end; ++cidx) {
          const size_t offset = cidx * num_splits;
          auto ctx_slice =
              xt::view(xrns_temp, xt::range(cidx * size, cidx * size + size));

          seal::Plaintext pt;
          seal::Ciphertext ct;
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
  std::vector<uint64_t> u64tmp(size);
  absl::Span<uint64_t> out_span(u64tmp.data(), u64tmp.size());
  absl::Span<const uint64_t> inp(xrns_temp.data(), xrns_temp.size());
  ms_helper.ModulusDownRNS(inp, out_span);

  return DISPATCH_FM3264(field, "DecryptArray", [&]() {
    auto out = ring_zeros(field, size);
    auto out_ptr = reinterpret_cast<ring2k_t *>(out.data());
    std::transform(
        u64tmp.begin(), u64tmp.end(), out_ptr,
        [](uint64_t u) -> ring2k_t { return static_cast<ring2k_t>(u); });
    return out;
  });
}

BeaverHE::BeaverHE(std::shared_ptr<yasl::link::Context> lctx)
    : impl_(std::make_shared<Impl>(lctx)) {}

Beaver::Triple BeaverHE::Mul(FieldType field, size_t size) {
  YASL_ENFORCE(field == FieldType::FM32 or field == FieldType::FM64);
  yasl::CheckNotNull(impl_.get());
  return impl_->Mul(field, size);
}

Beaver::Triple BeaverHE::Dot(FieldType field, size_t M, size_t N, size_t K) {
  YASL_ENFORCE(field == FieldType::FM32 or field == FieldType::FM64);
  yasl::CheckNotNull(impl_.get());
  return impl_->Dot(field, M, N, K);
}

Beaver::Triple BeaverHE::And(FieldType field, size_t size) {
  // YASL_THROW("Not supported.");
  return {
      ring_zeros(field, size),
      ring_zeros(field, size),
      ring_zeros(field, size),
  };
}

Beaver::Pair BeaverHE::Trunc(FieldType field, size_t size, size_t bits) {
  // YASL_THROW("Not supported.");
  return {
      ring_zeros(field, size),
      ring_zeros(field, size),
  };
}

ArrayRef BeaverHE::RandBit(FieldType field, size_t size) {
  // YASL_THROW("Not supported.");
  return ring_zeros(field, size);
}

}  // namespace spu::mpc
