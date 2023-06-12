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
#include "libspu/mpc/cheetah/arith/cheetah_dot.h"

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
#include "seal/util/rlwe.h"
#include "spdlog/spdlog.h"
#include "yacl/link/link.h"

#include "libspu/core/shape_util.h"
#include "libspu/mpc/cheetah/arith/common.h"
#include "libspu/mpc/cheetah/arith/conv2d_prot.h"
#include "libspu/mpc/cheetah/arith/matmat_prot.h"
#include "libspu/mpc/cheetah/rlwe/modswitch_helper.h"
#include "libspu/mpc/cheetah/rlwe/utils.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {

struct CheetahDot::Impl : public EnableCPRNG {
 public:
  const bool kUseModDownOptimization = true;
  static constexpr size_t kParallelStride = 1;
  static constexpr size_t kCtAsyncParallel = 8;

  explicit Impl(std::shared_ptr<yacl::link::Context> lctx)
      : lctx_(std::move(lctx)) {}

  ~Impl() = default;

  std::unique_ptr<Impl> Fork();

  struct Conv2DMeta {
    Conv2DProtocol::Meta prot_meta;
    bool is_tensor;
    size_t n_tensor_poly;
    size_t n_kernel_poly;
    size_t n_output_poly;
  };

  // Compute C = A*B where |A|=dims[0]xdims[1], |B|=dims[1]xdims[2
  ArrayRef DotOLE(const ArrayRef &prv, yacl::link::Context *conn,
                  const Shape3D &dim3, bool is_lhs);

  ArrayRef Conv2dOLE(const ArrayRef &inp, yacl::link::Context *conn,
                     int64_t input_batch, const Shape3D &tensor_shape,
                     int64_t num_kernels, const Shape3D &kernel_shape,
                     const Shape2D &window_strides, bool is_tensor);

  ArrayRef doConv2dOLEForEncryptor(FieldType field,
                                   absl::Span<const RLWEPt> poly_ntt,
                                   const Conv2DMeta &meta,
                                   const Conv2DProtocol &conv2d,
                                   yacl::link::Context *conn);

  template <typename T0, typename T1>
  void doConv2dOLECtPtMul(absl::Span<const T0> tensors,
                          absl::Span<const T1> kernels, const Conv2DMeta &meta,
                          const Conv2DProtocol &conv2d, absl::Span<RLWECt> out);

  ArrayRef doConv2dOLEForEvaluator(FieldType field,
                                   absl::Span<const RLWEPt> poly_ntt,
                                   const Conv2DMeta &meta,
                                   const Conv2DProtocol &conv2d,
                                   yacl::link::Context *conn);

  void encodeBatchInput(const ArrayRef &batch_inp, const Conv2DMeta &meta,
                        const Conv2DProtocol &conv2d, bool need_encrypt,
                        absl::Span<RLWEPt> out);

  ArrayRef parseBatchedConv2dResult(FieldType field, const Conv2DMeta &meta,
                                    const Conv2DProtocol &prot,
                                    absl::Span<const RLWEPt> polys);

  static seal::EncryptionParameters DecideSEALParameters(uint32_t ring_bitlen) {
    size_t poly_deg;
    std::vector<int> modulus_bits;
    // NOTE(juhou): we need Q=sum(modulus_bits) > 2*k for multiplying two k-bit
    // elements.
    // 1. We need the (N, Q) pair satisifies the security.
    //    Check `seal/util/globals.cpp` for the recommendation HE parameters.
    // 2. We prefer modulus_bits[i] to be around 49-bit aiming to use AVX512 for
    // acceleration if avaiable.
    // 3. We need some bits for margin. That is Q > 2*k + margin for errorless
    // w.h.p. We set margin=32bit
    if (ring_bitlen <= 32) {
      poly_deg = 4096;
      // ~ 64 + 32 bit
      modulus_bits = {59, 37};
    } else if (ring_bitlen <= 64) {
      poly_deg = 8192;
      // ~ 128 + 32 bit
      modulus_bits = {57, 57, 45};
    } else {
      poly_deg = 16384;
      // ~ 256 + 30 bit
      modulus_bits = {59, 59, 59, 59, 49};
    }

    auto scheme_type = seal::scheme_type::ckks;
    auto parms = seal::EncryptionParameters(scheme_type);

    parms.set_use_special_prime(false);
    parms.set_poly_modulus_degree(poly_deg);
    parms.set_coeff_modulus(seal::CoeffModulus::Create(poly_deg, modulus_bits));
    return parms;
  }

  static inline uint32_t FieldBitLen(FieldType f) { return 8 * SizeOf(f); }

  void LazyInit(size_t field_bitlen);

  static void SubPlainInplace(RLWECt &ct, const RLWEPt &pt,
                              const seal::SEALContext &context) {
    SPU_ENFORCE(ct.parms_id() == pt.parms_id());
    auto cntxt_dat = context.get_context_data(ct.parms_id());
    SPU_ENFORCE(cntxt_dat != nullptr);
    const auto &parms = cntxt_dat->parms();
    const auto &modulus = parms.coeff_modulus();
    size_t num_coeff = ct.poly_modulus_degree();
    size_t num_modulus = ct.coeff_modulus_size();

    for (size_t l = 0; l < num_modulus; ++l) {
      auto *op0 = ct.data(0) + l * num_coeff;
      const auto *op1 = pt.data() + l * num_coeff;
      seal::util::sub_poly_coeffmod(op0, op1, num_coeff, modulus[l], op0);
    }
  }

  void KeepModulus(RLWECt &ct, size_t num_to_keep,
                   const seal::SEALContext &context) const {
    SPU_ENFORCE(num_to_keep >= 1 && num_to_keep <= ct.coeff_modulus_size());
    if (num_to_keep == ct.coeff_modulus_size()) {
      // nothing to do
      return;
    }

    auto cntxt = context.get_context_data(ct.parms_id());
    YACL_ENFORCE(cntxt != nullptr);
    size_t index = cntxt->chain_index();
    YACL_ENFORCE((index + 1) >= num_to_keep);

    auto target_context = cntxt;
    auto pool =
        seal::MemoryManager::GetPool(seal::mm_prof_opt::mm_force_thread_local);

    while (target_context->chain_index() >= num_to_keep) {
      using namespace seal::util;
      auto rns_tool = target_context->rns_tool();
      auto ntt_tables = target_context->small_ntt_tables();
      if (ct.is_ntt_form()) {
        SEAL_ITERATE(iter(ct), ct.size(), [&](auto I) {
          rns_tool->divide_and_round_q_last_ntt_inplace(I, ntt_tables, pool);
        });
      } else {
        SEAL_ITERATE(iter(ct), ct.size(), [&](auto I) {
          rns_tool->divide_and_round_q_last_inplace(I, pool);
        });
      }

      auto next_context = target_context->next_context_data();
      SPU_ENFORCE(next_context != nullptr);

      RLWECt next_ct(pool);
      next_ct.resize(context, next_context->parms_id(), ct.size());
      SEAL_ITERATE(iter(ct, next_ct), ct.size(), [&](auto I) {
        set_poly(get<0>(I), ct.poly_modulus_degree(),
                 ct.coeff_modulus_size() - 1, get<1>(I));
      });
      next_ct.is_ntt_form() = ct.is_ntt_form();
      target_context = next_context;
      std::swap(next_ct, ct);
    }
    SPU_ENFORCE_EQ(num_to_keep, ct.coeff_modulus_size());
  }

  // Enc(Delta*m) -> Enc(Delta*m - r), r where r is sampled from Rq
  // One share of `m` is round(r/Delta) mod t
  // The other share of `m` is Dec(Enc(Delta*m - r))
  void H2A(absl::Span<RLWECt> ct, absl::Span<RLWEPt> rnd_mask,
           size_t target_modulus_size, const seal::PublicKey &pk,
           const seal::SEALContext &context) {
    seal::Evaluator evaluator(context);
    size_t num_poly = ct.size();
    SPU_ENFORCE(num_poly > 0);
    SPU_ENFORCE_EQ(rnd_mask.size(), num_poly);

    yacl::parallel_for(
        0, num_poly, kParallelStride, [&](size_t bgn, size_t end) {
          for (size_t idx = bgn; idx < end; ++idx) {
            // decrease q <- q' for efficiency
            if (ct[idx].is_ntt_form()) {
              evaluator.transform_from_ntt_inplace(ct[idx]);
            }
            KeepModulus(ct[idx], target_modulus_size, context);

            // TODO(juhou): improve the performance of pk encryption of zero.
            // ct <- ct + enc(0)
            RLWECt zero_ct;
            seal::util::encrypt_zero_asymmetric(pk, context, ct[idx].parms_id(),
                                                ct[idx].is_ntt_form(), zero_ct);
            evaluator.add_inplace(ct[idx], zero_ct);
            SPU_ENFORCE(!ct[idx].is_ntt_form());

            // sample r <- Rq
            // (ct[0] - r, ct[1]) <- ct
            UniformPoly(context, &rnd_mask[idx], ct[idx].parms_id());
            SubPlainInplace(ct[idx], rnd_mask[idx], context);
          }
        });
  }

 private:
  std::shared_ptr<yacl::link::Context> lctx_;

  mutable std::shared_mutex context_lock_;
  // field_bitlen -> functor mapping
  std::unordered_map<size_t, std::shared_ptr<seal::SEALContext>> seal_cntxts_;
  std::unordered_map<size_t, std::shared_ptr<seal::SecretKey>> secret_keys_;
  std::unordered_map<size_t, std::shared_ptr<seal::PublicKey>> peer_pub_keys_;
  // ModulusSwitchHelper for encoding
  std::unordered_map<size_t, std::shared_ptr<ModulusSwitchHelper>> ecd_mswh_;
  // ModulusSwitchHelper for decoding
  std::unordered_map<size_t, std::shared_ptr<ModulusSwitchHelper>> dcd_mswh_;
  std::unordered_map<size_t, std::shared_ptr<seal::Encryptor>> sym_encryptors_;
  std::unordered_map<size_t, std::shared_ptr<seal::Decryptor>> decryptors_;
};

std::unique_ptr<CheetahDot::Impl> CheetahDot::Impl::Fork() {
  auto f = std::make_unique<Impl>(lctx_->Spawn());
  if (seal_cntxts_.size() == 0) return f;
  std::unique_lock<std::shared_mutex> guard(context_lock_);

  f->seal_cntxts_ = seal_cntxts_;
  f->secret_keys_ = secret_keys_;
  f->peer_pub_keys_ = peer_pub_keys_;
  f->ecd_mswh_ = ecd_mswh_;
  f->dcd_mswh_ = dcd_mswh_;
  f->sym_encryptors_ = sym_encryptors_;
  f->decryptors_ = decryptors_;
  return f;
}

void CheetahDot::Impl::LazyInit(size_t field_bitlen) {
  {
    std::shared_lock<std::shared_mutex> guard(context_lock_);
    if (seal_cntxts_.find(field_bitlen) != seal_cntxts_.end()) {
      return;
    }
  }
  // double-checking
  std::unique_lock<std::shared_mutex> guard(context_lock_);
  if (seal_cntxts_.find(field_bitlen) != seal_cntxts_.end()) {
    return;
  }

  auto parms = DecideSEALParameters(field_bitlen);
  auto *this_context =
      new seal::SEALContext(parms, true, seal::sec_level_type::none);
  seal::KeyGenerator keygen(*this_context);
  auto *rlwe_sk = new seal::SecretKey(keygen.secret_key());

  auto pk = keygen.create_public_key();
  // NOTE(juhou): we patched seal/util/serializable.h
  auto pk_buf = EncodeSEALObject(pk.obj());
  // exchange the public key
  int nxt_rank = lctx_->NextRank();
  lctx_->SendAsync(nxt_rank, pk_buf, "send Pk");
  pk_buf = lctx_->Recv(nxt_rank, "recv pk");
  auto peer_public_key = std::make_shared<seal::PublicKey>();
  DecodeSEALObject(pk_buf, *this_context, peer_public_key.get());

  auto modulus = parms.coeff_modulus();
  if (parms.use_special_prime()) {
    modulus.pop_back();
  }
  size_t ecd_modulus_sze = modulus.size();
  parms.set_coeff_modulus(modulus);
  seal::SEALContext ecd_ms_context(parms, false, seal::sec_level_type::none);
  if (kUseModDownOptimization) {
    // NOTE: we can drop some modulus before H2A
    modulus.pop_back();
    if (field_bitlen > 64) {
      modulus.pop_back();
    }
  }
  size_t dcd_modulus_sze = modulus.size();
  parms.set_coeff_modulus(modulus);
  seal::SEALContext dcd_ms_context(parms, false, seal::sec_level_type::none);

  seal_cntxts_.emplace(field_bitlen, this_context);
  secret_keys_.emplace(field_bitlen, rlwe_sk);
  peer_pub_keys_.emplace(field_bitlen, peer_public_key);
  ecd_mswh_.emplace(field_bitlen,
                    new ModulusSwitchHelper(ecd_ms_context, field_bitlen));
  dcd_mswh_.emplace(field_bitlen,
                    new ModulusSwitchHelper(dcd_ms_context, field_bitlen));

  sym_encryptors_.emplace(field_bitlen,
                          new seal::Encryptor(*this_context, *rlwe_sk));
  decryptors_.emplace(field_bitlen,
                      new seal::Decryptor(*this_context, *rlwe_sk));

  SPDLOG_INFO("CheetahDot uses {}@{} modulus {} degree for {} bit ring",
              ecd_modulus_sze, dcd_modulus_sze, parms.poly_modulus_degree(),
              field_bitlen);
}

ArrayRef CheetahDot::Impl::DotOLE(const ArrayRef &prv_mat,
                                  yacl::link::Context *conn,
                                  const Shape3D &dim3, bool is_lhs) {
  if (conn == nullptr) {
    conn = lctx_.get();
  }
  int nxt_rank = conn->NextRank();
  auto eltype = prv_mat.eltype();
  SPU_ENFORCE(eltype.isa<RingTy>(), "must be ring_type, got={}", eltype);
  SPU_ENFORCE(prv_mat.numel() > 0);

  if (is_lhs) {
    SPU_ENFORCE_EQ(prv_mat.numel(), dim3[0] * dim3[1]);
  } else {
    SPU_ENFORCE_EQ(prv_mat.numel(), dim3[1] * dim3[2]);
  }

  auto field = eltype.as<Ring2k>()->field();
  const size_t field_bitlen = FieldBitLen(field);
  LazyInit(field_bitlen);

  const auto &this_context = *seal_cntxts_.find(field_bitlen)->second;
  auto &this_encryptor = sym_encryptors_.find(field_bitlen)->second;
  auto &this_decryptor = decryptors_.find(field_bitlen)->second;
  auto &this_ecd_ms = ecd_mswh_.find(field_bitlen)->second;
  auto &this_dcd_ms = dcd_mswh_.find(field_bitlen)->second;
  seal::Evaluator evaluator(this_context);

  MatMatProtocol matmat(this_context, *this_ecd_ms, /*mont*/ true);
  MatMatProtocol::Meta meta;
  meta.dims = dim3;
  auto subshape = matmat.GetSubMatShape(meta);
  size_t lhs_n = matmat.GetLeftSize(meta, subshape);
  size_t rhs_n = matmat.GetRightSize(meta, subshape);
  size_t out_n = matmat.GetOutSize(meta, subshape);
  bool to_encrypt_lhs = lhs_n < rhs_n;
  bool need_encrypt = (is_lhs ^ to_encrypt_lhs) == 0;

  std::vector<RLWEPt> encoded_mat(is_lhs ? lhs_n : rhs_n);
  if (is_lhs) {
    matmat.EncodeLHS(prv_mat, meta, need_encrypt, absl::MakeSpan(encoded_mat));
  } else {
    matmat.EncodeRHS(prv_mat, meta, need_encrypt, absl::MakeSpan(encoded_mat));
  }

  // convert local poly to NTT form to perform encryption / multiplication.
  yacl::parallel_for(0, encoded_mat.size(), kParallelStride,
                     [&](size_t bgn, size_t end) {
                       for (size_t i = bgn; i < end; ++i) {
                         NttInplace(encoded_mat[i], this_context);
                         if (not need_encrypt) {
                           matmat.Montgomerize({&encoded_mat[i], 1});
                         }
                       }
                     });

  if (need_encrypt) {
    // send ct
    for (size_t i = 0; i < encoded_mat.size(); ++i) {
      auto ct = this_encryptor->encrypt_symmetric(encoded_mat[i]).obj();
      auto ct_s = EncodeSEALObject(ct);
      conn->SendAsync(nxt_rank, ct_s, "send encrypted mat");
    }

    // wait for result
    std::vector<RLWECt> recv_ct(kCtAsyncParallel);
    std::vector<RLWEPt> result_poly(out_n);
    for (size_t i = 0; i < out_n; i += kCtAsyncParallel) {
      size_t this_batch = std::min(out_n - i, kCtAsyncParallel);
      for (size_t j = 0; j < this_batch; ++j) {
        auto ct_s = conn->Recv(nxt_rank, "recv result mat");
        DecodeSEALObject(ct_s, this_context, &recv_ct[j]);
      }

      yacl::parallel_for(
          0, this_batch, kParallelStride, [&](size_t bgn, size_t end) {
            for (size_t j = bgn; j < end; ++j) {
              if (not recv_ct[j].is_ntt_form()) {
                evaluator.transform_to_ntt_inplace(recv_ct[j]);
              }
              this_decryptor->decrypt(recv_ct[j], result_poly[i + j]);
            }
          });
    }

    // non-ntt form for ParseResult
    yacl::parallel_for(0, out_n, kParallelStride, [&](size_t bgn, size_t end) {
      for (size_t i = bgn; i < end; ++i) {
        InvNttInplace(result_poly[i], this_context);
      }
    });

    auto ret = matmat.ParseResult(field, meta, absl::MakeSpan(result_poly),
                                  *this_dcd_ms);
    return ret;
  }

  // recv ct from peer
  std::vector<RLWECt> encrypted_mat(is_lhs ? rhs_n : lhs_n);
  for (size_t i = 0; i < encrypted_mat.size(); ++i) {
    auto ct_s = conn->Recv(nxt_rank, "recv encrypted mat");
    DecodeSEALObject(ct_s, this_context, &encrypted_mat[i]);
  }

  std::vector<RLWECt> result_ct(out_n);
  if (is_lhs) {
    matmat.Compute(encoded_mat, encrypted_mat, meta, absl::MakeSpan(result_ct));
  } else {
    matmat.Compute(encrypted_mat, encoded_mat, meta, absl::MakeSpan(result_ct));
  }

  const auto &this_pk = peer_pub_keys_.find(field_bitlen)->second;

  std::vector<RLWEPt> mask_mat(out_n);
  H2A(absl::MakeSpan(result_ct), absl::MakeSpan(mask_mat),
      this_dcd_ms->coeff_modulus_size(), *this_pk, this_context);
  matmat.ExtractLWEsInplace(meta, absl::MakeSpan(result_ct));

  for (size_t i = 0; i < out_n; i += kCtAsyncParallel) {
    // NOTE(juhou): we do not send too much ct with Async
    size_t this_batch = std::min(out_n - i, kCtAsyncParallel);
    auto ct_s = EncodeSEALObject(result_ct[i]);
    conn->Send(nxt_rank, ct_s, "send result mat");

    for (size_t j = 1; j < this_batch; ++j) {
      auto ct_s = EncodeSEALObject(result_ct[i + j]);
      conn->SendAsync(nxt_rank, ct_s, "send result mat");
    }
  }

  return matmat.ParseResult(field, meta, absl::MakeSpan(mask_mat),
                            *this_dcd_ms);
}

void CheetahDot::Impl::encodeBatchInput(const ArrayRef &batch_inp,
                                        const Conv2DMeta &meta,
                                        const Conv2DProtocol &conv2d,
                                        bool need_encrypt,
                                        absl::Span<RLWEPt> out) {
  size_t input_batch = meta.prot_meta.input_batch;
  size_t num_poly_per_input = meta.n_tensor_poly / input_batch;
  SPU_ENFORCE_EQ(out.size(), meta.n_tensor_poly);

  const size_t numel = calcNumel(meta.prot_meta.input_shape);
  yacl::parallel_for(
      0, input_batch, kParallelStride, [&](size_t bgn, size_t end) {
        for (size_t ib = bgn; ib < end; ++ib) {
          absl::Span<RLWEPt> one_input_polys{
              out.data() + ib * num_poly_per_input, num_poly_per_input};
          conv2d.EncodeInput(batch_inp.slice(ib * numel, (ib + 1) * numel),
                             meta.prot_meta, need_encrypt, one_input_polys);
        }
      });
}

template <typename T0, typename T1>
void CheetahDot::Impl::doConv2dOLECtPtMul(absl::Span<const T0> tensors,
                                          absl::Span<const T1> kernels,
                                          const Conv2DMeta &meta,
                                          const Conv2DProtocol &conv2d,
                                          absl::Span<RLWECt> out) {
  const size_t input_batch = meta.prot_meta.input_batch;
  const size_t n_poly_per_tensor = meta.n_tensor_poly / input_batch;
  const size_t n_poly_per_out = meta.n_output_poly / input_batch;
  yacl::parallel_for(
      0, input_batch, kParallelStride, [&](size_t bgn, size_t end) {
        for (size_t ib = bgn; ib < end; ++ib) {
          absl::Span<const T0> one_input_polys{
              tensors.data() + ib * n_poly_per_tensor, n_poly_per_tensor};

          absl::Span<RLWECt> one_output_polys{out.data() + ib * n_poly_per_out,
                                              n_poly_per_out};

          conv2d.Compute(one_input_polys, kernels, meta.prot_meta,
                         one_output_polys);
        }
      });
}

ArrayRef CheetahDot::Impl::doConv2dOLEForEvaluator(
    FieldType field, absl::Span<const RLWEPt> poly_ntt, const Conv2DMeta &meta,
    const Conv2DProtocol &conv2d, yacl::link::Context *conn) {
  const size_t field_bitlen = FieldBitLen(field);
  const auto &this_context = *seal_cntxts_.find(field_bitlen)->second;
  const auto &this_dcd_ms = dcd_mswh_.find(field_bitlen)->second;
  const int nxt_rank = conn->NextRank();

  SPU_ENFORCE_EQ(poly_ntt.size(),
                 meta.is_tensor ? meta.n_tensor_poly : meta.n_kernel_poly);

  std::vector<RLWECt> recv_ct(meta.is_tensor ? meta.n_kernel_poly
                                             : meta.n_tensor_poly);

  for (size_t i = 0; i < recv_ct.size(); ++i) {
    auto ct_s = conn->Recv(nxt_rank, "recv encrypted mat");
    DecodeSEALObject(ct_s, this_context, &recv_ct[i]);
  }

  std::vector<RLWECt> result_ct(meta.n_output_poly);
  if (meta.is_tensor) {
    doConv2dOLECtPtMul<RLWEPt, RLWECt>(poly_ntt, absl::MakeSpan(recv_ct), meta,
                                       conv2d, absl::MakeSpan(result_ct));
  } else {
    doConv2dOLECtPtMul<RLWECt, RLWEPt>(absl::MakeSpan(recv_ct), poly_ntt, meta,
                                       conv2d, absl::MakeSpan(result_ct));
  }

  const auto &this_pk = peer_pub_keys_.find(field_bitlen)->second;

  std::vector<RLWEPt> mask_tensor(meta.n_output_poly);
  H2A(absl::MakeSpan(result_ct), absl::MakeSpan(mask_tensor),
      this_dcd_ms->coeff_modulus_size(), *this_pk, this_context);

  size_t n_poly_per_out = meta.n_output_poly / meta.prot_meta.input_batch;
  for (int64_t ib = 0; ib < meta.prot_meta.input_batch; ++ib) {
    absl::Span<RLWECt> one_output_polys{result_ct.data() + ib * n_poly_per_out,
                                        n_poly_per_out};
    conv2d.ExtractLWEsInplace(meta.prot_meta, one_output_polys);
  }

  for (size_t i = 0; i < meta.n_output_poly; i += kCtAsyncParallel) {
    size_t this_batch = std::min(meta.n_output_poly - i, kCtAsyncParallel);
    auto ct_s = EncodeSEALObject(result_ct[i]);
    conn->Send(nxt_rank, ct_s, "send result tensor");
    for (size_t j = 1; j < this_batch; ++j) {
      auto ct_s = EncodeSEALObject(result_ct[i + j]);
      conn->SendAsync(nxt_rank, ct_s, "send result tensor");
    }
  }

  return parseBatchedConv2dResult(field, meta, conv2d,
                                  absl::MakeSpan(mask_tensor));
}

ArrayRef CheetahDot::Impl::doConv2dOLEForEncryptor(
    FieldType field, absl::Span<const RLWEPt> poly_ntt, const Conv2DMeta &meta,
    const Conv2DProtocol &conv2d, yacl::link::Context *conn) {
  const size_t field_bitlen = FieldBitLen(field);
  const auto &this_context = *seal_cntxts_.find(field_bitlen)->second;
  auto &this_encryptor = sym_encryptors_.find(field_bitlen)->second;
  auto &this_decryptor = decryptors_.find(field_bitlen)->second;
  seal::Evaluator evaluator(this_context);

  const size_t num_ct_to_send = poly_ntt.size();
  const int nxt_rank = conn->NextRank();

  // send ct
  {
    std::vector<yacl::Buffer> ct_to_send(kCtAsyncParallel);
    for (size_t i = 0; i < num_ct_to_send; i += kCtAsyncParallel) {
      size_t this_batch = std::min(num_ct_to_send - i, kCtAsyncParallel);
      yacl::parallel_for(
          0, this_batch, kParallelStride, [&](size_t bgn, size_t end) {
            for (size_t k = bgn; k < end; ++k) {
              auto ct =
                  this_encryptor->encrypt_symmetric(poly_ntt[i + k]).obj();
              ct_to_send[k] = EncodeSEALObject(ct);
            }
          });

      conn->Send(nxt_rank, ct_to_send[0], "Conv2dOLE::Send ct");
      for (size_t k = 1; k < this_batch; ++k) {
        conn->SendAsync(nxt_rank, ct_to_send[k], "Conv2dOLE::Send ct");
      }
    }
  }

  // wait for result
  std::vector<RLWECt> recv_ct(kCtAsyncParallel);
  std::vector<RLWEPt> result_poly(meta.n_output_poly);
  for (size_t i = 0; i < meta.n_output_poly; i += kCtAsyncParallel) {
    size_t this_batch = std::min(meta.n_output_poly - i, kCtAsyncParallel);
    for (size_t j = 0; j < this_batch; ++j) {
      auto ct_s = conn->Recv(nxt_rank, "Conv2dOLE::Recv");
      DecodeSEALObject(ct_s, this_context, &recv_ct[j]);
    }

    yacl::parallel_for(
        0, this_batch, kParallelStride, [&](size_t bgn, size_t end) {
          for (size_t j = bgn; j < end; ++j) {
            evaluator.transform_to_ntt_inplace(recv_ct[j]);
            this_decryptor->decrypt(recv_ct[j], result_poly[i + j]);
          }
        });
  }

  // non-ntt form for parseBatchedConv2dResult
  yacl::parallel_for(0, meta.n_output_poly, kParallelStride,
                     [&](size_t bgn, size_t end) {
                       for (size_t i = bgn; i < end; ++i) {
                         InvNttInplace(result_poly[i], this_context);
                       }
                     });

  return parseBatchedConv2dResult(field, meta, conv2d,
                                  absl::MakeSpan(result_poly));
}

ArrayRef CheetahDot::Impl::Conv2dOLE(
    const ArrayRef &inp, yacl::link::Context *conn, int64_t input_batch,
    const Shape3D &tensor_shape, int64_t num_kernels,
    const Shape3D &kernel_shape, const Shape2D &window_strides,
    bool is_tensor) {
  if (conn == nullptr) {
    conn = lctx_.get();
  }

  auto eltype = inp.eltype();
  SPU_ENFORCE(eltype.isa<RingTy>(), "must be ring_type, got={}", eltype);
  SPU_ENFORCE(input_batch > 0 && num_kernels > 0);
  if (is_tensor) {
    SPU_ENFORCE_EQ(inp.numel(), calcNumel(tensor_shape) * input_batch);
  } else {
    SPU_ENFORCE_EQ(inp.numel(), calcNumel(kernel_shape) * num_kernels);
  }

  auto field = eltype.as<Ring2k>()->field();
  const size_t field_bitlen = FieldBitLen(field);
  LazyInit(field_bitlen);

  const auto &this_context = *seal_cntxts_.find(field_bitlen)->second;
  auto &this_ecd_ms = ecd_mswh_.find(field_bitlen)->second;
  Conv2DProtocol conv2d(this_context, *this_ecd_ms);

  Conv2DMeta meta;
  meta.prot_meta.input_batch = input_batch;
  meta.prot_meta.num_kernels = num_kernels;
  meta.prot_meta.input_shape = tensor_shape;
  meta.prot_meta.kernel_shape = kernel_shape;
  meta.prot_meta.window_strides = window_strides;

  auto subshape = conv2d.GetSubTensorShape(meta.prot_meta);
  meta.n_tensor_poly =
      input_batch * conv2d.GetInputSize(meta.prot_meta, subshape);
  meta.n_output_poly =
      input_batch * conv2d.GetOutSize(meta.prot_meta, subshape);
  meta.n_kernel_poly = conv2d.GetKernelSize(meta.prot_meta, subshape);
  meta.is_tensor = is_tensor;

  bool to_encrypt_tensor = meta.n_tensor_poly < meta.n_kernel_poly;
  bool need_encrypt = !(is_tensor ^ to_encrypt_tensor);
  std::vector<RLWEPt> encoded_poly;
  if (is_tensor) {
    encoded_poly.resize(meta.n_tensor_poly);
    encodeBatchInput(inp, meta, conv2d, need_encrypt,
                     absl::MakeSpan(encoded_poly));
  } else {
    encoded_poly.resize(meta.n_kernel_poly);
    conv2d.EncodeKernels(inp, meta.prot_meta, need_encrypt,
                         absl::MakeSpan(encoded_poly));
  }

  // convert local poly to NTT form to perform multiplication.
  yacl::parallel_for(0, encoded_poly.size(), kParallelStride,
                     [&](size_t bgn, size_t end) {
                       for (size_t i = bgn; i < end; ++i) {
                         NttInplace(encoded_poly[i], this_context);
                       }
                     });
  if (need_encrypt) {
    return doConv2dOLEForEncryptor(field, encoded_poly, meta, conv2d, conn);
  }
  return doConv2dOLEForEvaluator(field, encoded_poly, meta, conv2d, conn);
}

ArrayRef CheetahDot::Impl::parseBatchedConv2dResult(
    FieldType field, const Conv2DMeta &meta, const Conv2DProtocol &prot,
    absl::Span<const RLWEPt> polys) {
  SPU_ENFORCE_EQ(polys.size(), meta.n_output_poly);
  const auto &this_dcd_ms = dcd_mswh_.find(FieldBitLen(field))->second;

  std::array<int64_t, 4> oshape;
  oshape[0] = meta.prot_meta.input_batch;
  for (int d : {0, 1}) {
    oshape[d + 1] =
        (meta.prot_meta.input_shape[d] - meta.prot_meta.kernel_shape[d] +
         meta.prot_meta.window_strides[d]) /
        meta.prot_meta.window_strides[d];
  }
  oshape[3] = meta.prot_meta.num_kernels;

  const size_t n_poly_per_out = meta.n_output_poly / meta.prot_meta.input_batch;
  ArrayRef ret = ring_zeros(field, calcNumel(oshape));
  int64_t offset = 0;
  for (int64_t ib = 0; ib < meta.prot_meta.input_batch; ++ib) {
    // NxHxWxC layout for tensor
    // One slice is 1xHxWxC
    absl::Span<const RLWEPt> subpoly = {polys.data() + ib * n_poly_per_out,
                                        n_poly_per_out};
    auto slice = prot.ParseResult(field, meta.prot_meta, subpoly, *this_dcd_ms);
    int64_t n = slice.numel();
    DISPATCH_ALL_FIELDS(field, "", [&]() {
      ArrayView<ring2k_t> xret(ret);
      ArrayView<const ring2k_t> xslice(slice);
      pforeach(0, n, [&](int64_t i) { xret[i + offset] = xslice[i]; });
    });

    offset += n;
  }
  return ret;
}

CheetahDot::CheetahDot(std::shared_ptr<yacl::link::Context> lctx) {
  impl_ = std::make_unique<Impl>(lctx);
}

CheetahDot::~CheetahDot() = default;

ArrayRef CheetahDot::DotOLE(const ArrayRef &inp, yacl::link::Context *conn,
                            const Shape3D &dim3, bool is_lhs) {
  SPU_ENFORCE(impl_ != nullptr);
  SPU_ENFORCE(conn != nullptr);
  return impl_->DotOLE(inp, conn, dim3, is_lhs);
}

ArrayRef CheetahDot::DotOLE(const ArrayRef &inp, const Shape3D &dim3,
                            bool is_lhs) {
  SPU_ENFORCE(impl_ != nullptr);
  return impl_->DotOLE(inp, nullptr, dim3, is_lhs);
}

ArrayRef CheetahDot::Conv2dOLE(const ArrayRef &inp, yacl::link::Context *conn,
                               int64_t num_input, const Shape3D &tensor_shape,
                               int64_t num_kernels, const Shape3D &kernel_shape,
                               const Shape2D &window_strides, bool is_tensor) {
  SPU_ENFORCE(impl_ != nullptr);
  SPU_ENFORCE(conn != nullptr);
  return impl_->Conv2dOLE(inp, conn, num_input, tensor_shape, num_kernels,
                          kernel_shape, window_strides, is_tensor);
}

ArrayRef CheetahDot::Conv2dOLE(const ArrayRef &inp, int64_t num_input,
                               const Shape3D &tensor_shape, int64_t num_kernels,
                               const Shape3D &kernel_shape,
                               const Shape2D &window_strides, bool is_tensor) {
  SPU_ENFORCE(impl_ != nullptr);
  return impl_->Conv2dOLE(inp, nullptr, num_input, tensor_shape, num_kernels,
                          kernel_shape, window_strides, is_tensor);
}

}  // namespace spu::mpc::cheetah
