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
  explicit Impl(std::shared_ptr<yacl::link::Context> lctx)
      : lctx_(std::move(lctx)) {}

  ~Impl() = default;

  std::unique_ptr<Impl> Fork();

  // Compute C = A*B where |A|=dims[0]xdims[1], |B|=dims[1]xdims[2
  ArrayRef DotOLE(const ArrayRef &prv, yacl::link::Context *conn,
                  const Shape3D &dim3, bool is_lhs);

  ArrayRef Conv2dOLE(const ArrayRef &inp, yacl::link::Context *conn,
                     int64_t input_batch, const Shape3D &tensor_shape,
                     int64_t num_kernels, const Shape3D &kernel_shape,
                     const Shape2D &window_strides, bool is_tensor);

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
      modulus_bits = {49, 49};
    } else if (ring_bitlen <= 64) {
      poly_deg = 8192;
      // ~ 128 + 32 bit
      modulus_bits = {49, 56, 56};
    } else {
      poly_deg = 16384;
      // ~ 256 + 30 bit
      modulus_bits = {49, 59, 59, 59, 59};
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

  // Enc(Delta*m) -> Enc(Delta*m - r), r where r is sampled from Rq
  // One share of `m` is round(r/Delta) mod t
  // The other share of `m` is Dec(Enc(Delta*m - r))
  void H2A(absl::Span<RLWECt> ct, absl::Span<RLWEPt> rnd_mask,
           const seal::SEALContext &context, const seal::Evaluator &evaluator) {
    size_t num_poly = ct.size();
    SPU_ENFORCE(num_poly > 0);
    SPU_ENFORCE_EQ(rnd_mask.size(), num_poly);

    for (size_t idx = 0; idx < num_poly; ++idx) {
      UniformPoly(context, &rnd_mask[idx]);
      if (ct[idx].is_ntt_form()) {
        evaluator.transform_from_ntt_inplace(ct[idx]);
      }
      SubPlainInplace(ct[idx], rnd_mask[idx], context);
    }
  }

  static void RandomizeCipherForDecryption(absl::Span<RLWECt> ct_array,
                                           const seal::Encryptor &pk_encryptor,
                                           const seal::Evaluator &evaluator) {
    // TODO(juhou): combine this step into H2A
    RLWECt zero_ct;
    for (auto &i : ct_array) {
      pk_encryptor.encrypt_zero(i.parms_id(), zero_ct);
      if (zero_ct.is_ntt_form()) {
        evaluator.transform_from_ntt_inplace(zero_ct);
      }
      evaluator.add_inplace(i, zero_ct);
    }
  }

 private:
  std::shared_ptr<yacl::link::Context> lctx_;

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

std::unique_ptr<CheetahDot::Impl> CheetahDot::Impl::Fork() {
  auto f = std::make_unique<Impl>(lctx_->Spawn());
  if (seal_cntxts_.size() == 0) return f;
  std::unique_lock<std::shared_mutex> guard(context_lock_);

  f->seal_cntxts_ = seal_cntxts_;
  f->secret_keys_ = secret_keys_;
  f->pair_pub_keys_ = pair_pub_keys_;
  f->ms_helpers_ = ms_helpers_;
  f->sym_encryptors_ = sym_encryptors_;
  f->pk_encryptors_ = pk_encryptors_;
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

  SPDLOG_INFO("CheetahDot uses {} modulus {} degree for {} bit ring",
              modulus.size(), parms.poly_modulus_degree(), field_bitlen);
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
  auto &this_pk_encryptor = pk_encryptors_.find(field_bitlen)->second;
  auto &this_decryptor = decryptors_.find(field_bitlen)->second;
  auto &this_ms = ms_helpers_.find(field_bitlen)->second;
  seal::Evaluator evaluator(this_context);

  MatMatProtocol matmat(this_context, *this_ms);
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

  if (need_encrypt) {
    for (size_t i = 0; i < encoded_mat.size(); ++i) {
      NttInplace(encoded_mat[i], this_context);
      auto ct = this_encryptor->encrypt_symmetric(encoded_mat[i]).obj();
      auto ct_s = EncodeSEALObject(ct);
      conn->SendAsync(nxt_rank, ct_s, "send encrypted mat");
    }
    // wait for result
    std::vector<RLWEPt> result_poly(out_n);
    for (size_t i = 0; i < out_n; ++i) {
      auto ct_s = conn->Recv(nxt_rank, "recv result mat");
      RLWECt ct;
      DecodeSEALObject(ct_s, this_context, &ct);
      if (!ct.is_ntt_form()) {
        evaluator.transform_to_ntt_inplace(ct);
      }
      this_decryptor->decrypt(ct, result_poly[i]);
      InvNttInplace(result_poly[i], this_context);
    }

    return matmat.ParseResult(field, meta, absl::MakeSpan(result_poly));
  }

  // convert local poly to NTT form to perform multiplication.
  for (auto &poly : encoded_mat) {
    NttInplace(poly, this_context);
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

  std::vector<RLWEPt> mask_mat(out_n);
  H2A(absl::MakeSpan(result_ct), absl::MakeSpan(mask_mat), this_context,
      evaluator);
  RandomizeCipherForDecryption(absl::MakeSpan(result_ct), *this_pk_encryptor,
                               evaluator);
  matmat.ExtractLWEsInplace(meta, absl::MakeSpan(result_ct));
  for (size_t i = 0; i < out_n; ++i) {
    auto ct_s = EncodeSEALObject(result_ct[i]);
    conn->SendAsync(nxt_rank, ct_s, "send result mat");
  }

  return matmat.ParseResult(field, meta, absl::MakeSpan(mask_mat));
}

ArrayRef CheetahDot::Impl::Conv2dOLE(
    const ArrayRef &inp, yacl::link::Context *conn, int64_t input_batch,
    const Shape3D &tensor_shape, int64_t num_kernels,
    const Shape3D &kernel_shape, const Shape2D &window_strides,
    bool is_tensor) {
  if (conn == nullptr) {
    conn = lctx_.get();
  }
  int nxt_rank = conn->NextRank();
  auto eltype = inp.eltype();
  SPU_ENFORCE(eltype.isa<RingTy>(), "must be ring_type, got={}", eltype);
  SPU_ENFORCE(input_batch > 0 && num_kernels > 0);
  if (is_tensor) {
    // TODO(juhou): handle 4D for input tensor
    SPU_ENFORCE(input_batch == 1);
    SPU_ENFORCE_EQ(inp.numel(), calcNumel(tensor_shape) * input_batch);
  } else {
    SPU_ENFORCE_EQ(inp.numel(), calcNumel(kernel_shape) * num_kernels);
  }

  auto field = eltype.as<Ring2k>()->field();
  const size_t field_bitlen = FieldBitLen(field);
  LazyInit(field_bitlen);

  const auto &this_context = *seal_cntxts_.find(field_bitlen)->second;
  auto &this_encryptor = sym_encryptors_.find(field_bitlen)->second;
  auto &this_pk_encryptor = pk_encryptors_.find(field_bitlen)->second;
  auto &this_decryptor = decryptors_.find(field_bitlen)->second;
  auto &this_ms = ms_helpers_.find(field_bitlen)->second;
  seal::Evaluator evaluator(this_context);

  Conv2DProtocol conv2d(this_context, *this_ms);
  Conv2DProtocol::Meta meta;
  meta.num_kernels = num_kernels;
  meta.input_shape = tensor_shape;
  meta.kernel_shape = kernel_shape;
  meta.window_strides = window_strides;
  auto subshape = conv2d.GetSubTensorShape(meta);

  size_t num_poly_per_input = conv2d.GetInputSize(meta, subshape);
  size_t tensor_n = input_batch * num_poly_per_input;
  size_t kernel_n = conv2d.GetKernelSize(meta, subshape);
  size_t out_n = input_batch * conv2d.GetOutSize(meta, subshape);
  bool to_encrypt_tensor = tensor_n < kernel_n;
  bool need_encrypt = !(is_tensor ^ to_encrypt_tensor);

  std::vector<RLWEPt> encoded_poly(is_tensor ? tensor_n : kernel_n);
  if (is_tensor) {
    conv2d.EncodeInput(inp, meta, need_encrypt, absl::MakeSpan(encoded_poly));
  } else {
    conv2d.EncodeKernels(inp, meta, need_encrypt, absl::MakeSpan(encoded_poly));
  }

  if (need_encrypt) {
    for (size_t i = 0; i < encoded_poly.size(); ++i) {
      NttInplace(encoded_poly[i], this_context);
      auto ct = this_encryptor->encrypt_symmetric(encoded_poly[i]).obj();
      auto ct_s = EncodeSEALObject(ct);
      conn->SendAsync(nxt_rank, ct_s, "send encrypted mat");
    }
    // wait for result
    std::vector<RLWEPt> result_poly(out_n);
    for (size_t i = 0; i < out_n; ++i) {
      auto ct_s = conn->Recv(nxt_rank, "recv result mat");
      RLWECt ct;
      DecodeSEALObject(ct_s, this_context, &ct);
      if (!ct.is_ntt_form()) {
        evaluator.transform_to_ntt_inplace(ct);
      }
      this_decryptor->decrypt(ct, result_poly[i]);
      InvNttInplace(result_poly[i], this_context);
    }
    return conv2d.ParseResult(field, meta, absl::MakeSpan(result_poly));
  }

  // convert local poly to NTT form to perform multiplication.
  for (auto &poly : encoded_poly) {
    NttInplace(poly, this_context);
  }

  // recv ct from peer
  std::vector<RLWECt> encrypted_poly(is_tensor ? kernel_n : tensor_n);
  for (size_t i = 0; i < encrypted_poly.size(); ++i) {
    auto ct_s = conn->Recv(nxt_rank, "recv encrypted mat");
    DecodeSEALObject(ct_s, this_context, &encrypted_poly[i]);
  }

  std::vector<RLWECt> result_ct(out_n);
  if (is_tensor) {
    conv2d.Compute(encoded_poly, encrypted_poly, meta,
                   absl::MakeSpan(result_ct));
  } else {
    conv2d.Compute(encrypted_poly, encoded_poly, meta,
                   absl::MakeSpan(result_ct));
  }

  std::vector<RLWEPt> mask_tensor(out_n);
  H2A(absl::MakeSpan(result_ct), absl::MakeSpan(mask_tensor), this_context,
      evaluator);
  RandomizeCipherForDecryption(absl::MakeSpan(result_ct), *this_pk_encryptor,
                               evaluator);
  conv2d.ExtractLWEsInplace(meta, absl::MakeSpan(result_ct));
  for (size_t i = 0; i < out_n; ++i) {
    auto ct_s = EncodeSEALObject(result_ct[i]);
    conn->SendAsync(nxt_rank, ct_s, "send result mat");
  }
  return conv2d.ParseResult(field, meta, absl::MakeSpan(mask_tensor));
}

CheetahDot::CheetahDot(std::shared_ptr<yacl::link::Context> lctx) {
  impl_ = std::make_unique<Impl>(lctx);
}

CheetahDot::~CheetahDot() = default;

CheetahDot::CheetahDot(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

std::unique_ptr<CheetahDot> CheetahDot::Fork() {
  auto ptr = new CheetahDot(impl_->Fork());
  return std::unique_ptr<CheetahDot>(ptr);
}

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
                               const Shape3D &tensor_shape, int64_t num_kernels,
                               const Shape3D &kernel_shape,
                               const Shape2D &window_strides, bool is_tensor) {
  SPU_ENFORCE(impl_ != nullptr);
  SPU_ENFORCE(conn != nullptr);
  int64_t input_batch = 1;
  return impl_->Conv2dOLE(inp, conn, input_batch, tensor_shape, num_kernels,
                          kernel_shape, window_strides, is_tensor);
}

ArrayRef CheetahDot::Conv2dOLE(const ArrayRef &inp, const Shape3D &tensor_shape,
                               int64_t num_kernels, const Shape3D &kernel_shape,
                               const Shape2D &window_strides, bool is_tensor) {
  SPU_ENFORCE(impl_ != nullptr);
  int64_t input_batch = 1;
  return impl_->Conv2dOLE(inp, nullptr, input_batch, tensor_shape, num_kernels,
                          kernel_shape, window_strides, is_tensor);
}

}  // namespace spu::mpc::cheetah
