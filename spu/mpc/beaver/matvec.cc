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

#include "spu/mpc/beaver/matvec.h"

#include "seal/batchencoder.h"
#include "seal/context.h"
#include "seal/decryptor.h"
#include "seal/encryptor.h"
#include "seal/evaluator.h"
#include "seal/galoiskeys.h"
#include "seal/keygenerator.h"
#include "seal/publickey.h"
#include "seal/secretkey.h"
#include "yasl/base/exception.h"
#include "yasl/base/int128.h"
#include "yasl/utils/parallel.h"

#include "spu/core/type_util.h"
#include "spu/core/xt_helper.h"
#include "spu/mpc/beaver/matvec_helper.h"
#include "spu/mpc/beaver/modswitch_helper.h"
#include "spu/mpc/util/ring_ops.h"
#include "spu/mpc/util/seal_help.h"

namespace spu::mpc {

template <typename T>
static T CeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

struct MatVecProtocol::Impl {
 public:
  explicit Impl(const seal::GaloisKeys &rot_keys,
                const seal::SEALContext &context)
      : rot_keys_(rot_keys),
        context_(context),
        evaluator_(context),
        encoder_(context) {
    YASL_ENFORCE(context_.parameters_set());
    // YASL_ENFORCE(seal::is_metadata_valid_for(rot_keys_, context_));

    num_slots_ = context_.key_context_data()->parms().poly_modulus_degree();
  }

  size_t num_slots() const { return num_slots_; }

  yasl::Buffer EncryptVector(ArrayRef vec,
                             const MatVecHelper::MatViewMeta &meta,
                             const seal::Encryptor &sym_encryptor) const {
    YASL_ENFORCE(vec.isCompact());

    size_t nrows = meta.is_transposed ? meta.col_extent : meta.row_extent;
    size_t ncols = meta.is_transposed ? meta.row_extent : meta.col_extent;
    YASL_ENFORCE(nrows > 0 && ncols > 0 && ncols == (size_t)vec.numel());
    YASL_ENFORCE(std::min(nrows, ncols) <= num_slots());

    return EncryptVector(vec, sym_encryptor);
  }

  void EncodeSubMatrix(ArrayRef mat, const MatVecHelper::MatViewMeta &meta,
                       const ModulusSwitchHelper &ms_helper,
                       size_t target_prime_index,
                       absl::Span<seal::Plaintext> out) const {
    MatVecHelper helper(num_slots(), mat, meta);
    YASL_ENFORCE(out.size() == helper.NumDiagnoals());
    size_t num_diags = helper.NumDiagnoals();
    size_t giant_step = std::ceil(std::sqrt(1. * num_diags));
    size_t baby_step = CeilDiv(num_diags, giant_step);

    for (size_t k = 0; k < baby_step && k * giant_step < num_diags; ++k) {
      size_t rhs_rot = k * giant_step;
      yasl::parallel_for(0, giant_step, 1, [&](size_t bgn, size_t end) {
        std::vector<uint64_t> slots(num_slots());
        for (size_t j = bgn; j < end; ++j) {
          size_t diag_idx = rhs_rot + j;
          if (diag_idx >= num_diags) break;
          auto diag = helper.GetRotatedDiagnoal(diag_idx, rhs_rot);
          std::vector<uint64_t> padded_diag = ZeroPadAndTileVector(diag);

          absl::Span<const uint64_t> wrap_diag(padded_diag.data(),
                                               padded_diag.size());
          absl::Span<uint64_t> wrap_dst(slots.data(), slots.size());
          ms_helper.CenteralizeAt(wrap_diag, target_prime_index, wrap_dst);
          CATCH_SEAL_ERROR(encoder_.encode(slots, out[diag_idx]));
        }
      });
    }
  }

  void EncodeSubMatrix(ArrayRef mat, const MatVecHelper::MatViewMeta &meta,
                       absl::Span<seal::Plaintext> out) const {
    MatVecHelper helper(num_slots(), mat, meta);
    YASL_ENFORCE(out.size() == helper.NumDiagnoals());
    size_t num_diags = helper.NumDiagnoals();
    size_t giant_step = std::ceil(std::sqrt(1. * num_diags));
    size_t baby_step = CeilDiv(num_diags, giant_step);

    for (size_t k = 0; k < baby_step && k * giant_step < num_diags; ++k) {
      size_t rhs_rot = k * giant_step;
      yasl::parallel_for(0, giant_step, 1, [&](size_t bgn, size_t end) {
        for (size_t j = bgn; j < end; ++j) {
          size_t diag_idx = rhs_rot + j;
          if (diag_idx >= num_diags) break;
          auto diag = helper.GetRotatedDiagnoal(diag_idx, rhs_rot);
          EncodeVector(diag, &(out[diag_idx]));
        }
      });
    }
  }

  void EncodeSubMatrix(ArrayRef mat, const MatVecHelper::MatViewMeta &meta,
                       std::vector<seal::Plaintext> *out) const {
    yasl::CheckNotNull(out);
    MatVecHelper helper(num_slots(), mat, meta);
    out->resize(helper.NumDiagnoals());
    absl::Span<seal::Plaintext> out_wrap(out->data(), out->size());
    EncodeSubMatrix(mat, meta, out_wrap);
  }

  void Compute(const seal::Ciphertext &enc_vec,
               absl::Span<const seal::Plaintext> ecd_mat,
               const MatVecHelper::MatViewMeta &meta,
               seal::Ciphertext *out) const {
    yasl::CheckNotNull(out);
    size_t nrows = meta.is_transposed ? meta.col_extent : meta.row_extent;
    size_t ncols = meta.is_transposed ? meta.row_extent : meta.col_extent;
    YASL_ENFORCE(nrows > 0 && ncols > 0);
    YASL_ENFORCE(std::min(nrows, ncols) <= num_slots());

    size_t num_diags = Next2Pow(std::min(nrows, ncols));
    size_t giant_step = std::ceil(std::sqrt(1. * num_diags));
    size_t baby_step = CeilDiv(num_diags, giant_step);
    YASL_ENFORCE(ecd_mat.size() == num_diags);

    // The following inner product computation can be faster in the NTT form.
    std::vector<seal::Ciphertext> rotated_vec;
    if (enc_vec.is_ntt_form()) {
      rotated_vec = std::vector<seal::Ciphertext>(giant_step, enc_vec);
    } else {
      auto cpy{enc_vec};
      CATCH_SEAL_ERROR(evaluator_.transform_to_ntt_inplace(cpy));
      rotated_vec = std::vector<seal::Ciphertext>(giant_step, cpy);
    }

    // Baby-step rotations
    yasl::parallel_for(1, giant_step, 1, [&](size_t bgn, size_t end) {
      for (size_t j = bgn; j < end; ++j) {
        CATCH_SEAL_ERROR(
            evaluator_.rotate_rows_inplace(rotated_vec.at(j), j, rot_keys_));
      }
    });

    std::mutex result_guard;

    yasl::parallel_for(0, baby_step, 1, [&](size_t kbgn, size_t kend) {
      for (size_t k = kbgn; k < kend && k * giant_step < num_diags; ++k) {
        const size_t rhs_rot = k * giant_step;
        seal::Ciphertext inner_accum;
        for (size_t j = 0; j < giant_step; ++j) {
          const auto &rot_vec = rotated_vec[j];
          size_t diag_idx = rhs_rot + j;
          if (diag_idx >= num_diags) break;
          // some diagnoals might be zero.
          if (ecd_mat[diag_idx].is_zero()) continue;

          if (inner_accum.size() > 0) {
            seal::Ciphertext mul;
            CATCH_SEAL_ERROR(
                evaluator_.multiply_plain(rot_vec, ecd_mat[diag_idx], mul));
            CATCH_SEAL_ERROR(evaluator_.add_inplace(inner_accum, mul));
          } else {
            CATCH_SEAL_ERROR(evaluator_.multiply_plain(
                rot_vec, ecd_mat[diag_idx], inner_accum));
          }
        }  // inner-loop

        // TODO(juhou) in what cases that all the inner diagnoals are zero.
        YASL_ENFORCE(inner_accum.size() > 0);

        // Giant-step rotations
        CATCH_SEAL_ERROR(
            evaluator_.rotate_rows_inplace(inner_accum, rhs_rot, rot_keys_));
        // race-write
        std::lock_guard<std::mutex> lock(result_guard);
        if (out->size() > 0) {
          CATCH_SEAL_ERROR(evaluator_.add_inplace(*out, inner_accum));
        } else {
          *out = inner_accum;
        }
      }  // outter-loop
    });

    // Handle partial sum
    if (nrows < ncols) {
      size_t cext = Next2Pow(ncols);
      size_t rext = Next2Pow(nrows);
      size_t half_slots = num_slots() >> 1;
      for (size_t rot = rext; rot < cext; rot <<= 1) {
        auto cpy{*out};
        if (rot == half_slots) {
          CATCH_SEAL_ERROR(evaluator_.rotate_columns_inplace(cpy, rot_keys_));
        } else {
          CATCH_SEAL_ERROR(evaluator_.rotate_rows_inplace(cpy, rot, rot_keys_));
        }
        CATCH_SEAL_ERROR(evaluator_.add_inplace(*out, cpy));
      }
    }

    // ensure out.is_ntt_form() == enc_vec.is_ntt_form()
    if (out->is_ntt_form() != enc_vec.is_ntt_form()) {
      if (out->is_ntt_form()) {
        CATCH_SEAL_ERROR(evaluator_.transform_from_ntt_inplace(*out));
      } else {
        CATCH_SEAL_ERROR(evaluator_.transform_to_ntt_inplace(*out));
      }
    }
  }

 protected:
  std::vector<uint64_t> ZeroPadAndTileVector(ArrayRef vec) const {
    size_t max_pack = num_slots();
    size_t vec_dim = static_cast<size_t>(vec.numel());
    YASL_ENFORCE(vec_dim > 0 && vec_dim <= max_pack);
    auto field = vec.eltype().as<Ring2k>()->field();

    std::vector<uint64_t> out(max_pack);

    DISPATCH_FM3264(field, "EncVec", [&]() {
      using ring2k_u = typename std::make_unsigned<ring2k_t>::type;
      // NOTE(juhou): we need to cast to unsigned for FM32 to make the high-end
      // to all zeros when casting to uint64_t
      auto xvec = xt_adapt<ring2k_u>(vec);
      // zero-pad to 2-align
      size_t padded_sze = Next2Pow(xvec.size());
      std::transform(
          xvec.begin(), xvec.end(), out.data(),
          [](ring2k_u x) -> uint64_t { return static_cast<uint64_t>(x); });

      std::fill_n(out.data() + xvec.size(), padded_sze - xvec.size(), 0);

      if (padded_sze != max_pack) {
        // repeat the vector to full fill all the slots
        size_t nrep = max_pack / padded_sze;
        for (size_t r = 1; r < nrep; ++r) {
          std::copy_n(out.data(), padded_sze, out.data() + r * padded_sze);
        }
      }
      return;
    });
    return out;
  }

  void EncodeVector(ArrayRef vec, seal::Plaintext *out) const {
    yasl::CheckNotNull(out);
    auto padded = ZeroPadAndTileVector(vec);
    CATCH_SEAL_ERROR(encoder_.encode(padded, *out));
  }

  yasl::Buffer EncryptVector(ArrayRef vec,
                             const seal::Encryptor &sym_encryptor) const {
    seal::Plaintext pt;
    EncodeVector(vec, &pt);
    auto ct = sym_encryptor.encrypt_symmetric(pt);
    return EncodeSEALObject(ct.obj());
  }

 private:
  const seal::GaloisKeys &rot_keys_;
  const seal::SEALContext &context_;

  size_t num_slots_{0};
  seal::Evaluator evaluator_;
  seal::BatchEncoder encoder_;
};

MatVecProtocol::MatVecProtocol(const seal::GaloisKeys &rot_keys,
                               const seal::SEALContext &context)
    : impl_(std::make_shared<Impl>(rot_keys, context)) {}

size_t MatVecProtocol::num_slots() const {
  yasl::CheckNotNull(impl_.get());
  return impl_->num_slots();
}

// Encrypt a compact vector and serialize it to buffer.
yasl::Buffer MatVecProtocol::EncryptVector(
    ArrayRef vec, const MatVecHelper::MatViewMeta &meta,
    const seal::Encryptor &sym_encryptor) const {
  yasl::CheckNotNull(impl_.get());
  return impl_->EncryptVector(vec, meta, sym_encryptor);
}

// Encode a compact submatrix. The submatrix is defined by `meta`.
void MatVecProtocol::EncodeSubMatrix(ArrayRef mat,
                                     const MatVecHelper::MatViewMeta &meta,
                                     std::vector<seal::Plaintext> *out) const {
  yasl::CheckNotNull(impl_.get());
  return impl_->EncodeSubMatrix(mat, meta, out);
}

void MatVecProtocol::EncodeSubMatrix(ArrayRef mat,
                                     const MatVecHelper::MatViewMeta &meta,
                                     absl::Span<seal::Plaintext> out) const {
  yasl::CheckNotNull(impl_.get());
  return impl_->EncodeSubMatrix(mat, meta, out);
}

void MatVecProtocol::EncodeSubMatrix(ArrayRef mat,
                                     const MatVecHelper::MatViewMeta &meta,
                                     const ModulusSwitchHelper &ms_helper,
                                     size_t target_prime_index,
                                     absl::Span<seal::Plaintext> out) const {
  yasl::CheckNotNull(impl_.get());
  return impl_->EncodeSubMatrix(mat, meta, ms_helper, target_prime_index, out);
}

void MatVecProtocol::Compute(const seal::Ciphertext &enc_vec,
                             absl::Span<const seal::Plaintext> ecd_mat,
                             const MatVecHelper::MatViewMeta &meta,
                             seal::Ciphertext *out) const {
  yasl::CheckNotNull(impl_.get());
  return impl_->Compute(enc_vec, ecd_mat, meta, out);
}

}  // namespace spu::mpc
