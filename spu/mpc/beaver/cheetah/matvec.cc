// Copyright 2022 Ant Group Co., Ltd.
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

#include "spu/mpc/beaver/cheetah/matvec.h"

#include <seal/evaluator.h>
#include <seal/util/numth.h>

#include <array>

#include "absl/numeric/bits.h"
#include "xtensor/xview.hpp"
#include "yacl/base/exception.h"
#include "yacl/utils/parallel.h"

#include "spu/core/xt_helper.h"
#include "spu/mpc/beaver/cheetah/util.h"
#include "spu/mpc/util/ring_ops.h"
#include "spu/mpc/util/seal_help.h"  // NttInplace

namespace spu::mpc {

template <typename T>
inline static T CeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

static std::array<size_t, 2> CostModel(const MatVecProtocol::Meta& meta,
                                       size_t N, double cpu_weight,
                                       double band_weight) {
  // some heuristics
  const double enc_cost = 1.0;
  const double mul_cost = 4.0;
  const double dec_cost = 0.5;
  const double commu_cost = 1.0;

  double min_cost = std::numeric_limits<double>::max();
  std::array<size_t, 2> ret{0};
  for (size_t d0 = 1; d0 <= std::min(N, meta.nrows); ++d0) {
    for (size_t d1 = 1; d1 <= std::min(N, meta.ncols); ++d1) {
      if (d0 * d1 > N) continue;

      size_t ct_in = CeilDiv(meta.ncols, d1);
      size_t ct_out = CeilDiv(meta.nrows, d0);

      size_t mul = ct_out * ct_in;
      double cpu_cost = ct_in * enc_cost + mul * mul_cost + ct_out * dec_cost;
      double band_cost = (ct_in + ct_out) * commu_cost;

      double cost =
          (1. - cpu_weight) * cpu_cost + (1 - band_weight) * band_cost;

      if (cost < min_cost) {
        min_cost = cost;
        ret[0] = d0;
        ret[1] = d1;
      }
    }
  }
  return ret;
}

static std::array<size_t, 2> GetSubMatrixShape(const MatVecProtocol::Meta& meta,
                                               size_t poly_degree) {
  // we balance the cpu/bandwidth
  return CostModel(meta, poly_degree, 0.5, 0.5);
}

/// Concatenate the specified submatrix into one vector in row-major
/// Zero-padding the bottom-rows and right-most-columns of submatrix if
/// `extents` are smaller than the given `submat_shape`
static ArrayRef ConcatSubMatrix(const ArrayRef& mat,
                                const MatVecProtocol::Meta& meta,
                                const std::array<size_t, 2>& starts,
                                const std::array<size_t, 2>& extents,
                                const std::array<size_t, 2>& submat_shape) {
  const Type& eltype = mat.eltype();
  const auto field = eltype.as<Ring2k>()->field();
  // NOTE: zero padding via initialization
  ArrayRef flatten = ring_zeros(field, submat_shape[0] * submat_shape[1]);

  DISPATCH_ALL_FIELDS(field, "ConcatSubMatrix", [&]() {
    auto dst_mat = xt_mutable_adapt<ring2k_t>(flatten);
    auto mat_view = xt_adapt<ring2k_t>(mat);
    dst_mat = dst_mat.reshape(submat_shape);
    mat_view = mat_view.reshape({meta.nrows, meta.ncols});

    auto row_range = xt::range(starts[0], starts[0] + extents[0]);
    auto col_range = xt::range(starts[1], starts[1] + extents[1]);

    xt::view(dst_mat, xt::range(0, extents[0]), xt::range(0, extents[1])) =
        xt::view(mat_view, row_range, col_range);
  });

  return flatten;
}

MatVecProtocol::MatVecProtocol(const seal::SEALContext& context,
                               const ModulusSwitchHelper& ms_helper)
    : poly_deg_(context.key_context_data()->parms().poly_modulus_degree()),
      encoder_(context, ms_helper),
      context_(context) {
  YACL_ENFORCE(context_.parameters_set());
}

bool MatVecProtocol::IsValidMeta(const Meta& meta) const {
  return meta.nrows > 0 && meta.ncols > 0;
}

size_t MatVecProtocol::GetEncVectorSize(const Meta& meta) const {
  YACL_ENFORCE(IsValidMeta(meta));
  auto submat_shape = GetSubMatrixShape(meta, poly_degree());
  return CeilDiv(meta.ncols, submat_shape[1]);
}

size_t MatVecProtocol::GetMatVecSize(const Meta& meta) const {
  YACL_ENFORCE(IsValidMeta(meta));
  auto submat_shape = GetSubMatrixShape(meta, poly_degree());
  return CeilDiv(meta.nrows, submat_shape[0]);
}

void MatVecProtocol::MatVecNoExtract(const Meta& meta,
                                     const std::vector<RLWEPt>& mat,
                                     const std::vector<RLWECt>& vec,
                                     std::vector<RLWECt>* out) const {
  YACL_ENFORCE(IsValidMeta(meta));
  yacl::CheckNotNull(out);
  for (const auto& mat_pt : mat) {
    YACL_ENFORCE(seal::is_metadata_valid_for(mat_pt, context_));
  }
  for (const auto& vec_pt : vec) {
    YACL_ENFORCE(seal::is_metadata_valid_for(vec_pt, context_));
  }

  auto submat_shape = GetSubMatrixShape(meta, poly_degree());
  size_t num_row_blks = CeilDiv(meta.nrows, submat_shape[0]);
  size_t num_col_blks = CeilDiv(meta.ncols, submat_shape[1]);
  YACL_ENFORCE_EQ(seal::util::mul_safe(num_row_blks, num_col_blks), mat.size());
  YACL_ENFORCE_EQ(num_col_blks, vec.size());

  out->resize(num_row_blks);
  seal::Evaluator evaluator(context_);
  for (size_t rb = 0; rb < num_row_blks; ++rb) {
    RLWECt accumulated;
    for (size_t cb = 0; cb < num_col_blks; ++cb) {
      const auto& mat_pt = mat[rb * num_col_blks + cb];
      size_t n = mat_pt.coeff_count();
      if (0 == n) continue;
      if (std::all_of(mat_pt.data(), mat_pt.data() + n,
                      [](uint64_t x) { return x == 0; })) {
        continue;
      }

      RLWECt tmp;
      evaluator.multiply_plain(vec.at(cb), mat_pt, tmp);
      if (accumulated.size() > 0) {
        evaluator.add_inplace(accumulated, tmp);
      } else {
        accumulated = tmp;
      }
    }
    YACL_ENFORCE(accumulated.size() > 0,
                 fmt::format("all zero matrix is not supported for MatVec"));

    // position form for RLWE2LWE
    if (accumulated.is_ntt_form()) {
      evaluator.transform_from_ntt(accumulated, out->at(rb));
    } else {
      out->at(rb) = accumulated;
    }
  }

  ExtractLWEsInplace(meta, *out);
}

void MatVecProtocol::ExtractLWEsInplace(const Meta& meta,
                                        std::vector<RLWECt>& rlwes) const {
  auto submat_shape = GetSubMatrixShape(meta, poly_degree());
  size_t num_row_blks = CeilDiv(meta.nrows, submat_shape[0]);
  YACL_ENFORCE_EQ(rlwes.size(), num_row_blks);
  for (const auto& rlwe : rlwes) {
    YACL_ENFORCE(seal::is_metadata_valid_for(rlwe, context_));
    YACL_ENFORCE(!rlwe.is_ntt_form() && rlwe.size() == 2);
  }

  std::set<size_t> to_keep;
  for (size_t r = 0; r < submat_shape[0]; ++r) {
    size_t target_coeff = r * submat_shape[1];
    to_keep.insert(target_coeff);
  }
  for (size_t rb = 0; rb + 1 < num_row_blks; ++rb) {
    KeepCoefficientsInplace(rlwes[rb], to_keep);
  }

  // take care the last row-block which might contains less rows
  to_keep.clear();
  size_t last_rb = num_row_blks - 1;
  for (size_t r = 0; r < submat_shape[0]; ++r) {
    size_t row = last_rb * submat_shape[0] + r;
    if (row >= meta.nrows) break;
    size_t target_coeff = r * submat_shape[1];
    to_keep.insert(target_coeff);
  }
  KeepCoefficientsInplace(rlwes[last_rb], to_keep);
}

void MatVecProtocol::ExtractLWEs(const Meta& meta,
                                 const std::vector<RLWECt>& rlwes,
                                 std::vector<LWECt>* out) const {
  auto submat_shape = GetSubMatrixShape(meta, poly_degree());
  size_t num_row_blks = CeilDiv(meta.nrows, submat_shape[0]);
  YACL_ENFORCE_EQ(rlwes.size(), num_row_blks);
  for (const auto& rlwe : rlwes) {
    YACL_ENFORCE(seal::is_metadata_valid_for(rlwe, context_));
    YACL_ENFORCE(!rlwe.is_ntt_form() && rlwe.size() == 2);
  }

  out->resize(meta.nrows);

  for (size_t rb = 0; rb < num_row_blks; ++rb) {
    for (size_t r = 0; r < submat_shape[0]; ++r) {
      size_t row = rb * submat_shape[0] + r;
      if (row >= meta.nrows) break;
      size_t target_coeff = r * submat_shape[1];
      out->at(row) = LWECt(rlwes[rb], target_coeff, context_);
    }
  }
}

void MatVecProtocol::MatVec(const Meta& meta, const std::vector<RLWEPt>& mat,
                            const std::vector<RLWECt>& vec,
                            std::vector<LWECt>* out) const {
  std::vector<RLWECt> rlwes;
  MatVecNoExtract(meta, mat, vec, &rlwes);
  ExtractLWEs(meta, rlwes, out);
}

void MatVecProtocol::EncodeVector(const Meta& meta, const ArrayRef& vec,
                                  std::vector<RLWEPt>* out) const {
  YACL_ENFORCE(IsValidMeta(meta));
  yacl::CheckNotNull(out);

  const Type& eltype = vec.eltype();
  YACL_ENFORCE(eltype.isa<RingTy>(), "must be ring_type, got={}", eltype);
  YACL_ENFORCE_EQ(static_cast<size_t>(vec.numel()), meta.ncols);

  auto submat_shape = GetSubMatrixShape(meta, poly_degree());
  size_t num_subvec = CeilDiv(meta.ncols, submat_shape[1]);
  out->resize(num_subvec);

  for (size_t idx = 0; idx < num_subvec; ++idx) {
    size_t bgn = idx * submat_shape[1];
    size_t end = std::min(bgn + submat_shape[1], meta.ncols);
    auto subvec = vec.slice(bgn, end);
    encoder_.Backward(subvec, out->data() + idx, /*scale_delta*/ true);
  }
}

void MatVecProtocol::EncodeMatrix(const Meta& meta, const ArrayRef& mat,
                                  std::vector<RLWEPt>* out) const {
  YACL_ENFORCE(IsValidMeta(meta));
  yacl::CheckNotNull(out);
  YACL_ENFORCE_EQ(seal::util::mul_safe(meta.nrows, meta.ncols),
                  static_cast<size_t>(mat.numel()));

  const Type& eltype = mat.eltype();
  YACL_ENFORCE(eltype.isa<RingTy>(), "must be ring_type, got={}", eltype);

  auto submat_shape = GetSubMatrixShape(meta, poly_degree());
  size_t num_row_blks = CeilDiv(meta.nrows, submat_shape[0]);
  size_t num_col_blks = CeilDiv(meta.ncols, submat_shape[1]);
  out->resize(seal::util::mul_safe(num_row_blks, num_col_blks));

  constexpr size_t kParallelGrain = 1;
  yacl::parallel_for(
      0, num_row_blks, kParallelGrain, [&](size_t rb_bgn, size_t rb_end) {
        auto out_ptr = out->data() + rb_bgn * num_col_blks;
        for (size_t rblk = rb_bgn; rblk < rb_end; ++rblk) {
          std::array<size_t, 2> starts, extents;
          size_t row_bgn = rblk * submat_shape[0];
          size_t row_end = std::min(meta.nrows, row_bgn + submat_shape[0]);
          starts[0] = row_bgn;
          extents[0] = row_end - row_bgn;

          for (size_t cblk = 0; cblk < num_col_blks; ++cblk) {
            size_t col_bgn = cblk * submat_shape[1];
            size_t col_end = std::min(meta.ncols, col_bgn + submat_shape[1]);
            starts[1] = col_bgn;
            extents[1] = col_end - col_bgn;

            auto submat =
                ConcatSubMatrix(mat, meta, starts, extents, submat_shape);
            encoder_.Forward(submat, out_ptr, /*scale*/ false);
            NttInplace(*out_ptr++, context_);
          }
        }
      });
}

ArrayRef MatVecProtocol::ParseMatVecResult(
    FieldType field, const Meta& meta, const std::vector<RLWEPt>& rlwes) const {
  YACL_ENFORCE(IsValidMeta(meta));

  auto submat_shape = GetSubMatrixShape(meta, poly_degree());
  size_t num_row_blks = CeilDiv(meta.nrows, submat_shape[0]);
  YACL_ENFORCE_EQ(num_row_blks, rlwes.size());
  size_t coeff_count = rlwes[0].coeff_count();
  for (const auto& rlwe : rlwes) {
    YACL_ENFORCE(seal::is_metadata_valid_for(rlwe, context_));
    YACL_ENFORCE_EQ(rlwe.coeff_count(), coeff_count);
  }

  const size_t poly_deg = poly_degree();
  const size_t num_modulus = rlwes[0].coeff_count() / poly_degree();
  std::vector<uint64_t> coeff_rns(submat_shape[0] * num_modulus);

  auto out = ring_zeros(field, meta.nrows);
  for (size_t rb = 0; rb < num_row_blks; ++rb) {
    size_t row_bgn = rb * submat_shape[0];
    size_t row_end = std::min(meta.nrows, row_bgn + submat_shape[0]);
    size_t num_coeff = row_end - row_bgn;
    // Take the needed coefficients (RNS form) then compute the ModulusDown.
    for (size_t r = 0; r < num_coeff; ++r) {
      size_t row = rb * submat_shape[0] + r;
      YACL_ENFORCE(row < meta.nrows);
      size_t target_coeff = r * submat_shape[1];
      auto dst_ptr = coeff_rns.data() + r;
      auto src_ptr = rlwes[rb].data() + target_coeff;

      for (size_t l = 0; l < num_modulus; ++l) {
        *dst_ptr = *src_ptr;
        dst_ptr += num_coeff;
        src_ptr += poly_deg;
      }
    }

    absl::Span<const uint64_t> inp(coeff_rns.data(), num_coeff * num_modulus);
    encoder_.ms_helper().ModulusDownRNS(inp, out.slice(row_bgn, row_end));
  }
  return out;
}

}  // namespace spu::mpc
