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
//

#include "libspu/mpc/cheetah/arith/matmat_prot.h"

#include <functional>
#include <unordered_map>

#include "seal/seal.h"
#include "spdlog/spdlog.h"
#include "yacl/utils/parallel.h"

#include "libspu/core/shape_util.h"  //calcNumel
#include "libspu/mpc/cheetah/arith/vector_encoder.h"
#include "libspu/mpc/cheetah/rlwe/utils.h"
#include "libspu/mpc/utils/ring_ops.h"

template <>
struct std::hash<spu::mpc::cheetah::MatMatProtocol::Meta> {
  size_t operator()(
      const spu::mpc::cheetah::MatMatProtocol::Meta& s) const noexcept {
    using namespace spu::mpc::cheetah;
    // FIXME(juhou): use a better way for hash
    size_t h = std::hash<std::string>()("MatMatProtocol::Meta");
    for (int i : {0, 1, 2}) {
      h = (h << 1) ^ std::hash<int64_t>()(s.dims[i]);
    }
    return h;
  }
};

namespace spu::mpc::cheetah {
bool operator==(const MatMatProtocol::Meta& x, const MatMatProtocol::Meta& y) {
  return x.dims == y.dims;
}

class LHSIndexer {
 public:
  explicit LHSIndexer(const Shape3D& meta) {
    offsets_[0] = meta[1] * meta[2];
    offsets_[1] = meta[1] - 1;
  }

  size_t operator()(size_t r, size_t c) const {
    return r * offsets_[0] + offsets_[1] - c;
  }

  size_t offsets_[2];
};

class RHSIndexer {
 public:
  explicit RHSIndexer(const Shape3D& meta) { offset_ = meta[1]; }
  size_t operator()(size_t r, size_t c) const { return c * offset_ + r; }
  size_t offset_{0};
};

class ResultIndexer {
 public:
  explicit ResultIndexer(const Shape3D& meta) {
    offsets_[0] = meta[1] * meta[2];
    offsets_[1] = meta[1];
    offsets_[2] = meta[1] - 1;
  }

  size_t operator()(size_t r, size_t c) const {
    return r * offsets_[0] + c * offsets_[1] + offsets_[2];
  }

  size_t offsets_[3];
};

template <typename Indexer>
ArrayRef ConcatSubMatrix(const ArrayRef& mat, const Shape2D& mat_shape,
                         const Shape2D& starts, const Shape2D& extents,
                         const Shape2D& submat_shape, int64_t num_el,
                         const Indexer& indexer) {
  const Type& eltype = mat.eltype();
  SPU_ENFORCE(eltype.isa<RingTy>(), "must be ring_type, got={}", eltype);
  SPU_ENFORCE_EQ(mat.numel(), mat_shape[0] * mat_shape[1]);
  SPU_ENFORCE(num_el >= submat_shape[0] * submat_shape[1]);
  for (size_t d : {0, 1}) {
    SPU_ENFORCE(starts[d] < mat_shape[d]);
    SPU_ENFORCE(extents[d] > 0);
    SPU_ENFORCE(starts[d] + extents[d] <= mat_shape[d]);
  }

  const auto field = eltype.as<Ring2k>()->field();
  // NOTE: zero padding via initialization
  ArrayRef flatten = ring_zeros(field, num_el);

  DISPATCH_ALL_FIELDS(field, "", [&]() {
    for (int64_t r = 0, rr = starts[0]; r < extents[0]; ++r, ++rr) {
      for (int64_t c = 0, cc = starts[1]; c < extents[1]; ++c, ++cc) {
        flatten.at<ring2k_t>(indexer(r, c)) =
            mat.at<ring2k_t>(rr * mat_shape[1] + cc);
      }
    }
  });

  return flatten;
}

MatMatProtocol::MatMatProtocol(const seal::SEALContext& context,
                               const ModulusSwitchHelper& ms_helper,
                               bool use_montgomery_fma)
    : use_montgomery_fma_(use_montgomery_fma),
      context_(context),
      ms_helper_(ms_helper) {
  SPU_ENFORCE(context_.parameters_set());
  SPU_ENFORCE(context_.first_parms_id() == ms_helper_.parms_id());

  poly_deg_ = context_.first_context_data()->parms().poly_modulus_degree();
  vencoder_ = std::make_unique<VectorEncoder>(context_, ms_helper_);

  if (use_montgomery_fma_) {
    const auto& cntxt_data = context.first_context_data();
    const auto& modulus = cntxt_data->parms().coeff_modulus();

    for (const seal::Modulus& moduli : modulus) {
      uint64_t prime_inv = [](uint64_t prime) {
        uint64_t inv = 1;
        for (int i = 0; i < 63; ++i) {
          inv *= prime;
          prime *= prime;
        }
        return inv;  // prime^{-1} mod 2^64
      }(moduli.value());

      montgomery_precond_.push_back(prime_inv);
    }
  }
}

bool MatMatProtocol::IsValidMeta(const Meta& meta) const {
  return calcNumel(absl::MakeSpan(meta.dims)) > 0;
}

bool MatMatProtocol::IsValidSubShape(const Shape3D& shape) const {
  int64_t n = calcNumel(absl::MakeSpan(shape));
  return (n > 0 && n <= poly_deg_);
}

size_t MatMatProtocol::GetLeftSize(const Meta& meta,
                                   const Shape3D& subshape) const {
  SPU_ENFORCE(IsValidMeta(meta));
  SPU_ENFORCE(IsValidSubShape(subshape));
  return CeilDiv(meta.dims[0], subshape[0]) *
         CeilDiv(meta.dims[1], subshape[1]);
}

size_t MatMatProtocol::GetRightSize(const Meta& meta,
                                    const Shape3D& subshape) const {
  SPU_ENFORCE(IsValidMeta(meta));
  SPU_ENFORCE(IsValidSubShape(subshape));
  return CeilDiv(meta.dims[1], subshape[1]) *
         CeilDiv(meta.dims[2], subshape[2]);
}

size_t MatMatProtocol::GetOutSize(const Meta& meta,
                                  const Shape3D& subshape) const {
  SPU_ENFORCE(IsValidMeta(meta));
  SPU_ENFORCE(IsValidSubShape(subshape));
  return CeilDiv(meta.dims[0], subshape[0]) *
         CeilDiv(meta.dims[2], subshape[2]);
}

Shape3D MatMatProtocol::GetSubMatShape(const Meta& meta) const {
  static std::unordered_map<Meta, Shape3D> memo_;
  static std::shared_mutex lock_;
  {
    std::shared_lock<std::shared_mutex> guard(lock_);
    auto val = memo_.find(meta);
    if (val != memo_.end()) return val->second;
  }
  std::unique_lock<std::shared_mutex> guard(lock_);
  auto val = memo_.find(meta);
  if (val != memo_.end()) return val->second;

  constexpr int64_t enc_price = 3;
  constexpr int64_t eval_price = 1;
  constexpr int64_t dec_price = 2;

  Shape3D blk;
  Shape3D subshape;
  int64_t min_cost = std::numeric_limits<int64_t>::max();

  for (int64_t d0 = 1; d0 <= meta.dims[0]; ++d0) {
    if (d0 > poly_deg_) break;
    blk[0] = CeilDiv(meta.dims[0], d0);

    for (int64_t d1 = 1; d1 <= meta.dims[1]; ++d1) {
      if (d0 * d1 > poly_deg_) break;
      blk[1] = CeilDiv(meta.dims[1], d1);

      for (int64_t d2 = 1; d2 <= meta.dims[2]; ++d2) {
        if (d0 * d1 * d2 > poly_deg_) break;
        blk[2] = CeilDiv(meta.dims[2], d2);

        int pivot = blk[0] < blk[2] ? 0 : 2;
        int64_t enc_cost = blk[1] * blk[pivot] * enc_price;
        int64_t eval_cost = blk[0] * blk[1] * blk[2] * eval_price;
        int64_t dec_cost = blk[0] * blk[2] * dec_price;

        int64_t cost = enc_cost + eval_cost + dec_cost;

        if (cost <= min_cost) {
          min_cost = cost;
          subshape[0] = d0;
          subshape[1] = d1;
          subshape[2] = d2;
        }
      }
    }
  }

  memo_.insert({meta, subshape});
  return subshape;
}

void MatMatProtocol::EncodeLHS(const ArrayRef& mat, const Meta& meta,
                               bool need_encrypt,
                               absl::Span<RLWEPt> out) const {
  int pivot = 0;
  EncodeMatrix<LHSIndexer>(mat, meta, pivot, need_encrypt, out);
}

void MatMatProtocol::EncodeRHS(const ArrayRef& mat, const Meta& meta,
                               bool need_encrypt,
                               absl::Span<RLWEPt> out) const {
  int pivot = 1;
  EncodeMatrix<RHSIndexer>(mat, meta, pivot, need_encrypt, out);
}

template <typename Indexer>
void MatMatProtocol::EncodeMatrix(const ArrayRef& mat, const Meta& meta,
                                  int pivot, bool need_encrypt,
                                  absl::Span<RLWEPt> out) const {
  const int R = pivot;
  const int C = pivot + 1;
  auto subshape = GetSubMatShape(meta);
  int64_t num_row_blocks = CeilDiv(meta.dims[R], subshape[R]);
  int64_t num_col_blocks = CeilDiv(meta.dims[C], subshape[C]);
  SPU_ENFORCE_EQ(static_cast<int64_t>(out.size()),
                 num_row_blocks * num_col_blocks);

  Shape2D mat_shape = {meta.dims[R], meta.dims[C]};
  Shape2D submat_shape = {subshape[R], subshape[C]};

  Indexer indexer(subshape);
  std::array<int64_t, 2> extents;
  for (int64_t rb = 0; rb < num_row_blocks; ++rb) {
    int64_t row_start = rb * subshape[R];
    int64_t row_end = std::min(meta.dims[R], row_start + subshape[R]);
    extents[0] = row_end - row_start;
    for (int64_t cb = 0; cb < num_col_blocks; ++cb) {
      int64_t col_start = cb * subshape[C];
      int64_t col_end = std::min(meta.dims[C], col_start + subshape[C]);
      extents[1] = col_end - col_start;

      auto flatten =
          ConcatSubMatrix<Indexer>(mat, mat_shape, {row_start, col_start},
                                   extents, submat_shape, poly_deg_, indexer);

      vencoder_->Forward(flatten, &out[rb * num_col_blocks + cb], need_encrypt);
    }
  }
}

template <>
void MatMatProtocol::FusedMulAddInplace(RLWECt& acc, const RLWECt& lhs,
                                        const RLWEPt& rhs) const {
  SPU_ENFORCE(lhs.parms_id() == rhs.parms_id());
  auto cntxt_data = context_.get_context_data(lhs.parms_id());
  SPU_ENFORCE(cntxt_data != nullptr);

  if (acc.size() == 0) {
    acc.resize(context_, lhs.parms_id(), lhs.size());
    acc.is_ntt_form() = lhs.is_ntt_form();
  } else {
    SPU_ENFORCE_EQ(acc.size(), lhs.size());
    SPU_ENFORCE(acc.parms_id() == lhs.parms_id());
    SPU_ENFORCE(acc.is_ntt_form() && lhs.is_ntt_form());
  }

  auto parms = cntxt_data->parms();
  size_t coeff_count = parms.poly_modulus_degree();
  const auto& modulus = parms.coeff_modulus();

  for (size_t k = 0; k < lhs.size(); ++k) {
    using namespace seal::util;
    const auto* op0 = lhs.data(k);
    const auto* op1 = rhs.data();
    auto* dst = acc.data(k);
    for (size_t j = 0; j < modulus.size(); ++j) {
      if (use_montgomery_fma_) {
        const uint64_t prime = modulus[j].value();
        const uint64_t prime_inv = montgomery_precond_.at(j);
        const uint64_t tbl[2]{prime, 0};

        unsigned long long wide[2], H;
        for (size_t i = 0; i < coeff_count; ++i) {
          multiply_uint64(*op0++, *op1++, wide);
          uint64_t R = wide[0] * prime_inv;
          multiply_uint64_hw64(R, prime, &H);
          uint64_t r = static_cast<uint64_t>(wide[1] - H) + prime;
          r -= tbl[r < prime];
          r += *dst;
          *dst++ = (r - tbl[r < prime]);
        }
      } else {
        for (size_t i = 0; i < coeff_count; ++i, ++dst) {
          *dst = multiply_add_uint_mod(*op0++, *op1++, *dst, modulus[j]);
        }
      }
    }
  }
}

template <>
void MatMatProtocol::FusedMulAddInplace(RLWECt& acc, const RLWEPt& lhs,
                                        const RLWECt& rhs) const {
  FusedMulAddInplace<RLWECt, RLWECt, RLWEPt>(acc, rhs, lhs);
}

template <>
void MatMatProtocol::FusedMulAddInplace(RLWEPt& acc, const RLWEPt& lhs,
                                        const RLWEPt& rhs) const {
  SPU_ENFORCE(lhs.parms_id() == rhs.parms_id());
  SPU_ENFORCE(lhs.coeff_count() == rhs.coeff_count());
  auto cntxt = context_.get_context_data(lhs.parms_id());
  SPU_ENFORCE(cntxt != nullptr);

  if (acc.coeff_count() == 0) {
    // acc += lhs * rhs
    acc.parms_id() = seal::parms_id_zero;
    acc.resize(lhs.coeff_count());
    acc.parms_id() = lhs.parms_id();
  }

  size_t coeff_count = cntxt->parms().poly_modulus_degree();
  const auto& modulus = cntxt->parms().coeff_modulus();
  const auto* op0 = lhs.data();
  const auto* op1 = rhs.data();
  auto* dst = acc.data();
  for (const auto& prime : modulus) {
    using namespace seal::util;
    for (size_t i = 0; i < coeff_count; ++i, ++dst) {
      *dst = multiply_add_uint_mod(*op0++, *op1++, *dst, prime);
    }
  }
}

void TakeCoefficientsFromPoly(const RLWEPt& poly, size_t poly_degree,
                              size_t num_modulus,
                              absl::Span<const size_t> target_coeffs,
                              absl::Span<uint64_t> out) {
  SPU_ENFORCE_EQ(poly.coeff_count(), poly_degree * num_modulus);
  size_t n = target_coeffs.size();
  SPU_ENFORCE(n <= poly_degree);
  SPU_ENFORCE_EQ(n * num_modulus, out.size());
  for (size_t i = 0; i < n; ++i) {
    size_t pos = target_coeffs[i];
    for (size_t j = 0; j < num_modulus; ++j) {
      out[j * n + i] = poly.data()[j * poly_degree + pos];
    }
  }
}

ArrayRef MatMatProtocol::ParseResult(
    FieldType field, const Meta& meta, absl::Span<const RLWEPt> ans_poly,
    const ModulusSwitchHelper& ms_helper) const {
  auto subdims = GetSubMatShape(meta);
  Shape2D out_blks = {CeilDiv(meta.dims[0], subdims[0]),
                      CeilDiv(meta.dims[2], subdims[2])};
  SPU_ENFORCE_EQ(static_cast<int64_t>(ans_poly.size()),
                 out_blks[0] * out_blks[1]);

  ResultIndexer ans_indexer(subdims);
  std::vector<size_t> target_coeffs(subdims[0] * subdims[2]);

  for (int64_t r = 0; r < subdims[0]; ++r) {
    for (int64_t c = 0; c < subdims[2]; ++c) {
      target_coeffs.at(r * subdims[2] + c) = ans_indexer(r, c);
    }
  }

  ArrayRef matmat = ring_zeros(field, meta.dims[0] * meta.dims[2]);

  for (int64_t rb = 0; rb < out_blks[0]; ++rb) {
    const auto* this_ans = ans_poly.data() + rb * out_blks[1];
    int64_t row_start = rb * subdims[0];
    int64_t row_end = std::min(row_start + subdims[0], meta.dims[0]);
    int64_t row_ext = row_end - row_start;

    for (int64_t cb = 0; cb < out_blks[1]; ++cb) {
      int64_t col_start = cb * subdims[2];
      int64_t col_end = std::min(col_start + subdims[2], meta.dims[2]);
      int64_t col_ext = col_end - col_start;

      size_t num_modulus = this_ans[cb].coeff_count() / poly_deg_;
      std::vector<uint64_t> subset(target_coeffs.size() * num_modulus);
      TakeCoefficientsFromPoly(this_ans[cb], poly_deg_, num_modulus,
                               absl::MakeSpan(target_coeffs),
                               absl::MakeSpan(subset));

      auto result_poly =
          ms_helper.ModulusDownRNS(field, absl::MakeSpan(subset));

      for (int64_t r = 0; r < row_ext; ++r) {
        for (int64_t c = 0; c < col_ext; ++c) {
          int64_t dst_idx = (r + row_start) * meta.dims[2] + col_start + c;
          DISPATCH_ALL_FIELDS(field, "", [&]() {
            matmat.at<ring2k_t>(dst_idx) =
                result_poly.at<ring2k_t>(r * subdims[2] + c);
          });
        }
      }
    }
  }

  return matmat;
}

ArrayRef MatMatProtocol::ParseResult(FieldType field, const Meta& meta,
                                     absl::Span<const RLWEPt> ans_poly) const {
  return ParseResult(field, meta, ans_poly, ms_helper_);
}

void MatMatProtocol::ExtractLWEsInplace(const Meta& meta,
                                        absl::Span<RLWECt> out) const {
  auto subdims = GetSubMatShape(meta);
  SPU_ENFORCE_EQ(out.size(), GetOutSize(meta, subdims));
  ResultIndexer ans_indexer(subdims);
  Shape2D out_blks = {CeilDiv(meta.dims[0], subdims[0]),
                      CeilDiv(meta.dims[2], subdims[2])};

  std::set<size_t> to_keep;
  for (int64_t r = 0; r < subdims[0]; ++r) {
    for (int64_t c = 0; c < subdims[2]; ++c) {
      to_keep.insert(ans_indexer(r, c));
    }
  }

  seal::Evaluator evaluator(context_);
  for (int64_t rb = 0; rb < out_blks[0]; ++rb) {
    auto* this_ans = out.data() + rb * out_blks[1];
    int64_t row_start = rb * subdims[0];
    int64_t row_end = std::min(row_start + subdims[0], meta.dims[0]);
    int64_t row_ext = row_end - row_start;

    for (int64_t cb = 0; cb < out_blks[1]; ++cb) {
      int64_t col_start = cb * subdims[2];
      int64_t col_end = std::min(col_start + subdims[2], meta.dims[2]);
      int64_t col_ext = col_end - col_start;

      if (this_ans[cb].is_ntt_form()) {
        evaluator.transform_from_ntt_inplace(this_ans[cb]);
      }

      if (row_ext == subdims[0] && col_ext == subdims[2]) {
        KeepCoefficientsInplace(this_ans[cb], to_keep);
      } else {
        // margin cases
        std::set<size_t> to_keep_on_margin;
        for (int64_t r = 0; r < row_ext; ++r) {
          for (int64_t c = 0; c < col_ext; ++c) {
            to_keep_on_margin.insert(ans_indexer(r, c));
          }
        }
        KeepCoefficientsInplace(this_ans[cb], to_keep_on_margin);
      }
    }
  }
}

template <typename LHS, typename RHS, typename O>
void MatMatProtocol::DoCompute(absl::Span<const LHS> lhs,
                               absl::Span<const RHS> rhs, const Meta& meta,
                               absl::Span<O> out) const {
  auto subshape = GetSubMatShape(meta);
  size_t lhs_n = GetLeftSize(meta, subshape);
  size_t rhs_n = GetRightSize(meta, subshape);
  size_t out_n = GetOutSize(meta, subshape);
  SPU_ENFORCE_EQ(lhs.size(), lhs_n);
  SPU_ENFORCE_EQ(rhs.size(), rhs_n);
  SPU_ENFORCE_EQ(out.size(), out_n);

  Shape3D dims;
  for (int d : {0, 1, 2}) {
    dims[d] = CeilDiv(meta.dims[d], subshape[d]);
  }

  constexpr int kMinLoopDim2 = 4;
  if (dims[2] < kMinLoopDim2) {
    // k, i, j
    for (int64_t k = 0; k < dims[2]; ++k) {
      yacl::parallel_for(0, dims[0], 1, [&](size_t bgn, size_t end) {
        for (size_t i = bgn; i < end; ++i) {
          auto lhs_row = lhs.data() + i * dims[1];
          auto out_row = out.data() + i * dims[2];
          yacl::parallel_for(0, dims[1], 1, [&](size_t bgn, size_t end) {
            for (size_t j = bgn; j < end; ++j) {
              auto rhs_row = rhs.data() + j * dims[2];
              FusedMulAddInplace<O, LHS, RHS>(out_row[k], lhs_row[j],
                                              rhs_row[k]);
            }
          });
        }
      });
    }
  } else {
    // i, k, j
    for (int64_t i = 0; i < dims[0]; ++i) {
      auto lhs_row = lhs.data() + i * dims[1];
      auto out_row = out.data() + i * dims[2];
      yacl::parallel_for(0, dims[2], 1, [&](size_t bgn, size_t end) {
        for (size_t k = bgn; k < end; ++k) {
          for (int64_t j = 0; j < dims[1]; ++j) {
            auto rhs_row = rhs.data() + j * dims[2];
            FusedMulAddInplace<O, LHS, RHS>(out_row[k], lhs_row[j], rhs_row[k]);
          }
        }
      });
    }
  }
}

void MatMatProtocol::Compute(absl::Span<const RLWEPt> lhs_mat,
                             absl::Span<const RLWEPt> rhs_mat, const Meta& meta,
                             absl::Span<RLWEPt> out_mat) const {
  DoCompute<RLWEPt, RLWEPt, RLWEPt>(lhs_mat, rhs_mat, meta, out_mat);
}

void MatMatProtocol::Compute(absl::Span<const RLWECt> lhs_mat,
                             absl::Span<const RLWEPt> rhs_mat, const Meta& meta,
                             absl::Span<RLWECt> out_mat) const {
  DoCompute<RLWECt, RLWEPt, RLWECt>(lhs_mat, rhs_mat, meta, out_mat);
}

void MatMatProtocol::Compute(absl::Span<const RLWEPt> lhs_mat,
                             absl::Span<const RLWECt> rhs_mat, const Meta& meta,
                             absl::Span<RLWECt> out_mat) const {
  DoCompute<RLWEPt, RLWECt, RLWECt>(lhs_mat, rhs_mat, meta, out_mat);
}

void MatMatProtocol::Montgomerize(absl::Span<RLWEPt> pt) const {
  if (not use_montgomery_fma_) {
    return;
  }
  size_t n = pt.size();
  for (size_t i = 0; i < n; ++i) {
    auto cntxt_data = context_.get_context_data(pt[i].parms_id());
    SPU_ENFORCE(cntxt_data != nullptr);
    const auto& modulus = cntxt_data->parms().coeff_modulus();

    uint64_t* pt_rns = pt[i].data();
    for (const auto& moduli : modulus) {
      uint64_t prime = moduli.value();
      uint64_t r0 = moduli.const_ratio()[0];  // r0 = hi64(2^128 / prime)
      uint64_t r1 = moduli.const_ratio()[1];  // r1 = lo64(2^128 / prime)
      // lazy Montgomery form, i.e., in the range [0, 2p).
      std::transform(pt_rns, pt_rns + poly_deg_, pt_rns,
                     [r0, r1, prime](uint64_t a) -> uint64_t {
                       // SEAL requires explicit ULL type.
                       unsigned long long hi;
                       seal::util::multiply_uint64_hw64(a, r0, &hi);
                       return -((a * r1) + static_cast<uint64_t>(hi)) * prime;
                     });
      pt_rns += poly_deg_;
    }
  }
}

}  // namespace spu::mpc::cheetah
