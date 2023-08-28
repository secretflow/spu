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
#include "seal/util/polyarithsmallmod.h"
#include "spdlog/spdlog.h"
#include "yacl/utils/parallel.h"
#include "yacl/utils/platform_utils.h"

#include "libspu/mpc/cheetah/arith/vector_encoder.h"
#include "libspu/mpc/cheetah/rlwe/utils.h"
#include "libspu/mpc/utils/ring_ops.h"

template <>
struct std::hash<spu::mpc::cheetah::MatMatProtocol::Meta> {
  size_t operator()(
      const spu::mpc::cheetah::MatMatProtocol::Meta& s) const noexcept {
    using namespace spu::mpc::cheetah;
    // FIXME(lwj): use a better way for hash
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
  explicit LHSIndexer(const Shape3D& meta, size_t poly_deg) {
    offsets_[0] = meta[1] * meta[2];
    offsets_[1] = meta[1] - 1;
    poly_deg_ = poly_deg;
  }

  inline int64_t get(int64_t r, int64_t c, bool& flip_sign) const {
    flip_sign = r == 0 && c > 0;
    return poly_deg_ * static_cast<int>(flip_sign) + r * offsets_[0] - c;
  }

  int64_t offsets_[2];
  int64_t poly_deg_ = 0;
};

class RHSIndexer {
 public:
  explicit RHSIndexer(const Shape3D& meta, size_t /*poly_deg*/) {
    offset_ = meta[1];
  }

  inline int64_t get(int64_t r, int64_t c, bool& flip_sign) const {
    flip_sign = false;
    return c * offset_ + r;
  }

  int64_t offset_{0};
};

class ResultIndexer {
 public:
  explicit ResultIndexer(const Shape3D& meta) {
    offsets_[0] = meta[1] * meta[2];
    offsets_[1] = meta[1];
  }

  inline int64_t get(int64_t r, int64_t c) const {
    return r * offsets_[0] + c * offsets_[1];
  }

  int64_t offsets_[2];
};

template <typename Indexer>
NdArrayRef ConcatSubMatrix(const NdArrayRef& mat, const Shape2D& mat_shape,
                           const Shape2D& starts, const Shape2D& extents,
                           const Shape2D& submat_shape, int64_t num_coeff,
                           const Indexer& indexer) {
  const Type& eltype = mat.eltype();
  SPU_ENFORCE(eltype.isa<RingTy>(), "must be ring_type, got={}", eltype);
  // SPU_ENFORCE(mat.ndim() == 2, "should be a 2D matrix");
  SPU_ENFORCE_EQ(mat.numel(), mat_shape[0] * mat_shape[1]);
  SPU_ENFORCE(num_coeff >= submat_shape[0] * submat_shape[1]);
  for (size_t d : {0, 1}) {
    SPU_ENFORCE(starts[d] < mat_shape[d]);
    SPU_ENFORCE(extents[d] > 0);
    SPU_ENFORCE(starts[d] + extents[d] <= mat_shape[d]);
  }

  const auto field = eltype.as<Ring2k>()->field();
  // NOTE: zero padding via initialization
  NdArrayRef flatten = ring_zeros(field, {num_coeff});

  DISPATCH_ALL_FIELDS(field, "ConcatSubMat", [&]() {
    using uT = std::make_unsigned<ring2k_t>::type;

    for (int64_t r = 0, rr = starts[0]; r < extents[0]; ++r, ++rr) {
      for (int64_t c = 0, cc = starts[1]; c < extents[1]; ++c, ++cc) {
        bool flip_sign = false;
        auto idx = indexer.get(r, c, flip_sign);
        auto v = mat.at<uT>(rr * mat_shape[1] + cc);
        flatten.at<uT>(idx) = flip_sign ? -v : v;
      }
    }
  });

  return flatten;
}

MatMatProtocol::MatMatProtocol(const seal::SEALContext& context,
                               const ModulusSwitchHelper& ms_helper,
                               bool disable_pack)
    : disable_pack_(disable_pack), context_(context), msh_(ms_helper) {
  SPU_ENFORCE(context_.parameters_set());
  SPU_ENFORCE(context_.first_parms_id() == msh_.parms_id());

  poly_deg_ = context_.first_context_data()->parms().poly_modulus_degree();
  vencoder_ = std::make_unique<VectorEncoder>(context_, msh_);
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
  return GetSubMatShape(meta, poly_deg_, disable_pack_);
}

Shape3D MatMatProtocol::GetSubMatShape(const Meta& meta, int64_t poly_deg,
                                       bool disable_pack) {
  const Shape3D& dim3 = meta.dims;
  const double cpu_price = 1.0;
  const double bandwidth_price = 1000.0;

  Shape3D subshape;
  Shape3D blk;

  const int64_t n = poly_deg;
  double min_cost = std::numeric_limits<double>::max();
  for (int64_t d0 = 1; d0 <= std::min(n, dim3[0]); d0 += 1) {
    blk[0] = CeilDiv(dim3[0], d0);

    int64_t d1 = 1;
    while (d1 <= dim3[1]) {
      if (d0 * d1 > n) {
        break;
      }
      blk[1] = CeilDiv(dim3[1], d1);

      int64_t d2 = std::min(dim3[2], n / d0 / d1);
      blk[2] = CeilDiv(dim3[2], d2);

      int64_t sent_ct = std::min(blk[0], blk[2]) * blk[1];
      int64_t response_ct = CeilDiv(blk[0] * blk[2], d1);
      int64_t num_automorph =
          disable_pack ? 0 : CeilDiv(dim3[0] * dim3[2], n) * d1;
      int64_t num_ct_mul = blk[0] * blk[1] * blk[2];

      double cost = (sent_ct + response_ct) * bandwidth_price +
                    num_automorph * cpu_price + num_ct_mul * cpu_price / 10.0;

      if (cost <= min_cost) {
        min_cost = cost;
        subshape = {d0, d1, d2};
      }

      d1 = disable_pack ? d1 + 1 : d1 * 2;
    }
  }

  return subshape;
}

void MatMatProtocol::EncodeLHS(const NdArrayRef& mat, const Meta& meta,
                               bool need_encrypt,
                               absl::Span<RLWEPt> out) const {
  int pivot = 0;
  EncodeMatrix<LHSIndexer>(mat, meta, pivot, need_encrypt,
                           LayoutType::row_major, out);
}

void MatMatProtocol::EncodeRHS(const NdArrayRef& mat, const Meta& meta,
                               bool need_encrypt,
                               absl::Span<RLWEPt> out) const {
  int pivot = 1;
  EncodeMatrix<RHSIndexer>(mat, meta, pivot, need_encrypt,
                           LayoutType::col_major, out);
}

template <typename Indexer>
void MatMatProtocol::EncodeMatrix(const NdArrayRef& mat, const Meta& meta,
                                  int pivot, bool need_encrypt,
                                  LayoutType layout,
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

  Indexer indexer(subshape, poly_deg_);
  size_t num_jobs = num_row_blocks * num_col_blocks;
  size_t wload = CalculateWorkLoad(num_jobs);

  yacl::parallel_for(0, num_jobs, wload, [&](int64_t job_bgn, int64_t job_end) {
    std::array<int64_t, 2> extents;
    for (int64_t job_id = job_bgn; job_id < job_end; ++job_id) {
      int64_t rblk = job_id / num_col_blocks;
      int64_t cblk = job_id % num_col_blocks;
      int64_t row_start = rblk * subshape[R];
      int64_t row_end = std::min(meta.dims[R], row_start + subshape[R]);
      extents[0] = row_end - row_start;

      int64_t col_start = cblk * subshape[C];
      int64_t col_end = std::min(meta.dims[C], col_start + subshape[C]);
      extents[1] = col_end - col_start;

      auto flatten =
          ConcatSubMatrix<Indexer>(mat, mat_shape, {row_start, col_start},
                                   extents, submat_shape, poly_deg_, indexer);

      int64_t poly_idx = layout == LayoutType::row_major
                             ? rblk * num_col_blocks + cblk
                             : cblk * num_row_blocks + rblk;

      vencoder_->Forward(flatten, &out[poly_idx], need_encrypt);
    }
  });
}

template <>
void MatMatProtocol::FusedMulAddInplace(RLWECt& acc, const RLWECt& lhs,
                                        const RLWEPt& rhs) const {
  namespace sut = seal::util;
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

  std::vector<uint64_t> tmp(coeff_count);
  for (size_t k = 0; k < lhs.size(); ++k) {
    const auto* op0 = lhs.data(k);
    const auto* op1 = rhs.data();
    auto* dst = acc.data(k);

    for (const auto& moduli : modulus) {
      if (yacl::hasAVX2()) {
        sut::dyadic_product_coeffmod(op0, op1, coeff_count, moduli, tmp.data());
        sut::add_poly_coeffmod(tmp.data(), dst, coeff_count, moduli, dst);
        op0 += coeff_count;
        op1 += coeff_count;
        dst += coeff_count;
      } else {
        // un-roll 4
        for (size_t ii = 0; ii < coeff_count; ii += 4, dst += 4) {
          dst[0] = sut::multiply_add_uint_mod(*op0++, *op1++, dst[0], moduli);
          dst[1] = sut::multiply_add_uint_mod(*op0++, *op1++, dst[1], moduli);
          dst[2] = sut::multiply_add_uint_mod(*op0++, *op1++, dst[2], moduli);
          dst[3] = sut::multiply_add_uint_mod(*op0++, *op1++, dst[3], moduli);
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

NdArrayRef MatMatProtocol::ParseResult(FieldType field, const Meta& meta,
                                       absl::Span<const RLWEPt> ans_poly,
                                       const ModulusSwitchHelper& msh) const {
  auto subdims = GetSubMatShape(meta);
  Shape2D out_blks = {CeilDiv(meta.dims[0], subdims[0]),
                      CeilDiv(meta.dims[2], subdims[2])};
  SPU_ENFORCE_EQ(static_cast<int64_t>(ans_poly.size()),
                 out_blks[0] * out_blks[1]);

  ResultIndexer ans_indexer(subdims);
  std::vector<size_t> target_coeffs(subdims[0] * subdims[2]);

  for (int64_t r = 0; r < subdims[0]; ++r) {
    for (int64_t c = 0; c < subdims[2]; ++c) {
      target_coeffs.at(r * subdims[2] + c) = ans_indexer.get(r, c);
    }
  }

  NdArrayRef matmat = ring_zeros(field, {meta.dims[0] * meta.dims[2]});

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

      auto result_poly = msh.ModulusDownRNS(field, {subdims[0], subdims[2]},
                                            absl::MakeSpan(subset));

      for (int64_t r = 0; r < row_ext; ++r) {
        for (int64_t c = 0; c < col_ext; ++c) {
          int64_t dst_idx = (r + row_start) * meta.dims[2] + col_start + c;
          DISPATCH_ALL_FIELDS(field, "ParseResult", [&]() {
            matmat.at<ring2k_t>(dst_idx) =
                result_poly.at<ring2k_t>(r * subdims[2] + c);
          });
        }
      }
    }
  }

  return matmat.reshape({meta.dims[0], meta.dims[2]});
}

struct PackRLWEMappingHelper {
  explicit PackRLWEMappingHelper(size_t poly_deg, size_t gap, size_t num_polys)
      : gap_(gap), num_polys_(num_polys), group_size_(poly_deg / gap) {}

  // the i-th poly's j-th coeffient is packed into where ?
  std::pair<size_t, size_t> GetPackedIndex(size_t poly_idx,
                                           size_t offset) const {
    SPU_ENFORCE(poly_idx < num_polys_);
    SPU_ENFORCE(offset < group_size_);

    auto idx0 = poly_idx / gap_;
    auto idx1 = offset * gap_ + poly_idx % gap_;
    return {idx0, idx1};
  }

  size_t gap_;
  size_t num_polys_;
  size_t group_size_;
};

NdArrayRef MatMatProtocol::ParseBatchPackedResult(
    FieldType field, size_t batch_size, const Meta& meta,
    absl::Span<const RLWEPt> polys, const ModulusSwitchHelper& msh) const {
  auto subshape = GetSubMatShape(meta);
  const int64_t out_n = GetOutSize(meta, subshape);
  SPU_ENFORCE_EQ(polys.size(),
                 CeilDiv<size_t>(out_n * batch_size, subshape[1]));

  const int64_t gap = subshape[1];
  const int64_t total_polys = out_n * batch_size;
  const int64_t polys_per_dot = out_n;
  const int64_t numel_per_dot = meta.dims[0] * meta.dims[2];

  std::vector<NdArrayRef> decoded_vectors(polys.size());
  yacl::parallel_for(0, polys.size(), CalculateWorkLoad(polys.size()),
                     [&](int64_t bgn, int64_t end) {
                       for (int64_t i = bgn; i < end; ++i) {
                         decoded_vectors[i] = msh.ModulusDownRNS(
                             field, {poly_deg_},
                             {polys[i].data(), polys[i].coeff_count()});
                       }
                     });

  PackRLWEMappingHelper mapper(poly_deg_, gap, total_polys);

  NdArrayRef mat =
      ring_zeros(field, {static_cast<int64_t>(batch_size * numel_per_dot)});

  for (int64_t idx = 0; idx < total_polys; idx += polys_per_dot) {
    int64_t num_col_blk = CeilDiv(meta.dims[2], subshape[2]);
    int64_t out_mat_idx = idx / polys_per_dot;
    auto out_slice =
        mat.slice({out_mat_idx * numel_per_dot},
                  {out_mat_idx * numel_per_dot + numel_per_dot}, {1});

    for (int64_t poly_idx = 0; poly_idx < polys_per_dot; ++poly_idx) {
      int64_t out_rblk = poly_idx / num_col_blk;
      int64_t out_cblk = poly_idx % num_col_blk;

      int64_t out_row_bgn = out_rblk * subshape[0];
      int64_t out_col_bgn = out_cblk * subshape[2];
      int64_t row_ext =
          std::min(out_row_bgn + subshape[0], meta.dims[0]) - out_row_bgn;
      int64_t col_ext =
          std::min(out_col_bgn + subshape[2], meta.dims[2]) - out_col_bgn;

      for (int64_t r = 0; r < row_ext; ++r) {
        for (int64_t c = 0; c < col_ext; ++c) {
          int64_t coeff_idx = r * subshape[2] + c;
          auto o = mapper.GetPackedIndex(idx + poly_idx, coeff_idx);
          const NdArrayRef& dcd_vec = decoded_vectors.at(o.first);
          std::memcpy(
              &out_slice.at((r + out_row_bgn) * meta.dims[2] + out_col_bgn + c),
              &dcd_vec.at(o.second), decoded_vectors[0].elsize());
        }
      }
    }
  }

  return mat.reshape(
      {static_cast<int64_t>(batch_size), meta.dims[0], meta.dims[2]});
}

NdArrayRef MatMatProtocol::ParseResult(
    FieldType field, const Meta& meta,
    absl::Span<const RLWEPt> ans_poly) const {
  return ParseResult(field, meta, ans_poly, msh_);
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
      to_keep.insert(ans_indexer.get(r, c));
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
            to_keep_on_margin.insert(ans_indexer.get(r, c));
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

  if (dims[0] >= dims[2]) {
    auto wload = CalculateWorkLoad(dims[0]);
    yacl::parallel_for(0, dims[0], wload, [&](int64_t bgn, int64_t end) {
      // Loop dim0
      for (int64_t i = bgn; i < end; ++i) {
        // out[i, k]
        // NOTE(lwj): LHS is stored in row-major
        auto lhs_row = lhs.data() + i * dims[1];
        auto out_row = out.data() + i * dims[2];

        for (int64_t k = 0; k < dims[2]; ++k) {
          auto rhs_col = rhs.data() + k * dims[1];
          for (int64_t j = 0; j < dims[1]; ++j) {
            // rhs[j, k]
            FusedMulAddInplace<O, LHS, RHS>(out_row[k], lhs_row[j], rhs_col[j]);
          }
        }
      }
    });
  } else {
    auto wload = CalculateWorkLoad(dims[2]);
    yacl::parallel_for(0, dims[2], wload, [&](int64_t bgn, int64_t end) {
      // Loop dim2
      for (int64_t k = bgn; k < end; ++k) {
        // NOTE(lwj): RHS is stored in column-major
        auto rhs_col = rhs.data() + k * dims[1];
        for (int64_t i = 0; i < dims[0]; ++i) {
          auto lhs_row = lhs.data() + i * dims[1];
          auto out_row = out.data() + i * dims[2];
          for (int64_t j = 0; j < dims[1]; ++j) {
            FusedMulAddInplace<O, LHS, RHS>(out_row[k], lhs_row[j], rhs_col[j]);
          }
        }
      }
    });
  }
}

void MatMatProtocol::Compute(absl::Span<const RLWEPt> lhs_mat,
                             absl::Span<const RLWEPt> rhs_mat, const Meta& meta,
                             absl::Span<RLWEPt> out_mat) const {
  for (auto& ct : out_mat) {
    ct.release();
  }
  DoCompute<RLWEPt, RLWEPt, RLWEPt>(lhs_mat, rhs_mat, meta, out_mat);
}

void MatMatProtocol::Compute(absl::Span<const RLWECt> lhs_mat,
                             absl::Span<const RLWEPt> rhs_mat, const Meta& meta,
                             absl::Span<RLWECt> out_mat) const {
  for (auto& ct : out_mat) {
    ct.release();
  }
  DoCompute<RLWECt, RLWEPt, RLWECt>(lhs_mat, rhs_mat, meta, out_mat);
}

void MatMatProtocol::Compute(absl::Span<const RLWEPt> lhs_mat,
                             absl::Span<const RLWECt> rhs_mat, const Meta& meta,
                             absl::Span<RLWECt> out_mat) const {
  for (auto& ct : out_mat) {
    ct.release();
  }
  DoCompute<RLWEPt, RLWECt, RLWECt>(lhs_mat, rhs_mat, meta, out_mat);
}

}  // namespace spu::mpc::cheetah