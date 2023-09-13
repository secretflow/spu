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

#pragma once
#include "libspu/core/prelude.h"
#include "libspu/mpc/cheetah/arith/common.h"
#include "libspu/mpc/cheetah/arith/vector_encoder.h"
#include "libspu/mpc/cheetah/rlwe/types.h"

namespace spu::mpc::cheetah {

class MatMatProtocol {
 public:
  struct Meta {
    // LHS dims[0]xdims[1], RHS dims[1]xdims[2]
    Shape3D dims;
  };

  explicit MatMatProtocol(const seal::SEALContext& context,
                          const ModulusSwitchHelper& ms_helper,
                          bool disable_pack = false);

  size_t GetLeftSize(const Meta& meta) const {
    return GetLeftSize(meta, GetSubMatShape(meta));
  }

  size_t GetRightSize(const Meta& meta) const {
    return GetRightSize(meta, GetSubMatShape(meta));
  }

  size_t GetOutSize(const Meta& meta) const {
    return GetOutSize(meta, GetSubMatShape(meta));
  }

  size_t GetLeftSize(const Meta& meta, const Shape3D& subshape) const;

  size_t GetRightSize(const Meta& meta, const Shape3D& subshape) const;

  size_t GetOutSize(const Meta& meta, const Shape3D& subshape) const;

  Shape3D GetSubMatShape(const Meta& meta) const;

  static Shape3D GetSubMatShape(const Meta& meta, int64_t poly_deg,
                                bool disable_pack = false);

  void EncodeLHS(const NdArrayRef& lhs_mat, const Meta& meta, bool need_encrypt,
                 absl::Span<RLWEPt> out) const;

  void EncodeRHS(const NdArrayRef& rhs_mat, const Meta& meta, bool need_encrypt,
                 absl::Span<RLWEPt> out) const;

  bool IsValidMeta(const Meta& meta) const;

  NdArrayRef ParseResult(FieldType field, const Meta& meta,
                         absl::Span<const RLWEPt> ans_poly) const;

  NdArrayRef ParseResult(FieldType field, const Meta& meta,
                         absl::Span<const RLWEPt> ans_poly,
                         const ModulusSwitchHelper& msh) const;

  // Coefficients via Packed Batched MatMul
  // output shape batch_size x dims[0] x dims[2]
  NdArrayRef ParseBatchPackedResult(FieldType field, size_t batch_size,
                                    const Meta& meta,
                                    absl::Span<const RLWEPt> polys,
                                    const ModulusSwitchHelper& msh) const;

  // Coefficients via Packed MatMul
  // output shape dims[0] x dims[2]
  NdArrayRef ParsePackedResult(FieldType field, const Meta& meta,
                               absl::Span<const RLWEPt> ans_poly,
                               const ModulusSwitchHelper& msh) const;

  void ExtractLWEsInplace(const Meta& meta, absl::Span<RLWECt> rlwe) const;

  // LHS_mat * RHS_mat
  // LHS = RLWECt, RHS = RLWEPt (when LHS is smaller)
  // LHS = RLWEPt, RHS = RLWECt (when RHS is smaller)
  // LHS = RLWEPt, RHS = RLWEPt (for debugging)
  void Compute(absl::Span<const RLWEPt> lhs_mat,
               absl::Span<const RLWEPt> rhs_mat, const Meta& meta,
               absl::Span<RLWEPt> out_mat) const;

  void Compute(absl::Span<const RLWECt> lhs_mat,
               absl::Span<const RLWEPt> rhs_mat, const Meta& meta,
               absl::Span<RLWECt> out_mat) const;

  void Compute(absl::Span<const RLWEPt> lhs_mat,
               absl::Span<const RLWECt> rhs_mat, const Meta& meta,
               absl::Span<RLWECt> out_mat) const;

 private:
  // work horse
  template <typename LHS, typename RHS, typename O>
  void DoCompute(absl::Span<const LHS> lhs, absl::Span<const RHS> rhs,
                 const Meta& meta, absl::Span<O> out) const;

  // accum += x * y
  template <class T0, class T1, class T2>
  void FusedMulAddInplace(T0& accum, const T1& x, const T2& y) const;

  bool IsValidSubShape(const Shape3D& shape) const;

  enum class LayoutType { row_major, col_major };

  template <typename Indexer>
  void EncodeMatrix(const NdArrayRef& mat, const Meta& meta, int pivot,
                    bool need_encrypt, LayoutType layout,
                    absl::Span<RLWEPt> out) const;

  int64_t poly_deg_{0};
  bool disable_pack_ = false;
  seal::SEALContext context_;
  ModulusSwitchHelper msh_;
  std::unique_ptr<VectorEncoder> vencoder_{nullptr};
};

}  // namespace spu::mpc::cheetah