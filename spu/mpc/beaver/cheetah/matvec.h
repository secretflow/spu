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

#pragma once
#include "yasl/base/exception.h"

#include "spu/mpc/beaver/cheetah/poly_encoder.h"
#include "spu/mpc/beaver/cheetah/types.h"

namespace spu::mpc {

class MatVecProtocol {
 public:
  struct Meta {
    size_t nrows;
    size_t ncols;
  };

  explicit MatVecProtocol(const seal::SEALContext& context,
                          const ModulusSwitchHelper& ms_helper);

  void EncodeVector(const Meta& meta, const ArrayRef& vec,
                    std::vector<RLWEPt>* out) const;

  void EncodeMatrix(const Meta& meta, const ArrayRef& mat,
                    std::vector<RLWEPt>* out) const;

  // matrix-vector multiplication. result at RLWECt *without* cleaning up the
  // un-used polynomial coefficients. Call `ExtractLWEs` to clean up
  // those coefficients.
  void MatVecNoExtract(const Meta& meta, const std::vector<RLWEPt>& encoded_mat,
                       const std::vector<RLWECt>& vec_cipher,
                       std::vector<RLWECt>* out) const;

  void MatVec(const Meta& meta, const std::vector<RLWEPt>& encoded_mat,
              const std::vector<RLWECt>& vec_cipher,
              std::vector<LWECt>* out) const;

  void ExtractLWEs(const Meta& meta, const std::vector<RLWECt>& rlwe,
                   std::vector<LWECt>* out) const;

  void ExtractLWEsInplace(const Meta& meta, std::vector<RLWECt>& rlwe) const;

  ArrayRef ParseMatVecResult(FieldType field, const Meta& meta,
                             const std::vector<RLWEPt>& rlwe) const;

  inline size_t poly_degree() const { return poly_deg_; }

 protected:
  bool IsValidMeta(const Meta& meta) const;

 private:
  size_t poly_deg_{0};

  PolyEncoder encoder_;
  seal::SEALContext context_;
};

}  // namespace spu::mpc
