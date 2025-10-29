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

#pragma once
#include <memory>

#include "absl/types/span.h"
#include "seal/util/ntt.h"
#include "seal/batchencoder.h"

#include "libspu/mpc/cheetah/arith/common.h"
#include "libspu/mpc/cheetah/rlwe/types.h"

namespace spu::mpc::cheetah {

class SIMDBatchMMProt {
 public:

  struct Meta {
    uint64_t batch;
    // LHS dims[0]xdims[1], RHS dims[1]xdims[2]
    // LHS is input, RHS is plaintext weight
    Shape3D dims;
  };


  static constexpr int kNoiseFloodBits = 40;

  explicit SIMDBatchMMProt(uint64_t simd_lane, uint64_t prime_modulus);

  ~SIMDBatchMMProt();

  SIMDBatchMMProt& operator=(const SIMDBatchMMProt&) = delete;

  SIMDBatchMMProt(const SIMDBatchMMProt&) = delete;

  SIMDBatchMMProt(SIMDBatchMMProt&&) = delete;

  // Same as SIMDMulProt
  void SymEncrypt(absl::Span<const RLWEPt> polys,
                  const RLWESecretKey& secret_key,
                  const seal::SEALContext& context, bool save_seed,
                  absl::Span<RLWECt> out) const;

  void EncodeBatch(absl::Span<const uint64_t> array,
                   absl::Span<RLWEPt> batch_out, seal::BatchEncoder& encoder) const;

  void DecodeBatch(absl::Span<const RLWEPt> polys,
                   absl::Span<uint64_t> array, seal::BatchEncoder& encoder) const;

  Shape2D ComputeInShape(const Meta& meta);

  size_t ComputeInputCtNum(const Meta& meta, Shape2D in_shape) const;

  size_t ComputeWeightPtNum(const Meta& meta, Shape2D in_shape) const;

  size_t ComputeOutputCtNum(const Meta& meta, Shape2D in_shape) const;


  // Transform weight matrix into diag packed vectors
  void PrepareWeightVector(const Meta& meta, Shape2D in_shape,
                           absl::Span<const uint64_t> weight,
                           absl::Span<uint64_t> weight_vec) const;

  // Transform input matrix into col packed vectors
  void PrepareInputVector(const Meta& meta, Shape2D in_shape,
                          absl::Span<const uint64_t> input,
                          absl::Span<uint64_t> input_vec) const;

  // Input col packed ct and diag packed pt
  // Output col packed ct
  void BatchMatMatMul(const Meta& meta, Shape2D in_shape,
                 absl::Span<RLWECt> lhs_input, absl::Span<const RLWEPt> rhs_weight,
                 const RLWEPublicKey& public_key, const GaloisKeys& gal_keys,
                 const seal::SEALContext& context, 
                 absl::Span<RLWECt> out) const;

  // Parse the result matrix from col packed polys
  // ans_poly is the decoded vector of the decrypted plaintext poly result
  void ParseResult(const Meta& meta, Shape2D in_shape,
                   absl::Span<const uint64_t> ans_poly,
                   absl::Span<uint64_t> res_mat) const;

  // Shape2D GetInShape() const { return in_shape_; }

  inline int64_t SIMDLane() const { return simd_lane_; }

  const seal::Modulus& modulus() const { return prime_modulus_; }


 private:
  void NoiseFloodInplace(RLWECt &ct, const seal::SEALContext &context);

  uint64_t simd_lane_;
  uint64_t row_size_;
  seal::Modulus prime_modulus_;
  std::unique_ptr<seal::util::NTTTables> encode_tabl_;
  // Shape2D in_shape_{0, 0};
};

} // namespace spu::mpc::cheetah






