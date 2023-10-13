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

#include <memory>

#include "libspu/core/ndarray_ref.h"

namespace spu::mpc::semi2k {

class Beaver {
 public:
  // TODO: replace NdArrayRef with none-typed buffer
  using Triple = std::tuple<NdArrayRef, NdArrayRef, NdArrayRef>;
  using Pair = std::pair<NdArrayRef, NdArrayRef>;

  virtual ~Beaver() = default;

  virtual Triple Mul(FieldType field, const Shape& shape) = 0;

  // TODO: change And interface to buffer(size_t bits, size_t numel)
  virtual Triple And(FieldType field, const Shape& shape) = 0;

  virtual Triple Dot(FieldType field, int64_t M, int64_t N, int64_t K) = 0;

  // ret[0] = random value in ring 2k
  // ret[1] = ret[0] >> bits
  // ABY3, truncation pair method.
  // Ref: Section 5.1.2 https://eprint.iacr.org/2018/403.pdf
  virtual Pair Trunc(FieldType field, const Shape& shape, size_t bits) = 0;

  // ret[0] = random value in ring 2k
  // ret[1] = (ret[0] << 1) >> (1+bits)
  //          as share of (ret[0] mod 2^(k-1)) / 2^bits
  // ret[2] = ret[0] >> (k-1)
  //          as share of MSB(ret[0]) randbit
  // use for Probabilistic truncation over Z2K
  // https://eprint.iacr.org/2020/338.pdf
  virtual Triple TruncPr(FieldType field, const Shape& shape, size_t bits) = 0;

  virtual NdArrayRef RandBit(FieldType field, const Shape& shape) = 0;

  // Generate share permutation pair.
  /*
          ┌───────────────────────┐
          │                       │   A i
  Perm    │      Permutation      ├─────►
  ───────►│                       │   B i
          │    Pair  Generator    ├─────►
          │                       │
          └───────────────────────┘

                InversePerm(A) = B

  if perm_rank == lctx->Rank(); perm not empty.
  */
  virtual Pair PermPair(FieldType field, const Shape& shape, size_t perm_rank,
                        absl::Span<const int64_t> perm_vec) = 0;

  virtual std::unique_ptr<Beaver> Spawn() = 0;
};

}  // namespace spu::mpc::semi2k
