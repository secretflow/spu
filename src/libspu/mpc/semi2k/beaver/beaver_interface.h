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

#include "yacl/base/buffer.h"

#include "libspu/core/shape.h"
#include "libspu/mpc/common/prg_tensor.h"
#include "libspu/spu.h"

namespace spu::mpc::semi2k {

class Beaver {
 public:
  using PrgSeedBuff = yacl::Buffer;

  enum ReplayStatus {
    Init = 1,
    Replay = 2,
    TransposeReplay = 3,
  };

  struct ReplayDesc {
    ReplayStatus status{Init};
    PrgCounter prg_counter;
    PrgSeed seed;
    std::vector<PrgSeedBuff> encrypted_seeds;
    int64_t size;
    FieldType field;
    ElementType eltype;
  };

  using Array = yacl::Buffer;
  using Triple = std::tuple<Array, Array, Array>;
  using PremTriple = std::tuple<Array, Array, Index>;
  using Pair = std::pair<Array, Array>;

  virtual ~Beaver() = default;

  virtual Triple Mul(FieldType field, int64_t size,
                     ReplayDesc* x_desc = nullptr, ReplayDesc* y_desc = nullptr,
                     ElementType eltype = ElementType::kRing) = 0;

  virtual Pair MulPriv(FieldType field, int64_t size,
                       ElementType eltype = ElementType::kRing) = 0;

  virtual Pair Square(FieldType field, int64_t size,
                      ReplayDesc* x_desc = nullptr) = 0;

  virtual Triple And(int64_t size /*in bytes*/) = 0;

  virtual Triple Dot(FieldType field, int64_t m, int64_t n, int64_t k,
                     ReplayDesc* x_desc = nullptr,
                     ReplayDesc* y_desc = nullptr) = 0;

  // ret[0] = random value in ring 2k
  // ret[1] = ret[0] >> bits
  // ABY3, truncation pair method.
  // Ref: Section 5.1.2 https://eprint.iacr.org/2018/403.pdf
  virtual Pair Trunc(FieldType field, int64_t size, size_t bits) = 0;

  // ret[0] = random value in ring 2k
  // ret[1] = (ret[0] << 1) >> (1+bits)
  //          as share of (ret[0] mod 2^(k-1)) / 2^bits
  // ret[2] = ret[0] >> (k-1)
  //          as share of MSB(ret[0]) randbit
  // use for Probabilistic truncation over Z2K
  // https://eprint.iacr.org/2020/338.pdf
  virtual Triple TruncPr(FieldType field, int64_t size, size_t bits) = 0;

  virtual Array RandBit(FieldType field, int64_t size) = 0;

  // Generate share permutation pair.
  /*
          ┌───────────────────────┐   A i
          │                       ├─────►
  size    │      Permutation      │   B i
 ────────►│                       ├─────►
          │    Pair  Generator    │   π
          │                       ├─────►
          └───────────────────────┘

           InversePermute(A, π) = B

  if rank == perm_rank ret[2] is π, otherwise, ret[2] is empty.
  perm_rank should use ret[2] as a Span<const int64_t>(buffer, size) view.
  */
  virtual PremTriple PermPair(FieldType field, int64_t size,
                              size_t perm_rank) = 0;

  // Generate another share permutation pair.
  /*
          ┌───────────────────────┐   A i
          │                       ├─────►
  size    │      Permutation      │   B i
 ────────►│                       ├─────►
          │    Pair  Generator    │   π
          │                       ├─────►
          └───────────────────────┘

           Permute(A, π) = B

  if rank == perm_rank ret[2] is π, otherwise, ret[2] is empty.
  perm_rank should use ret[2] as a Span<const int64_t>(buffer, size) view.
  */
  virtual PremTriple GeneralPermPair(FieldType field, int64_t size,
                                     size_t perm_rank) = 0;

  virtual std::unique_ptr<Beaver> Spawn() = 0;

  // ret[0] (in a share) = ret[1] (in b share)
  // ref: https://eprint.iacr.org/2020/338
  virtual Pair Eqz(FieldType field, int64_t size) = 0;
};

}  // namespace spu::mpc::semi2k
