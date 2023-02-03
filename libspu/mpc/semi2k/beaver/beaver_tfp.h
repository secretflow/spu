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

#include "yacl/link/context.h"

#include "libspu/mpc/semi2k/beaver/trusted_party.h"

namespace spu::mpc::semi2k {

// Trusted First Party beaver implementation.
//
// Warn: The first party acts TrustedParty directly, it is NOT SAFE and SHOULD
// NOT BE used in production.
//
// Check security implications before moving on.
class BeaverTfpUnsafe final {
 protected:
  // Only for rank0 party.
  TrustedParty tp_;

  std::shared_ptr<yacl::link::Context> lctx_;

  PrgSeed seed_;

  PrgCounter counter_;

 public:
  using Triple = std::tuple<ArrayRef, ArrayRef, ArrayRef>;
  using Pair = std::pair<ArrayRef, ArrayRef>;

 public:
  explicit BeaverTfpUnsafe(std::shared_ptr<yacl::link::Context> lctx);

  Triple Mul(FieldType field, size_t size);

  Triple And(FieldType field, size_t size);

  Triple Dot(FieldType field, size_t M, size_t N, size_t K);

  Pair Trunc(FieldType field, size_t size, size_t bits);

  // ret[0] = random value in ring 2k
  // ret[1] = (ret[0] << 1) >> (1+bits)
  //          as share of (ret[0] mod 2^(k-1)) / 2^bits
  // ret[2] = ret[0] >> (k-1)
  //          as share of MSB(ret[0]) randbit
  // use for Probabilistic truncation over Z2K
  // https://eprint.iacr.org/2020/338.pdf
  Triple TruncPr(FieldType field, size_t size, size_t bits);

  ArrayRef RandBit(FieldType field, size_t size);

  std::shared_ptr<yacl::link::Context> GetLink() const { return lctx_; }
};

}  // namespace spu::mpc::semi2k
