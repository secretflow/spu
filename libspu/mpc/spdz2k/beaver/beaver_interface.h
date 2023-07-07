// Copyright 2023 Ant Group Co., Ltd.
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

#include "libspu/core/array_ref.h"

namespace spu::mpc::spdz2k {

class Beaver {
 public:
  using Triple = std::tuple<ArrayRef, ArrayRef, ArrayRef>;
  using Pair = std::pair<ArrayRef, ArrayRef>;
  using Pair_Pair = std::pair<Pair, Pair>;
  using Triple_Pair = std::pair<Triple, Triple>;

  virtual ~Beaver() = default;

  virtual uint128_t InitSpdzKey(FieldType field, size_t s) = 0;

  virtual ArrayRef AuthArrayRef(const ArrayRef& value, FieldType field,
                                size_t k, size_t s) = 0;

  virtual Pair AuthCoinTossing(FieldType field, size_t size, size_t k,
                               size_t s) = 0;

  virtual Triple_Pair AuthMul(FieldType field, size_t size, size_t k,
                              size_t s) = 0;

  virtual Triple_Pair AuthDot(FieldType field, size_t M, size_t N, size_t K,
                              size_t k, size_t s) = 0;

  virtual Triple_Pair AuthAnd(FieldType field, size_t size, size_t s) = 0;

  virtual Pair_Pair AuthTrunc(FieldType field, size_t size, size_t bits,
                              size_t k, size_t s) = 0;

  virtual Pair AuthRandBit(FieldType field, size_t size, size_t k,
                           size_t s) = 0;

  // Check the opened value only
  virtual bool BatchMacCheck(const ArrayRef& open_value, const ArrayRef& mac,
                             size_t k, size_t s) = 0;

  // Open the low k_bits of value only
  virtual std::pair<ArrayRef, ArrayRef> BatchOpen(const ArrayRef& value,
                                                  const ArrayRef& mac, size_t k,
                                                  size_t s) = 0;

  // public coin, used in malicious model, all party generate new seed, then
  // get exactly the same random variable.
  virtual ArrayRef genPublCoin(FieldType field, size_t numel) = 0;
};

}  // namespace spu::mpc::spdz2k
