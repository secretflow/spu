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

#include "spu/core/array_ref.h"

namespace spu::mpc {

// The beaver interface.
class Beaver {
 public:
  using Triple = std::tuple<ArrayRef, ArrayRef, ArrayRef>;
  using Pair = std::pair<ArrayRef, ArrayRef>;

  virtual ~Beaver() = default;

  virtual Triple Mul(FieldType field, size_t size) = 0;

  virtual Triple And(FieldType field, size_t size) = 0;

  virtual Triple Dot(FieldType field, size_t M, size_t N, size_t K) = 0;

  // out.b = out.a >> bits, only for TruncateABY3.
  virtual bool SupportTrunc() { return true; }
  virtual Pair Trunc(FieldType field, size_t size, size_t bits) = 0;

  // Return size of random bits, with given field.
  virtual bool SupportRandBit() { return true; }
  virtual ArrayRef RandBit(FieldType field, size_t size) = 0;
};

}  // namespace spu::mpc
