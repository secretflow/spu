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
#include <string>

#include "spu/core/type_util.h"

namespace spu::mpc::util {

// The complexity expression library

using Value = size_t;

class BaseExpr {
 public:
  virtual ~BaseExpr() = default;
  virtual std::string expr() const = 0;
  virtual Value eval(FieldType field, size_t npc) const = 0;
};

using CExpr = std::shared_ptr<BaseExpr>;

// Represent number of bits of the ring, aka, the `k` of `2^k`
CExpr K();
// Represent number of parties.
CExpr N();

CExpr Unknown();
CExpr Const(Value v);

CExpr Log(const CExpr& x);
CExpr Log(Value x);

CExpr operator+(const CExpr& x, const CExpr& y);
CExpr operator+(const CExpr& x, Value y);
CExpr operator+(Value x, const CExpr& y);

CExpr operator-(const CExpr& x, const CExpr& y);
CExpr operator-(const CExpr& x, Value y);
CExpr operator-(Value x, const CExpr& y);

CExpr operator*(const CExpr& x, const CExpr& y);
CExpr operator*(const CExpr& x, Value y);
CExpr operator*(Value x, const CExpr& y);

}  // namespace spu::mpc::util
