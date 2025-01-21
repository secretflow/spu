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

#include <map>
#include <memory>
#include <string>

// The complexity expression library
namespace spu::ce {

using Value = size_t;
using Params = std::map<std::string, Value>;

class BaseExpr {
 public:
  virtual ~BaseExpr() = default;

  // Return the human-readable format of this expression.
  virtual std::string expr() const = 0;

  // Evaluate this expression, with given variable binding.
  virtual Value eval(const Params& params) const = 0;
};

using CExpr = std::shared_ptr<BaseExpr>;

CExpr Const(Value v);
CExpr Variable(std::string name, std::string desc);

// Expose common used parameters.
inline CExpr K() { return Variable("K", "Number of bits of a mod 2^k ring"); }
inline CExpr N() { return Variable("N", "Represent number of parties."); }

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

}  // namespace spu::ce
