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

#include "libspu/core/cexpr.h"

#include <cmath>
#include <functional>
#include <sstream>
#include <type_traits>

#include "fmt/format.h"

#include "libspu/core/prelude.h"

namespace spu::ce {
namespace {

class ConstantExpr : public BaseExpr {
  Value const val_;

 public:
  explicit ConstantExpr(Value val) : val_(val) {}
  std::string expr() const override { return fmt::format("{}", val_); }
  Value eval(const Params& params) const override { return val_; }
};

class VariableExpr : public BaseExpr {
  std::string const name_;
  std::string const desc_;

 public:
  explicit VariableExpr(std::string name, std::string desc)
      : name_(std::move(name)), desc_(std::move(desc)) {}
  std::string expr() const override { return name_; }
  Value eval(const Params& params) const override {
    auto itr = params.find(name_);
    SPU_ENFORCE(itr != params.end(), "varialbe not found: {}", name_);
    return itr->second;
  }
};

class LogExpr : public BaseExpr {
  CExpr operand_;

 public:
  explicit LogExpr(CExpr operand) : operand_(std::move(operand)) {}
  std::string expr() const override {
    return fmt::format("log({})", operand_->expr());
  }
  Value eval(const Params& params) const override {
    return std::ceil((std::log2(operand_->eval(params))));
  }
};

class BaseBinaryExpr : public BaseExpr {
 public:
  virtual size_t priority() const = 0;
};

template <typename Fn, char const* Name, size_t kPriority>
class BinaryExpr : public BaseBinaryExpr {
  CExpr const lhs_;
  CExpr const rhs_;

 public:
  explicit BinaryExpr(CExpr lhs, CExpr rhs)
      : lhs_(std::move(lhs)), rhs_(std::move(rhs)) {}

  std::string expr() const override {
    // return fmt::format("({}){}({})", lhs_->expr(), Name, rhs_->expr());
    std::stringstream ss;
    auto print_operand = [&](const CExpr& operand) {
      if (auto bin_op = std::dynamic_pointer_cast<BaseBinaryExpr>(operand)) {
        if (bin_op->priority() < this->priority()) {
          ss << "(" << operand->expr() << ")";
        } else {
          ss << operand->expr();
        }
      } else {
        ss << operand->expr();
      }
    };
    print_operand(lhs_);
    ss << Name;
    print_operand(rhs_);
    return ss.str();
  }

  Value eval(const Params& params) const override {
    // Value eval(FieldType field, size_t npc) const override {
    return Fn{}(lhs_->eval(params), rhs_->eval(params));
  }

  size_t priority() const override { return kPriority; }
};

CExpr Add(const CExpr& x, const CExpr& y) {
  static char kAddName[] = "+";
  using AddExpr = BinaryExpr<std::plus<>, kAddName, 1>;
  return std::make_shared<AddExpr>(x, y);
}

CExpr Sub(const CExpr& x, const CExpr& y) {
  static char kSubName[] = "-";
  using SubExpr = BinaryExpr<std::minus<>, kSubName, 1>;
  return std::make_shared<SubExpr>(x, y);
}

CExpr Mul(const CExpr& x, const CExpr& y) {
  static char kMulName[] = "*";
  using MulExpr = BinaryExpr<std::multiplies<>, kMulName, 2>;
  return std::make_shared<MulExpr>(x, y);
}

}  // namespace

CExpr Const(Value v) { return std::make_unique<ConstantExpr>(v); }
CExpr Variable(std::string name, std::string desc) {
  return std::make_shared<VariableExpr>(std::move(name), std::move(desc));
}

CExpr Log(Value x) { return Log(Const(x)); }
CExpr Log(const CExpr& x) { return std::make_shared<LogExpr>(x); }

CExpr operator+(const CExpr& x, const CExpr& y) { return Add(x, y); }
CExpr operator+(const CExpr& x, Value y) { return Add(x, Const(y)); }
CExpr operator+(Value x, const CExpr& y) { return Add(Const(x), y); }

CExpr operator-(const CExpr& x, const CExpr& y) { return Sub(x, y); }
CExpr operator-(const CExpr& x, Value y) { return Sub(x, Const(y)); }
CExpr operator-(Value x, const CExpr& y) { return Sub(Const(x), y); }

CExpr operator*(const CExpr& x, const CExpr& y) { return Mul(x, y); }
CExpr operator*(const CExpr& x, Value y) { return Mul(x, Const(y)); }
CExpr operator*(Value x, const CExpr& y) { return Mul(Const(x), y); }

}  // namespace spu::ce
