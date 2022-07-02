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

#include "spu/mpc/util/cexpr.h"

#include <functional>
#include <sstream>
#include <type_traits>

#include "fmt/format.h"

namespace spu::mpc::util {
namespace {

class ConstExpr : public BaseExpr {
  Value const val_;

 public:
  explicit ConstExpr(Value val) : val_(val) {}
  std::string expr() const override { return fmt::format("{}", val_); }
  Value eval(FieldType field, size_t npc) const override { return val_; }
};
CExpr makeConstExpr(Value v) { return std::make_shared<ConstExpr>(v); }

class VariableExpr : public BaseExpr {};

class VariableK : public VariableExpr {
 public:
  explicit VariableK() {}
  std::string expr() const override { return "k"; }
  Value eval(FieldType field, size_t npc) const override {
    return SizeOf(field) * 8;
  }
};
CExpr makeVariableK() { return std::make_shared<VariableK>(); }
class VariableN : public VariableExpr {
 public:
  explicit VariableN() {}
  std::string expr() const override { return "n"; }
  Value eval(FieldType field, size_t npc) const override { return npc; }
};
CExpr makeVariableN() { return std::make_shared<VariableN>(); }

using UnaryFnPtr = std::add_pointer<Value(Value)>::type;
template <UnaryFnPtr Fn, char const* Name>
class UnaryExpr : public BaseExpr {
  CExpr operand_;

 public:
  explicit UnaryExpr(CExpr operand) : operand_(std::move(operand)) {}

  std::string expr() const override {
    return fmt::format("{}({})", Name, operand_->expr());
  }

  Value eval(FieldType field, size_t npc) const override {
    return Fn(operand_->eval(field, npc));
  }
};

static char kLogName[] = "log";
inline Value logFn(Value in) { return std::ceil((std::log2(in))); }
using LogExpr = UnaryExpr<logFn, kLogName>;
CExpr makeLogExpr(const CExpr& in) { return std::make_shared<LogExpr>(in); }

class BaseBinaryExpr : public BaseExpr {
 public:
  virtual size_t priority() const = 0;
};
using BinaryFnPtr = std::add_pointer<Value(Value, Value)>::type;
template <BinaryFnPtr Fn, char const* Name, size_t kPriority>
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

  Value eval(FieldType field, size_t npc) const override {
    return Fn(lhs_->eval(field, npc), rhs_->eval(field, npc));
  }

  size_t priority() const override { return kPriority; }
};

char kAddName[] = "+";
Value addFn(Value x, Value y) { return x + y; }
using AddExpr = BinaryExpr<addFn, kAddName, 1>;
CExpr makeAddExpr(const CExpr& x, const CExpr& y) {
  return std::make_shared<AddExpr>(x, y);
}

char kSubName[] = "-";
Value subFn(Value x, Value y) { return x - y; }
using SubExpr = BinaryExpr<subFn, kSubName, 1>;
CExpr makeSubExpr(const CExpr& x, const CExpr& y) {
  return std::make_shared<SubExpr>(x, y);
}
char kMulName[] = "*";
Value mulFn(Value x, Value y) { return x * y; }
using MulExpr = BinaryExpr<mulFn, kMulName, 2>;
CExpr makeMulExpr(const CExpr& x, const CExpr& y) {
  return std::make_shared<MulExpr>(x, y);
}

}  // namespace

///////////////////////////////////////////////////////////////////////////////////
// public interface.
///////////////////////////////////////////////////////////////////////////////////
CExpr Const(Value v) { return makeConstExpr(v); }
CExpr K() { return makeVariableK(); }
CExpr N() { return makeVariableN(); }
CExpr Log(const CExpr& x) { return makeLogExpr(x); }
CExpr Log(Value x) { return makeLogExpr(makeConstExpr(x)); }
CExpr operator+(const CExpr& x, const CExpr& y) { return makeAddExpr(x, y); }
CExpr operator+(const CExpr& x, Value y) {
  return makeAddExpr(x, makeConstExpr(y));
}
CExpr operator+(Value x, const CExpr& y) {
  return makeAddExpr(makeConstExpr(x), y);
}
CExpr operator-(const CExpr& x, const CExpr& y) { return makeSubExpr(x, y); }
CExpr operator-(const CExpr& x, Value y) {
  return makeSubExpr(x, makeConstExpr(y));
}
CExpr operator-(Value x, const CExpr& y) {
  return makeSubExpr(makeConstExpr(x), y);
}
CExpr operator*(const CExpr& x, const CExpr& y) { return makeMulExpr(x, y); }
CExpr operator*(const CExpr& x, Value y) {
  return makeMulExpr(x, makeConstExpr(y));
}
CExpr operator*(Value x, const CExpr& y) {
  return makeMulExpr(makeConstExpr(x), y);
}

}  // namespace spu::mpc::util
