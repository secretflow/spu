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

#include "mlir/Transforms/DialectConversion.h"
#include "yacl/base/int128.h"

#include "libspu/dialect/pphlo/IR/types.h"
#include "libspu/dialect/pphlo/transforms/fixedpoint_conversion/config.h"

namespace mlir::spu::pphlo::fixedpoint::builder {

class FxpBuilder {
 private:
  // MLIR thingy
  OpBuilder &builder_;
  Location loc_;
  // Fxp thingy
  Config config_;
  Type base_fxp_type_;
  // Type tools
  TypeTools tools_;

 public:
  FxpBuilder(OpBuilder &rewriter, Location loc, const Config &config,
             Type base_fxp_type)
      : builder_(rewriter),
        loc_(loc),
        config_(config),
        base_fxp_type_(base_fxp_type),
        tools_(loc_->getContext()) {}

  FxpBuilder(OpBuilder &rewriter, Location loc, Type base_fxp_type)
      : builder_(rewriter),
        loc_(loc),
        base_fxp_type_(base_fxp_type),
        tools_(loc_->getContext()) {}

  FxpBuilder(FxpBuilder &) = delete;

  void replaceBaseFxpType(Type new_type) { base_fxp_type_ = new_type; }

  TypeTools &getTypeTools() { return tools_; }

  const Config &getConfig() const { return config_; }

  Type getIntTypeWithSameWidth(Type type, bool unsign = false);
  int64_t getCurrentFxpBits();
  int64_t getCurrentFxpWidth();

  // Create a constant
  Value fxp_constant(double values);
  Value int_constant(int128_t values);
  Value uint_constant(uint128_t values);
  Value fxp_constant_with_type(Type fp_type, llvm::ArrayRef<double> values);
  Value int_constant_with_type(Type int_type, llvm::ArrayRef<int128_t> values);

  // Create ops
  Value mul(Value lhs, Value rhs, SignType sign = SignType::Unknown);
  Value mul_no_trunc(Value lhs, Value rhs);
  Value square(Value in);

  Value dot(Value lhs, Value rhs);
  Value truncation(Value in, int64_t bits_to_trunc,
                   SignType sign = SignType::Unknown);

  Value add(Value lhs, Value rhs);
  Value substract(Value lhs, Value rhs);

  Value select(Value pred, Value on_true, Value on_false, Type result_type);
  Value greater(Value lhs, Value rhs);
  Value less(Value lhs, Value rhs);
  Value equal(Value lhs, Value rhs);

  Value negate(Value in);
  Value bitcast(Value in, Type result_type);
  Value convert(Value in, Type result_type);

  Value concate(llvm::ArrayRef<Value> ops, int64_t axis);
  Value reshape(Value in, llvm::ArrayRef<int64_t> shape);

  Value clamp(Value min, Value in, Value max);
  Value floor(Value in);

  Value prefix_or(Value in);
  Value arshift(Value in, int64_t bits);
  Value rshift(Value in, int64_t bits);
  Value lshift(Value in, int64_t bits);

  Value bitrev(Value in, int64_t start, int64_t end);
  Value bitdeintel(Value in);
  Value bitparity(Value in, int64_t bits);

  Value popcnt(Value in, int64_t bits = 0);

  Value sign(Value in, bool ignore_zero = true);

  // name mangled
  Value xor_(Value lhs, Value rhs);
  Value and_(Value lhs, Value rhs);

  void debug_print(Value in);
};

}  // namespace mlir::spu::pphlo::fixedpoint::builder
