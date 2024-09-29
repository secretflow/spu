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

#include <cstdint>

#include "mlir/Transforms/DialectConversion.h"

#include "libspu/dialect/pphlo/IR/types.h"

namespace mlir::spu::pphlo {

struct FxpWidthConfig {
  int64_t f16_width;
  int64_t f16_frac_bits;
  int64_t f32_width;
  int64_t f32_frac_bits;
  int64_t f64_width;
  int64_t f64_frac_bits;
};

class SecretFloatConverter : public TypeConverter {
 private:
  FxpWidthConfig config_;

  Type convertFloatType(FloatType type) const;

  Type convertSecretType(SecretType type) const;

 public:
  explicit SecretFloatConverter(FxpWidthConfig config);

  ShapedType toFixedPointIfPossible(ShapedType in) const;
};

class FloatConverter : public TypeConverter {
 private:
  FxpWidthConfig config_;

  Type convertFloatType(FloatType type) const;

 public:
  explicit FloatConverter(FxpWidthConfig config);
};

Value convertFloatToFixed(OpBuilder& builder, Location loc, Value in,
                          Type fxp_type);

}  // namespace mlir::spu::pphlo
