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
//

// Canonicalization patterns for the MHLO dialect.

include "mlir/IR/PatternBase.td"
include "libspu/dialect/pphlo/IR/ops.td"

// not_equal(x, y) => not(equal(x, y))
def NotEqualToNotWithEqual : Pat<
  (PPHLO_NotEqualOp $lhs, $rhs),
  (PPHLO_NotOp (PPHLO_EqualOp $lhs, $rhs))
>;

// greater_equal(x, y) => !less(x, y)
def GreaterEqualToNotLess : Pat<
  (PPHLO_GreaterEqualOp $lhs, $rhs),
  (PPHLO_NotOp (PPHLO_LessOp $lhs, $rhs))
>;

// less_equal(x, y) => !greater(x, y)
def LessEqualToNotGreater : Pat<
  (PPHLO_LessEqualOp $lhs, $rhs),
  (PPHLO_NotOp (PPHLO_GreaterOp $lhs, $rhs))
>;

// max(x, y) => select(greater(x, y), x, y)
def MaxToSelect: Pat<
  (PPHLO_MaxOp $lhs, $rhs),
  (PPHLO_SelectOp (PPHLO_GreaterOp $lhs, $rhs), $lhs, $rhs)
>;

// min(x, y) => select(less(x, y), x, y)
def MinToSelect : Pat<
  (PPHLO_MinOp $lhs, $rhs),
  (PPHLO_SelectOp (PPHLO_LessOp $lhs, $rhs), $lhs, $rhs)
>;

// greater(x, y) => less(y, x)
def GreaterToLess : Pat<
  (PPHLO_GreaterOp $lhs, $rhs),
  (PPHLO_LessOp $rhs, $lhs)
>;

// clamp(minv, x, maxv) => min(max(minv, x), maxv)
def ClampToMinMax : Pat<
  (PPHLO_ClampOp $minv, $operand, $maxv),
  (PPHLO_MinOp (PPHLO_MaxOp $minv, $operand), $maxv)
>;
