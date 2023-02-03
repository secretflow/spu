// Copyright 2022 Ant Group Co., Ltd.
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
include "pphlo_ops.td"

// select(not(p), t, f) => select(p, f, t)
def FusePredNegIntoSelect : Pat<
  (PPHLO_SelectOp (PPHLO_NotOp $pred), $on_true, $on_false),
  (PPHLO_SelectOp $pred, $on_false, $on_true)>;
