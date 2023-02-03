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
//

#ifndef SPU_DIALECT_PPHLO_BASE_ENUMS
#define SPU_DIALECT_PPHLO_BASE_ENUMS

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Type visibility enum definitions.
//===----------------------------------------------------------------------===//
def PPHLO_VISIBILITY_PUBLIC : I32EnumAttrCase<"VIS_PUBLIC", 0>;
def PPHLO_VISIBILITY_SECRET : I32EnumAttrCase<"VIS_SECRET", 1>;
def PPHLO_VisibilityAttr
    : I32EnumAttr<"Visibility", "PPHLO type visibility, has MPC meaning.",
                  [PPHLO_VISIBILITY_PUBLIC, PPHLO_VISIBILITY_SECRET]> {
  let cppNamespace = "::mlir::pphlo";
  let genSpecializedAttr = 0;
}

#endif  // SPU_DIALECT_PPHLO_BASE_ENUMS
