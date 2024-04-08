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

include "libspu/dialect/pphlo/dialect.td"

//===----------------------------------------------------------------------===//
// Sort direction enum definitions.
//===----------------------------------------------------------------------===//
def PPHLO_SORT_DIRECTION_ASC : I32EnumAttrCase<"ASC", 0>;
def PPHLO_SORT_DIRECTION_DES : I32EnumAttrCase<"DES", 1>;

def PPHLO_SortDirection : I32EnumAttr<"SortDirection",
    "Which mode to sort.",
    [
      PPHLO_SORT_DIRECTION_ASC,
      PPHLO_SORT_DIRECTION_DES
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::spu::pphlo";
}

def PPHLO_SortDirectionAttr : EnumAttr<PPHlo_Dialect, PPHLO_SortDirection, "sort_direction">;

#endif  // SPU_DIALECT_PPHLO_BASE_ENUMS
