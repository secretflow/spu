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

#include "mlir/IR/Attributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"

#include "spu/dialect/pphlo_struct_attrs.h.inc"

#define GET_ATTRDEF_CLASSES
#include "spu/dialect/pphlo_attrs.h.inc"

namespace mlir::pphlo {

void printConvolutionDimensions(AsmPrinter &p, Operation *,
                                ConvDimensionNumbersAttr dnums);

}
