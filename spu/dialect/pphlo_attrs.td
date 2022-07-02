//
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

#ifndef SPU_DIALECT_PPHLO_ATTRS
#define SPU_DIALECT_PPHLO_ATTRS

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "pphlo_dialect.td"

def PPHloDim : ArrayRefParameter<"int64_t", "Dimension">;

def GatherDimensionNumbers : AttrDef<PPHlo_Dialect, "GatherDimensionNumbers"> {
  let mnemonic = "gather";
  let summary = "Attribute that models the dimension information for gather";
  let parameters = (ins PPHloDim
                    : $offsetDims, PPHloDim
                    : $collapsedSliceDims, PPHloDim
                    : $startIndexMap, "int64_t"
                    : $indexVectorDim);
  let hasCustomAssemblyFormat = 1;
}

def ConvDimensionNumbers : AttrDef<PPHlo_Dialect, "ConvDimensionNumbers"> {
  let cppNamespace = "::mlir::pphlo";
  let mnemonic = "conv";
  let summary = "Structure of dimension information for conv op";
  let parameters = (ins
    "int64_t":$inputBatchDimension,
    "int64_t":$inputFeatureDimension,
    PPHloDim:$inputSpatialDimensions,
    "int64_t":$kernelInputFeatureDimension,
    "int64_t":$kernelOutputFeatureDimension,
    PPHloDim:$kernelSpatialDimensions,
    "int64_t":$outputBatchDimension,
    "int64_t":$outputFeatureDimension,
    PPHloDim:$outputSpatialDimensions
  );
  let hasCustomAssemblyFormat = 1;
}

#endif  // SPU_DIALECT_PPHLO_ATTRS
