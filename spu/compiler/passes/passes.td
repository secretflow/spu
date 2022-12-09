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

// pphlo passes

include "mlir/Pass/PassBase.td"

def HloLegalizeToPPHloPass : Pass<"hlo-legalize-to-pphlo", "ModuleOp"> {
  let summary = "Legalize from HLO dialect to pphlo dialect.";
  let constructor = "createLegalizeToPPHloPass()";
  let dependentDialects = ["pphlo::PPHloDialect"];
  let options = [
    Option<"io_visibility_json_", "io-visibility-json", "std::string", "",
           "io visibility of this module">,
  ];
}

def DecomposeComparison : Pass<"decompose-comparison", "func::FuncOp"> {
  let summary = "Decompose comparisons into basic ops.";
  let constructor = "createDecomposeComparisonPass()";
  let dependentDialects = ["pphlo::PPHloDialect"];
}

def LowerConversionCast : Pass<"lower-conversion-cast", "func::FuncOp"> {
  let summary = "Lower UnrealizedConversionCastOp created during dialect conversion.";
  let constructor = "createLowerConversionCastPass()";
  let dependentDialects = ["pphlo::PPHloDialect"];
}

def DecomposeMinMax : Pass<"decompose-minmax", "func::FuncOp"> {
  let summary = "Decompose min/max into select with greater/less ops.";
  let constructor = "createDecomposeMinMaxPass()";
  let dependentDialects = ["pphlo::PPHloDialect"];
}

def ReduceTrunc : Pass<"reduce-truncation", "func::FuncOp"> {
  let summary = "Reduce number of truncation by reassociate ops.";
  let constructor = "createReduceTruncationPass()";
  let dependentDialects = ["pphlo::PPHloDialect"];
}

def LowerMixedTypeOp : Pass<"lower-mixed-type-op", "func::FuncOp"> {
  let summary = "Lower into mixed-type dot/mul to help reduce number of trunctions";
  let constructor = "createLowerMixedTypeOpPass()";
  let dependentDialects = ["pphlo::PPHloDialect"];
}

def OptimizeMaxPooling: Pass<"optimize-maxpool", "func::FuncOp"> {
  let summary = "Optimize performance of select and scatter";
  let constructor = "createOptimizeMaxPoolingPass()";
  let dependentDialects = ["pphlo::PPHloDialect"];
}

def OptimizeSelect: Pass<"optimize-select", "func::FuncOp"> {
  let summary = "Preconvert pred to ashare for better select perf";
  let constructor = "createOptimizeSelectPass()";
  let dependentDialects = ["pphlo::PPHloDialect"];
}

def OptimizeSqrtToRsqrt: Pass<"optimize-sqrt-to-rsqrt", "func::FuncOp"> {
  let summary = "Rewrite 1/(sqrt(x)+small_const) into rsqrt(x+small_const)";
  let constructor = "createOptimizeSqrtToRsqrtPass()";
  let dependentDialects = ["pphlo::PPHloDialect"];
}
