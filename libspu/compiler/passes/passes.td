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
    ListOption<"input_vis_list_", "input_vis_list", "std::string", "input visibilities to entry point function">,
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

def OptimizeSqrtPlusEps: Pass<"optimize-sqrt-plus-eps", "func::FuncOp"> {
  let summary = "Rewrite sqrt(x)+small_const into sqrt(x+small_const)";
  let constructor = "createOptimizeSqrtPlusEps()";
  let dependentDialects = ["pphlo::PPHloDialect"];
}

def RewriteDivSqrtPatterns: Pass<"rewrite-div-sqrt-pattern", "func::FuncOp"> {
  let summary = "Rewrite x/sqrt(x+eps) -> x*rsqrt(x+eps)";
  let constructor = "createRewriteDivSqrtPatterns()";
  let dependentDialects = ["pphlo::PPHloDialect"];
}

def ExpandSecretGather: Pass<"expand-secret-gather", "func::FuncOp"> {
  let summary = "Rewrite Gather with secret indexing to loop with DynamicUpdateSlice";
  let constructor = "createExpandSecretGatherPass()";
  let dependentDialects = ["pphlo::PPHloDialect"];
}

def OptimizeDenominatorWithBcast: Pass<"optimize-denominator-with-broadcast", "func::FuncOp"> {
  let summary = "Optimize x/broadcast(y) into x*broadcast(1/y)";
  let constructor = "createOptimizeDenominatorWithBroadcast()";
  let dependentDialects = ["pphlo::PPHloDialect"];
}
