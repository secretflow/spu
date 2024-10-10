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
  let dependentDialects = ["pphlo::PPHloDialect", "mlir::arith::ArithDialect"];
  let options = [
    ListOption<"input_vis_list_", "input_vis_list", "std::string", "input visibilities to entry point function">,
  ];
}

def LowerConversionCast : Pass<"lower-conversion-cast", "func::FuncOp"> {
  let summary = "Lower UnrealizedConversionCastOp created during dialect conversion.";
  let constructor = "createLowerConversionCastPass()";
  let dependentDialects = ["pphlo::PPHloDialect"];
}

def GeneralDecomposeOps : Pass<"general-decompose-ops", "func::FuncOp"> {
  let summary = "Decompose high-level ops into basic ops.";
  let constructor = "createGeneralDecomposeOps()";
  let dependentDialects = ["pphlo::PPHloDialect"];
}

def SecretDecomposeOps : Pass<"secret-decompose-ops", "func::FuncOp"> {
  let summary = "Decompose high-level ops into basic ops.";
  let constructor = "createSecretDecomposeOps()";
  let dependentDialects = ["pphlo::PPHloDialect"];
}

def ReduceTrunc : Pass<"reduce-truncation", "func::FuncOp"> {
  let summary = "Reduce number of truncation by reassociate ops.";
  let constructor = "createReduceTruncationPass()";
  let dependentDialects = ["pphlo::PPHloDialect"];
}

def LowerMixedTypeOp : Pass<"lower-mixed-type-op", "func::FuncOp"> {
  let summary = "Lower into mixed-type dot/mul to help reduce number of truncations";
  let constructor = "createLowerMixedTypeOpPass()";
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

def RewriteSignbitPatterns: Pass<"rewrite-signbit-pattern", "func::FuncOp"> {
  let summary = "Rewrite x >> (n-1) bits to SignOp";
  let constructor = "createRewriteSignbitPatterns()";
  let dependentDialects = ["pphlo::PPHloDialect"];
}

def OptimizeDenominatorWithBcast: Pass<"optimize-denominator-with-broadcast", "func::FuncOp"> {
  let summary = "Optimize x/broadcast(y) into x*broadcast(1/y)";
  let constructor = "createOptimizeDenominatorWithBroadcast()";
  let dependentDialects = ["pphlo::PPHloDialect"];
}

def SortLowering: Pass<"sort-lowering", "func::FuncOp"> {
  let summary = "Lower some simple sort to simple sort op";
  let constructor = "createSortLowering()";
  let dependentDialects = ["pphlo::PPHloDialect"];
}

def ConvertPushDown: Pass<"convert-push-down", "func::FuncOp"> {
  let summary = "Push convert later";
  let constructor = "createConvertPushDownPass()";
  let dependentDialects = ["pphlo::PPHloDialect"];
}

def PartialSortToTopK: Pass<"partial-sort-to-topk", "func::FuncOp"> {
  let summary = "Convert partial sort to topk";
  let constructor = "createPartialSortToTopK()";
  let dependentDialects = ["pphlo::PPHloDialect"];
}

def InlineSecretControlFlow: Pass<"inline-secret-control-flow", "func::FuncOp"> {
  let summary = "Flatten secret control flow";
  let constructor = "createInlineSecretControlFlow()";
  let dependentDialects = ["pphlo::PPHloDialect"];
}

def DotGeneralToDot: Pass<"dot-general-to-dot", "func::FuncOp"> {
  let summary = "Lower DotGeneral to Dot";
  let constructor = "createDotGeneralToDot()";
  let dependentDialects = ["pphlo::PPHloDialect"];
}

def LowerSecretFloatToFixedPoint: Pass<"lower-sfloat-to-fxp", "func::FuncOp"> {
  let summary = "Lower secret float to fixedpoint";
  let constructor = "createLowerSecretFloatToFxp()";
  let dependentDialects = ["pphlo::PPHloDialect"];
  let options = [
    // Fxp width config
    Option<"f16_width_",         "f16_width"   , "int64_t", "64" , "f16 to fxp width">,
    Option<"f32_width_",         "f32_width"   , "int64_t", "64" , "f32 to fxp width">,
    Option<"f64_width_",         "f64_width"   , "int64_t", "64" , "f64 to fxp width">,
    Option<"f16_fraction_bits_", "f16_fxp_bits", "int64_t", "18" , "f16 to fxp fraction bits">,
    Option<"f32_fraction_bits_", "f32_fxp_bits", "int64_t", "18" , "f32 to fxp fraction bits">,
    Option<"f64_fraction_bits_", "f64_fxp_bits", "int64_t", "18" , "f64 to fxp fraction bits">,
  ];
}

def ExpandFixedPointApproximations: Pass<"expand-fixedpoint-approx", "func::FuncOp"> {
  let summary = "Expand secret non-linear ops to fixedpoint based approximations";
  let constructor = "createExpandFixedPointApprox()";
  let dependentDialects = ["pphlo::PPHloDialect", "tensor::TensorDialect"];
  let options = [
    // Approximation config
    Option<"lower_accuracy_rsqrt_", "lower_accuracy_rsqrt", "bool", "false", "Use a lower accuracy rsqrt approximation">,
    // Div
    Option<"div_iter_", "divide_iter", "int64_t", "2", "Number of iterations for divide approximation">,
    // Exp
    Option<"exp_mode_", "exp_mode", "std::string", "\"taylor\"", "Algorithm for exponential approximation">,
    Option<"exp_iter_", "exp_iter", "int64_t"    , "8"     , "Number of iterations for exponential approximation">,
    // Log
    Option<"log_mode_" , "log_mode" , "std::string", "\"minmax\"", "Algorithm for log approximation">,
    Option<"log_iter_" , "log_iter" , "int64_t"    , "3"   , "Number of iterations for log approximation">,
    Option<"log_order_", "log_order", "int64_t"    , "8"  , "Number of orders for log approximation">,
    // Sigmoid
    Option<"sigmoid_mode_" , "sigmoid_mode" , "std::string", "\"real\"", "Algorithm for sigmoid approximation">,
    // Sine/Cosine
    Option<"sin_cos_iter_", "sin_cos_iter", "int64_t", "10", "Number of iterations for sine/cosine approximation">,
  ];
}

def LowerPPHloFloatInputs: Pass<"lower-pphlo-float-inputs", "func::FuncOp"> {
  let summary = "Legalize floating point to fixed point finalize stage";
  let constructor = "createLowerPPHloFloatInputs()";
  let dependentDialects = ["pphlo::PPHloDialect"];
  let options = [
    // Fxp width config
    Option<"f16_width_",         "f16_width"   , "int64_t", "64" , "f16 to fxp width">,
    Option<"f32_width_",         "f32_width"   , "int64_t", "64" , "f32 to fxp width">,
    Option<"f64_width_",         "f64_width"   , "int64_t", "64" , "f64 to fxp width">,
    Option<"f16_fraction_bits_", "f16_fxp_bits", "int64_t", "18" , "f16 to fxp fraction bits">,
    Option<"f32_fraction_bits_", "f32_fxp_bits", "int64_t", "18" , "f32 to fxp fraction bits">,
    Option<"f64_fraction_bits_", "f64_fxp_bits", "int64_t", "18" , "f64 to fxp fraction bits">,
  ];
}

def RangeOptimization: Pass<"range-optimization", "func::FuncOp"> {
  let summary = "Range based optimization";
  let constructor = "createRangeOptimization()";
  let dependentDialects = ["pphlo::PPHloDialect"];
}

def LegalizeToSCF : Pass<"legalize-to-scf", "func::FuncOp"> {
  let summary = "Legalize from PPHLO dialect to SCF dialect.";
  let constructor = "createLegalizeToSCF()";
  let dependentDialects = ["pphlo::PPHloDialect",
                           "mlir::tensor::TensorDialect",
                           "mlir::scf::SCFDialect"];
}

def LegalizeToTensor : Pass<"legalize-to-tensor", "func::FuncOp"> {
  let summary = "Legalize from PPHLO dialect to tensor dialect.";
  let constructor = "createLegalizeToTensor()";
  let dependentDialects = ["pphlo::PPHloDialect",
                           "mlir::tensor::TensorDialect",
                           "mlir::linalg::LinalgDialect"];
}

def LegalizeToLinalg : Pass<"legalize-to-linalg", "func::FuncOp"> {
  let summary = "Legalize from PPHLO dialect to linalg dialect.";
  let constructor = "createLegalizeToLinalg()";
  let dependentDialects = ["pphlo::PPHloDialect",
                           "mlir::tensor::TensorDialect",
                           "mlir::linalg::LinalgDialect"];
}

def LegalizeToArith : Pass<"legalize-to-arith", "func::FuncOp"> {
  let summary = "Legalize from PPHLO dialect to aruth dialect.";
  let constructor = "createLegalizeToArith()";
  let dependentDialects = ["pphlo::PPHloDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::math::MathDialect",
                           "mlir::scf::SCFDialect"];
}