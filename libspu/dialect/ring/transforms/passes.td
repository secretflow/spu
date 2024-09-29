//
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

// ring passes

include "mlir/Pass/PassBase.td"

def DecayPublicFixedPoint : Pass<"decay-public-fixedpoint", "func::FuncOp"> {
  let summary = "Decay public fixedpoint to signless integer";
  let constructor = "createDecayPublicFixedpoint()";
  let dependentDialects = ["ring::RingDialect",
                           "mlir::tensor::TensorDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::linalg::LinalgDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::math::MathDialect"
                          ];
}

def IntrinsicCleanup : Pass<"intrinsic-cleanup", "func::FuncOp"> {
  let summary = "Cleanup lowering internal intrinsic";
  let constructor = "createIntrinsicCleanup()";
}

def  : Pass<"decay-public-fixedpoint", "func::FuncOp"> {
  let summary = "Decay public fixedpoint to signless integer";
  let constructor = "createDecayPublicFixedpoint()";
  let dependentDialects = ["ring::RingDialect",
                           "mlir::tensor::TensorDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::linalg::LinalgDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::math::MathDialect"
                          ];
}

def PreRingLoweringOpDecompose : Pass<"pre-ring-lowering-op-decompose", "ModuleOp"> {
  let summary = "Op decompose for legalize to ring.";
  let constructor = "createPreRingLoweringOpDecompose()";
}

def PPHloLegalizeToRing : Pass<"pphlo-legalize-to-ring", "ModuleOp"> {
  let summary = "Legalize from PPHLO dialect to Ring dialect.";
  let constructor = "createLegalizeToRing()";
  let dependentDialects = ["ring::RingDialect",
                           "mlir::tensor::TensorDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::linalg::LinalgDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::math::MathDialect"
                          ];
}
