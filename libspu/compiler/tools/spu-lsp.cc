// Copyright 2023 Ant Group Co., Ltd.
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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

#include "libspu/dialect/pphlo/IR/dialect.h"
#include "libspu/dialect/ring/IR/dialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::mhlo::MhloDialect, mlir::stablehlo::StablehloDialect,
                  mlir::spu::pphlo::PPHloDialect, mlir::spu::ring::RingDialect,
                  mlir::linalg::LinalgDialect, mlir::arith::ArithDialect,
                  mlir::math::MathDialect, mlir::func::FuncDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
  mlir::func::registerInlinerExtension(registry);
  return static_cast<int>(
      mlir::failed(mlir::MlirLspServerMain(argc, argv, registry)));
}
