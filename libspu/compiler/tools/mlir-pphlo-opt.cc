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

#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
// #include "mlir-hlo/Dialect/mhlo/transforms/register_passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"

#include "libspu/compiler/passes/register_passes.h"
#include "libspu/dialect/pphlo_dialect.h"

int main(int argc, char **argv) {
  mlir::registerTransformsPasses();
  mlir::pphlo::registerAllPPHloPasses();
  mlir::mhlo::registerAllMhloPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::mhlo::MhloDialect, mlir::stablehlo::StablehloDialect,
                  mlir::pphlo::PPHloDialect, mlir::func::FuncDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "MLIR pphlo pass driver\n", registry));
}
