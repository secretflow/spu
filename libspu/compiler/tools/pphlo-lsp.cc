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

#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

#include "libspu/dialect/pphlo/dialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::mhlo::MhloDialect, mlir::stablehlo::StablehloDialect,
                  mlir::spu::pphlo::PPHloDialect, mlir::func::FuncDialect>();
  mlir::func::registerInlinerExtension(registry);
  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
