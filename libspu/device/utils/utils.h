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

#pragma once

#include "mlir/IR/Operation.h"

#include "libspu/spu.pb.h"

namespace spu::device {

std::string defaultOpNamePrinter(mlir::Operation &op);

spu::SemanticType getSemanticTypeFromMlirType(mlir::Type mlir_ty);

spu::PtType getPtTypeFromMlirType(mlir::Type mlir_ty);

}  // namespace spu::device
