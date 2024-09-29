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

#include "libspu/device/executor.h"

namespace mlir::scf {
class SCFDialect;
}

namespace spu::device {

void dispatch(mlir::scf::SCFDialect *, SPUContext *sctx, SymbolScope *sscope,
              mlir::Operation &op, OpExecutor *executor,
              const ExecutionOptions &opts);

}  // namespace spu::device
