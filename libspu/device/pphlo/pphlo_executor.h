// Copyright 2022 Ant Group Co., Ltd.
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

#include <memory>

#include "libspu/core/context.h"
#include "libspu/device/executor.h"
#include "libspu/device/pphlo/pphlo_verifier.h"
#include "libspu/dialect/pphlo_ops.h"
#include "libspu/dialect/pphlo_types.h"
#include "libspu/kernel/hlo/casting.h"

namespace spu::device::pphlo {

class PPHloExecutor : public OpExecutor {
 public:
  void checkType(mlir::Type mlir_type, const spu::Value &v) const override;

  // return true if the operation has a corresponding kernel.
  bool hasKernel(mlir::Operation &op) const override;

  // run a kernel in a given region.
  void runKernelImpl(SPUContext *sctx, SymbolScope *sscope, mlir::Operation &op,
                     const ExecutionOptions &opts) override;
};

}  // namespace spu::device::pphlo
