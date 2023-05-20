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

#pragma once

#include <string>
#include <vector>

#include "libspu/core/context.h"
#include "libspu/core/value.h"
#include "libspu/device/executor.h"
#include "libspu/device/symbol_table.h"

#include "libspu/spu.pb.h"

namespace spu::device {

void execute(OpExecutor *executor, SPUContext *sctx,
             const ExecutableProto &executable, SymbolTable *env);

///
void execute(OpExecutor *executor, spu::SPUContext *sctx,
             const std::string &text,
             const std::vector<std::string> &input_names,
             const std::vector<std::string> &output_names, SymbolTable *env);

}  // namespace spu::device
