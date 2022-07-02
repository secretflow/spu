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

#include <chrono>
#include <memory>
#include <unordered_map>

#include "spu/device/symbol_table.h"
#include "spu/hal/context.h"
#include "spu/hal/value.h"

#include "spu/spu.pb.h"

namespace spu::device {

struct OpExecutionRecord {
  // total number of executation.
  size_t count = 0;
  // total elapsed time.
  std::chrono::duration<double> time = {};
};

using OpExecutionRecords = std::unordered_map<std::string, OpExecutionRecord>;

// The executor interface, an executor evaluates a texted code with given
// inputs, and produce expected outputs.
class Executor {
protected:
  HalContext *hctx_ = nullptr;

  // Profiling thingy
  OpExecutionRecords op_profile_records_;

public:
  explicit Executor(HalContext *hctx) : hctx_(hctx){};

  virtual ~Executor() = default;

  // Return the HAL context.
  HalContext *getContext() const { return hctx_; }

  /// Run a code snippet with given inputs.
  // return a list of output values.
  virtual std::vector<hal::Value>
  run(const std::string &code, const std::vector<hal::Value> &inputs) = 0;

  /// Return the op profiling records.
  const OpExecutionRecords &getProfileRecords() const {
    return op_profile_records_;
  }

  /// Evaluate an spu executable with given environment.
  void runWithEnv(const ExecutableProto &exec, SymbolTable *env);

  ///
  void runWithEnv(const std::string &text,
                  const std::vector<std::string> &input_names,
                  const std::vector<std::string> &output_names,
                  SymbolTable *env);
};

} // namespace spu::device
