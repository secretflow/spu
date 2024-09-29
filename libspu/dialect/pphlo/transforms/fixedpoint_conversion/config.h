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

#include <cstdint>
#include <string_view>

#include "libspu/spu.pb.h"

namespace mlir::spu::pphlo::fixedpoint {

::spu::ExpMode expModeFromString(std::string_view str);
::spu::LogMode logModeFromString(std::string_view str);
::spu::SigmoidMode sigmoidModeFromString(std::string_view str);

struct Config {
  bool lower_accuracy_rsqrt;
  int64_t div_iter;
  ::spu::ExpMode exp_mode;
  int64_t exp_iter;
  ::spu::LogMode log_mode;
  int64_t log_iter;
  int64_t log_order;
  ::spu::SigmoidMode sig_mode;
  int64_t sin_cos_iter;
};

}  // namespace mlir::spu::pphlo::fixedpoint
