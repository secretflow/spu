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

#include "libspu/dialect/pphlo/transforms/fixedpoint_conversion/config.h"

#include "libspu/core/prelude.h"

namespace mlir::spu::pphlo::fixedpoint {

::spu::ExpMode expModeFromString(std::string_view str) {
  if (str == "pade") {
    return ::spu::EXP_PADE;
  } else if (str == "taylor") {
    return ::spu::EXP_TAYLOR;
  } else {
    SPU_THROW("Unknow exp mode {}", str);
  }
}

::spu::LogMode logModeFromString(std::string_view str) {
  if (str == "pade") {
    return ::spu::LOG_PADE;
  } else if (str == "newton") {
    return ::spu::LOG_NEWTON;
  } else if (str == "minmax") {
    return ::spu::LOG_MINMAX;
  } else {
    SPU_THROW("Unknow log mode {}", str);
  }
}

::spu::SigmoidMode sigmoidModeFromString(std::string_view str) {
  if (str == "mm1") {
    return ::spu::SIGMOID_MM1;
  } else if (str == "seg3") {
    return ::spu::SIGMOID_SEG3;
  } else if (str == "real") {
    return ::spu::SIGMOID_REAL;
  } else {
    SPU_THROW("Unknow sigmoid mode {}", str);
  }
}

}  // namespace mlir::spu::pphlo::fixedpoint
