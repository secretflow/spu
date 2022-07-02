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

#include "spu/device/frame.h"

#include "mlir/IR/BuiltinTypes.h"
#include "yasl/base/exception.h"

#include "spu/device/pphlo_type_checker.h"
#include "spu/dialect/pphlo_types.h"

namespace spu::device {

void Frame::releaseValue(::mlir::Value operand) {
  YASL_ENFORCE(!segments_.empty(),
               "Need at least one activate segment running");
  segments_.back().values_.erase(operand);
}

void Frame::addValue(::mlir::Value operand, hal::Value &&val) {
  YASL_ENFORCE(!segments_.empty(),
               "Need at least one activate segment running");
  segments_.back().values_[operand] = std::move(val);
}

void Frame::addValue(::mlir::Value operand, const hal::Value &val) {
  YASL_ENFORCE(!segments_.empty(),
               "Need at least one activate segment running");
  segments_.back().values_[operand] = val;
}

const hal::Value *Frame::getValue(::mlir::Value operand) const {
  const hal::Value *val = nullptr;
  YASL_ENFORCE(!segments_.empty());
  for (auto siter = segments_.rbegin(); siter != segments_.rend(); ++siter) {
    auto iter = siter->values_.find(operand);
    if (iter != siter->values_.end()) {
      val = &iter->second;
      break;
    }
  }
  // If type checker is enabled, do it at getter time
  if ((val != nullptr) && with_type_checker_) {
    checkType(operand.getType().dyn_cast<::mlir::RankedTensorType>(), *val);
  }

  return val;
}

} // namespace spu::device
