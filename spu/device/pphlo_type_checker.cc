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

#include "spu/device/pphlo_type_checker.h"

#include "spu/dialect/pphlo_types.h"

namespace spu::device {

namespace {

DataType getDType(const mlir::Type &type) {
  if (auto ft = type.dyn_cast<mlir::FloatType>()) {
    return DT_FXP;
  }
  if (auto it = type.dyn_cast<mlir::IntegerType>()) {
    if (it.getWidth() == 1) {
      return DT_I1;
    }
    switch (it.getWidth()) {
    case 8:
      return it.isUnsigned() ? DT_U8 : DT_I8;
    case 16:
      return it.isUnsigned() ? DT_U16 : DT_I16;
    case 32:
      return it.isUnsigned() ? DT_U32 : DT_I32;
    case 64:
      return it.isUnsigned() ? DT_U64 : DT_I64;
    }
  }
  YASL_THROW("Hit unknown mlir type");
}

} // namespace

std::string toString(const ::mlir::Type &type) {
  std::string str;
  llvm::raw_string_ostream os(str);
  type.print(os);
  return str;
}

void checkShape(llvm::ArrayRef<int64_t> mlir_shape,
                const absl::Span<const int64_t> rt_shape) {
  YASL_ENFORCE(mlir_shape.size() == rt_shape.size(),
               "Runtime shape mismatch, expected={}, got={}", mlir_shape.size(),
               rt_shape.size());

  for (size_t idx = 0; idx < mlir_shape.size(); ++idx) {
    YASL_ENFORCE(mlir_shape[idx] == rt_shape[idx],
                 "Runtime shape mismatch at dim {}, expected={}, got={}", idx,
                 fmt::join(mlir_shape, "x"), fmt::join(rt_shape, "x"));
  }
}

void checkType(::mlir::RankedTensorType type, const hal::Value &v) {
  // Check shape
  checkShape(type.getShape(), v.shape());

  // dType checker
  mlir::pphlo::TypeTools tool;
  auto expectedType = getDType(tool.getExpressedType(type));
  YASL_ENFORCE(expectedType == v.dtype(), "Expected Type {}, got {}",
               expectedType, v.dtype());

  // vType checker
  if (tool.isMPCType<::mlir::pphlo::PublicType>(type)) {
    YASL_ENFORCE(v.isPublic());
  } else if (tool.isMPCType<::mlir::pphlo::SecretType>(type)) {
    YASL_ENFORCE(v.isSecret());
  } else {
    YASL_ENFORCE("Unknown vtype");
  }
}

} // namespace spu::device
