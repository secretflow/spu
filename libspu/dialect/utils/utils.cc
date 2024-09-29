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

#include "libspu/dialect/utils/utils.h"

namespace mlir::spu {

mlir::func::FuncOp get_entrypoint(ModuleOp op) {
  // Get the main function
  auto entry_func = op.lookupSymbol<mlir::func::FuncOp>("main");
  if (!entry_func) {
    auto funcs = op.getOps<func::FuncOp>();
    for (auto fcn : funcs) {
      if (fcn.isPrivate()) {
        continue;
      }
      if (entry_func) {
        return nullptr;
      }
      entry_func = fcn;
    }
  }

  return entry_func;
}

APInt convertFromInt128(int64_t nbits, const int128_t& v) {
  return APInt(nbits, ArrayRef{static_cast<uint64_t>(v),
                               static_cast<uint64_t>(v >> 64)});
}

int128_t convertFromAPInt(const APInt& v) {
  int128_t ret = 0;
  if (v.isNegative()) {
    ret = -1;
  }
  std::memcpy(&ret, v.getRawData(),
              std::min<unsigned int>(v.getNumWords() * sizeof(int64_t),
                                     sizeof(int128_t)));
  return ret;
}

}  // namespace mlir::spu
