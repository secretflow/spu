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

#include "libspu/dialect/pphlo/IR/assembly_format.h"

namespace mlir::spu::pphlo {

void printSliceRanges(OpAsmPrinter& p, Operation* op,
                      ArrayRef<int64_t> start_indices,
                      ArrayRef<int64_t> limit_indices,
                      ArrayRef<int64_t> strides) {
  p << "[";
  // Let's be safe if we're printing invalid IR somehow: this can't be parsed
  // back!
  if (start_indices.size() != limit_indices.size() ||
      start_indices.size() != strides.size()) {
    p << "start_indices: ";
    llvm::interleaveComma(start_indices, p);
    p << ", limit_indices: ";
    llvm::interleaveComma(limit_indices, p);
    p << ", strides: ";
    llvm::interleaveComma(strides, p);
    p << "]";
    return;
  }

  llvm::interleaveComma(llvm::zip(start_indices, limit_indices, strides), p,
                        [&](std::tuple<int64_t, int64_t, int64_t> pack) {
                          auto [start, limit, stride] = pack;
                          p << start << ":" << stride << ":" << limit;
                        });
  p << "]";
}

ParseResult parseSliceRanges(OpAsmParser& parser,
                             DenseI64ArrayAttr& start_indices,
                             DenseI64ArrayAttr& limit_indices,
                             DenseI64ArrayAttr& strides) {
  if (parser.parseLSquare()) {
    return failure();
  }
  // Parse groups of comma-separated: `start`:`stride`:`limit`
  SmallVector<int64_t> start;
  SmallVector<int64_t> limit;
  SmallVector<int64_t> stride;
  if (failed(parser.parseOptionalRSquare())) {
    do {
      start.emplace_back();
      stride.emplace_back();
      limit.emplace_back();
      if (parser.parseInteger(start.back()) || parser.parseColon() ||
          parser.parseInteger(stride.back()) || parser.parseColon() ||
          parser.parseInteger(limit.back())) {
        return failure();
      }
      if (succeeded(parser.parseOptionalRSquare())) {
        break;
      }
      if (failed(parser.parseComma())) {
        return failure();
      }
    } while (true);
  }

  start_indices = parser.getBuilder().getDenseI64ArrayAttr(start);
  limit_indices = parser.getBuilder().getDenseI64ArrayAttr(limit);
  strides = parser.getBuilder().getDenseI64ArrayAttr(stride);

  return success();
}

}  // namespace mlir::spu::pphlo
