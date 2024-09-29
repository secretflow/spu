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
#include <vector>

#include "absl/types/span.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"

#include "libspu/core/context.h"
#include "libspu/core/pt_buffer_view.h"
#include "libspu/dialect/pphlo/IR/ops.h"

namespace spu::kernel::hlo {

class HloBuilder final {
 public:
  using SortDirection = mlir::spu::pphlo::SortDirection;

  using ReduceType = enum : uint32_t {
    REDUCE_SUM = 0,
    REDUCE_AVG = 1,
    REDUCE_MAX = 2,
    REDUCE_MIN = 3,
  };

 public:
  HloBuilder();

  HloBuilder(HloBuilder &) = delete;

  void compile(const std::vector<mlir::Value> &outputs);
  std::vector<spu::MemRef> execute(spu::SPUContext *spu_ctx,
                                   std::vector<spu::MemRef> params = {});
  std::string EmitCodes() const;

  mlir::Value Constant(const PtBufferView &view, const Shape &out_shape);
  mlir::Value Argument(spu::PtType pt_type, spu::Visibility visibility,
                       const Shape &shape);

  mlir::Value Add(const mlir::Value &lhs, const mlir::Value &rhs);
  mlir::Value Sub(const mlir::Value &lhs, const mlir::Value &rhs);
  mlir::Value Mul(const mlir::Value &lhs, const mlir::Value &rhs);
  mlir::Value Div(const mlir::Value &lhs, const mlir::Value &rhs);

  mlir::Value Equal(const mlir::Value &lhs, const mlir::Value &rhs);
  mlir::Value And(const mlir::Value &lhs, const mlir::Value &rhs);
  mlir::Value Xor(const mlir::Value &lhs, const mlir::Value &rhs);
  mlir::Value Or(const mlir::Value &lhs, const mlir::Value &rhs);
  mlir::Value NotEqual(const mlir::Value &lhs, const mlir::Value &rhs);

  mlir::Value Max(const mlir::Value &lhs, const mlir::Value &rhs);
  mlir::Value Min(const mlir::Value &lhs, const mlir::Value &rhs);

  mlir::Value Greater(const mlir::Value &lhs, const mlir::Value &rhs);
  mlir::Value GreaterEqual(const mlir::Value &lhs, const mlir::Value &rhs);

  mlir::Value Less(const mlir::Value &lhs, const mlir::Value &rhs);
  mlir::Value LessEqual(const mlir::Value &lhs, const mlir::Value &rhs);

  mlir::Value Remainder(const mlir::Value &lhs, const mlir::Value &rhs);

  mlir::Value Not(const mlir::Value &input);
  mlir::Value Sine(const mlir::Value &input);
  mlir::Value Cosine(const mlir::Value &input);

  mlir::Value Seal(const mlir::Value &input);
  mlir::Value Reveal(const mlir::Value &input);
  mlir::Value Cast(const mlir::Value &input, spu::Visibility dst_vtype,
                   spu::PtType dst_dtype);

  mlir::Value Concatenate(const std::vector<mlir::Value> &ops, int64_t axis);

  mlir::Value Pad(const mlir::Value &input, const mlir::Value &pad_value,
                  const Sizes &edge_low, const Sizes &edge_high,
                  const Sizes &inner);

  mlir::Value Reduce(absl::Span<const mlir::Value> inputs,
                     absl::Span<const mlir::Value> init_values,
                     const Axes &dims_to_reduce, ReduceType reduce_type,
                     bool ignore_init_values = false);

  mlir::Value Select(const mlir::Value &pred, const mlir::Value &on_true,
                     const mlir::Value &on_false);

  std::vector<mlir::Value> SimpleSort(absl::Span<const mlir::Value> inputs,
                                      int64_t sort_dim, SortDirection direction,
                                      int64_t num_keys = 1);

  mlir::Value Slice(const mlir::Value &input, const Index &start,
                    const Index &end, const Strides &strides);

  std::vector<mlir::Value> Shuffle(absl::Span<const mlir::Value> inputs,
                                   int64_t axis);

  mlir::Value FilterByMask(const mlir::Value &input,
                           absl::Span<const uint8_t> mask);

  mlir::Value LinearGather(const mlir::Value &input, const Index &indices);

  mlir::Value LinearScatter(const mlir::Value &input, const mlir::Value &update,
                            const Index &indices);

  mlir::Value Broadcast(const mlir::Value &input, const Shape &to_shape,
                        const Axes &in_dims = {0});

  // static is enough since it does not need to build IR and compile
  static std::shared_ptr<yacl::Buffer> Dump(SPUContext *ctx,
                                            const spu::MemRef &input,
                                            spu::PtType pt_type,
                                            int64_t fxp_bits = 0);

 private:
  // helper op
  void Return(const std::vector<mlir::Value> &outputs);

  mlir::Type CommonType(llvm::ArrayRef<mlir::Type> in_types);

  // MLIR thingy
  mlir::MLIRContext mlir_ctx_;
  mlir::ModuleOp module_;
  mlir::OpBuilder builder_;
  mlir::Location loc_;

  // ref to main function op which has been inserted into module
  mlir::func::FuncOp main_fun_op_;

  // some helper vars.
  mlir::spu::pphlo::TypeTools type_tools_;
};

}  // namespace spu::kernel::hlo
