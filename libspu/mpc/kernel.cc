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

#include "libspu/mpc/kernel.h"

namespace spu::mpc {

void RandKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& type = ctx->getParam<SemanticType>(0);
  const auto& shape = ctx->getParam<Shape>(1);
  ctx->pushOutput(proc(ctx, type, shape));
}

void RandPermKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& shape = ctx->getParam<Shape>(0);
  ctx->pushOutput(proc(ctx, shape));
}

void UnaryKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam(0);
  ctx->pushOutput(proc(ctx, in));
}

void RevealToKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam(0);
  const auto rank = ctx->getParam<size_t>(1);
  ctx->pushOutput(proc(ctx, in, rank));
}

void ShiftKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam(0);
  const auto& bits = ctx->getParam<Sizes>(1);

  SPU_ENFORCE(
      bits.size() == 1 || in.numel() == static_cast<int64_t>(bits.size()),
      "numel mismatch {} {}", in.numel(), bits.size());

  ctx->pushOutput(proc(ctx, in, bits));
}

void TruncKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam(0);
  const auto& bit = ctx->getParam<size_t>(1);
  // const auto& sign = ctx->getParam<SignType>(2);

  ctx->pushOutput(proc(ctx, in, bit));
}

void BinaryKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& lhs = ctx->getParam(0);
  const auto& rhs = ctx->getParam(1);

  SPU_ENFORCE(lhs.shape() == rhs.shape(), "shape mismatch {} {}", lhs.shape(),
              rhs.shape());

  ctx->pushOutput(proc(ctx, lhs, rhs));
}

void MatmulKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& lhs = ctx->getParam(0);
  const auto& rhs = ctx->getParam(1);

  SPU_ENFORCE(lhs.shape()[1] == rhs.shape()[0], "invalid shape {} {}", lhs,
              rhs);

  ctx->pushOutput(proc(ctx, lhs, rhs));
}

void Conv2DKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& lhs = ctx->getParam(0);
  const auto& rhs = ctx->getParam(1);
  auto stride_h = ctx->getParam<int64_t>(2);
  auto stride_w = ctx->getParam<int64_t>(3);

  ctx->pushOutput(proc(ctx, lhs, rhs, stride_h, stride_w));
}

void BitrevKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam(0);
  size_t start = ctx->getParam<size_t>(1);
  size_t end = ctx->getParam<size_t>(2);

  ctx->pushOutput(proc(ctx, in, start, end));
}

void TruncAKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam(0);
  size_t bits = ctx->getParam<size_t>(1);
  SignType sign = ctx->getParam<SignType>(2);

  ctx->pushOutput(proc(ctx, in, bits, sign));
}

void BitSplitKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam(0);
  size_t stride = ctx->getParam<size_t>(1);

  ctx->pushOutput(proc(ctx, in, stride));
}

void CastTypeKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& val = ctx->getParam(0);
  const auto& to_type = ctx->getParam<Type>(1);

  ctx->pushOutput(proc(ctx, val, to_type));
}

void PermKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& x = ctx->getParam(0);
  const auto& y = ctx->getParam(1);

  SPU_ENFORCE(x.shape() == y.shape(), "shape mismatch {} {}", x.shape(),
              x.shape());
  SPU_ENFORCE(x.shape().ndim() == 1, "input should be a 1-d memref");

  ctx->pushOutput(proc(ctx, x, y));
}

void GenInvPermKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam(0);
  bool is_ascending = ctx->getParam<bool>(1);
  SPU_ENFORCE(in.shape().ndim() == 1, "input should be a 1-d memref");

  ctx->pushOutput(proc(ctx, in, is_ascending));
}

void MergeKeysKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam<std::vector<MemRef>>(0);
  bool is_ascending = ctx->getParam<bool>(1);

  ctx->pushOutput(proc(ctx, in, is_ascending));
}

void BroadcastKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam(0);
  const auto& to_shape = ctx->getParam<Shape>(1);
  const auto& in_dims = ctx->getParam<Axes>(2);

  ctx->pushOutput(proc(ctx, in, to_shape, in_dims));
}

void DimsBasedKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam(0);
  const auto& axes = ctx->getParam<Axes>(1);

  ctx->pushOutput(proc(ctx, in, axes));
}

void ShapeBasedKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam(0);
  const auto& to_shape = ctx->getParam<Shape>(1);

  ctx->pushOutput(proc(ctx, in, to_shape));
}

void ExtractSliceKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam(0);
  const auto& offsets = ctx->getParam<Index>(1);
  const auto& sizes = ctx->getParam<Shape>(2);
  const auto& strides = ctx->getParam<Strides>(3);

  ctx->pushOutput(proc(ctx, in, offsets, sizes, strides));
}

void InsertSliceKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam(0);
  const auto& update = ctx->getParam(1);
  const auto& offsets = ctx->getParam<Index>(2);
  const auto& strides = ctx->getParam<Strides>(3);
  const auto& prefer_in_place = ctx->getParam<bool>(4);

  ctx->pushOutput(proc(ctx, in, update, offsets, strides, prefer_in_place));
}

void PadKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam(0);
  const auto& padding_value = ctx->getParam(1);
  const auto& edge_low = ctx->getParam<Sizes>(2);
  const auto& edge_high = ctx->getParam<Sizes>(3);

  ctx->pushOutput(proc(ctx, in, padding_value, edge_low, edge_high));
}

void ConcateKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& ins = ctx->getParam<std::vector<MemRef>>(0);
  const auto& axis = ctx->getParam<int64_t>(1);

  ctx->pushOutput(proc(ctx, ins, axis));
}

void DecomposeKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam(0);
  ctx->pushOutput(proc(ctx, in));
};

void ComposeKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& ins = ctx->getParam<std::vector<MemRef>>(0);
  ctx->pushOutput(proc(ctx, ins));
};

void OramOneHotKernel::evaluate(KernelEvalContext* ctx) const {
  auto target = ctx->getParam(0);
  auto s = ctx->getParam<int64_t>(1);
  SPU_ENFORCE(target.shape().size() == 1 && target.shape()[0] == 1,
              "shape of target_point should be {1}");
  SPU_ENFORCE(s > 0, "db_size should greater than 0");

  ctx->pushOutput(proc(ctx, target, s));
}

void OramReadKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& onehot = ctx->getParam(0);
  const auto& db = ctx->getParam(1);
  auto offset = ctx->getParam<int64_t>(2);

  SPU_ENFORCE(onehot.shape().size() == 2 && onehot.shape()[0] == 1,
              "one hot should be of shape {1, db_size}");
  SPU_ENFORCE(db.shape().size() == 2, "database should be 2D");
  SPU_ENFORCE(onehot.shape()[1] == db.shape()[0],
              "onehot and database shape mismatch");

  ctx->pushOutput(proc(ctx, onehot, db, offset));
}

void RingCastKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam(0);
  const auto& to_type = ctx->getParam<SemanticType>(1);

  ctx->pushOutput(proc(ctx, in, to_type));
}

}  // namespace spu::mpc
