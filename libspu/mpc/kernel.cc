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
  const auto& shape = ctx->getParam<Shape>(0);

  auto res = proc(ctx, shape);

  ctx->setOutput(WrapValue(res));
}

void UnaryKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam<Value>(0);

  auto res = proc(ctx, UnwrapValue(in));

  ctx->setOutput(WrapValue(res));
}

void RevealToKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam<Value>(0);
  const auto rank = ctx->getParam<size_t>(1);

  auto res = proc(ctx, UnwrapValue(in), rank);

  ctx->setOutput(WrapValue(res));
}

void ShiftKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam<Value>(0);
  size_t bits = ctx->getParam<size_t>(1);

  auto res = proc(ctx, UnwrapValue(in), bits);

  ctx->setOutput(WrapValue(res));
}

void BinaryKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& lhs = ctx->getParam<Value>(0);
  const auto& rhs = ctx->getParam<Value>(1);

  SPU_ENFORCE(lhs.shape() == rhs.shape(), "shape mismatch {} {}", lhs.shape(),
              rhs.shape());

  auto z = proc(ctx, UnwrapValue(lhs), UnwrapValue(rhs));

  ctx->setOutput(WrapValue(z));
}

void MatmulKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& lhs = ctx->getParam<Value>(0);
  const auto& rhs = ctx->getParam<Value>(1);

  SPU_ENFORCE(lhs.shape()[1] == rhs.shape()[0], "invalid shape {} {}", lhs,
              rhs);

  ctx->setOutput(WrapValue(proc(ctx, lhs.data(), rhs.data())));
}

void Conv2DKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& lhs = ctx->getParam<Value>(0);
  const auto& rhs = ctx->getParam<Value>(1);
  auto stride_h = ctx->getParam<int64_t>(2);
  auto stride_w = ctx->getParam<int64_t>(3);

  auto z = proc(ctx, UnwrapValue(lhs), UnwrapValue(rhs), stride_h, stride_w);

  ctx->setOutput(WrapValue(z));
}

void BitrevKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam<Value>(0);
  size_t start = ctx->getParam<size_t>(1);
  size_t end = ctx->getParam<size_t>(2);

  auto z = proc(ctx, UnwrapValue(in), start, end);

  ctx->setOutput(WrapValue(z));
}

void TruncAKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam<Value>(0);
  size_t bits = ctx->getParam<size_t>(1);
  SignType sign = ctx->getParam<SignType>(2);

  auto z = proc(ctx, UnwrapValue(in), bits, sign);

  ctx->setOutput(WrapValue(z));
}

void BitSplitKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam<Value>(0);
  size_t stride = ctx->getParam<size_t>(1);

  auto res = proc(ctx, UnwrapValue(in), stride);

  ctx->setOutput(WrapValue(res));
}

void CastTypeKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& val = ctx->getParam<Value>(0);
  const auto& to_type = ctx->getParam<Type>(1);

  auto res = proc(ctx, UnwrapValue(val), to_type);

  ctx->setOutput(WrapValue(res));
}

void PermKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& x = ctx->getParam<Value>(0);
  const auto& y = ctx->getParam<Value>(1);

  SPU_ENFORCE(x.shape() == y.shape(), "shape mismatch {} {}", x.shape(),
              x.shape());
  SPU_ENFORCE(x.shape().ndim() == 1, "input should be a 1-d tensor");

  auto z = proc(ctx, UnwrapValue(x), UnwrapValue(y));

  ctx->setOutput(WrapValue(z));
}

void GenInvPermKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam<Value>(0);
  bool is_ascending = ctx->getParam<bool>(1);
  SPU_ENFORCE(in.shape().ndim() == 1, "input should be a 1-d tensor");

  auto y = proc(ctx, UnwrapValue(in), is_ascending);

  ctx->setOutput(WrapValue(y));
}

void MergeKeysKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam<std::vector<Value>>(0);
  bool is_ascending = ctx->getParam<bool>(1);
  std::vector<NdArrayRef> inputs;
  for (size_t i = 0; i < in.size(); ++i) {
    inputs.push_back(UnwrapValue(in[i]));
  }
  auto y = proc(ctx, inputs, is_ascending);

  ctx->setOutput(WrapValue(y));
}

void BroadcastKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam<Value>(0);
  const auto& to_shape = ctx->getParam<Shape>(1);
  const auto& in_dims = ctx->getParam<Axes>(2);

  auto z = proc(ctx, UnwrapValue(in), to_shape, in_dims);

  ctx->setOutput(WrapValue(z));
}

void DimsBasedKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam<Value>(0);
  const auto& axes = ctx->getParam<Axes>(1);

  auto z = proc(ctx, UnwrapValue(in), axes);

  ctx->setOutput(WrapValue(z));
}

void ShapeBasedKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam<Value>(0);
  const auto& to_shape = ctx->getParam<Shape>(1);

  auto z = proc(ctx, UnwrapValue(in), to_shape);

  ctx->setOutput(WrapValue(z));
}

void ExtractSliceKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam<Value>(0);
  const auto& start = ctx->getParam<Index>(1);
  const auto& end = ctx->getParam<Index>(2);
  const auto& strides = ctx->getParam<Strides>(3);

  auto z = proc(ctx, UnwrapValue(in), start, end, strides);

  ctx->setOutput(WrapValue(z));
}

void UpdateSliceKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam<Value>(0);
  const auto& update = ctx->getParam<Value>(1);
  const auto& start = ctx->getParam<Index>(2);

  auto z = proc(ctx, UnwrapValue(in), UnwrapValue(update), start);

  ctx->setOutput(WrapValue(z));
}

void PadKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam<Value>(0);
  const auto& padding_value = ctx->getParam<Value>(1);
  const auto& edge_low = ctx->getParam<Sizes>(2);
  const auto& edge_high = ctx->getParam<Sizes>(3);
  const auto& interior_padding = ctx->getParam<Sizes>(4);

  auto z = proc(ctx, UnwrapValue(in), UnwrapValue(padding_value), edge_low,
                edge_high, interior_padding);

  ctx->setOutput(WrapValue(z));
}

void ConcateKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& ins = ctx->getParam<std::vector<Value>>(0);
  const auto& axis = ctx->getParam<int64_t>(1);

  std::vector<NdArrayRef> unwrapped(ins.size());

  for (size_t idx = 0; idx < ins.size(); ++idx) {
    unwrapped[idx] = UnwrapValue(ins[idx]);
  }

  auto z = proc(ctx, unwrapped, axis);

  ctx->setOutput(WrapValue(z));
}

void OramOneHotKernel::evaluate(KernelEvalContext* ctx) const {
  auto target = ctx->getParam<Value>(0);
  auto s = ctx->getParam<int64_t>(1);
  SPU_ENFORCE(target.shape().size() == 1 && target.shape()[0] == 1,
              "shape of target_point should be {1}");
  SPU_ENFORCE(s > 0, "db_size should greater than 0");

  auto res = proc(ctx, UnwrapValue(target), s);

  ctx->setOutput(WrapValue(res));
}

void OramReadKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& onehot = ctx->getParam<Value>(0);
  const auto& db = ctx->getParam<Value>(1);
  auto offset = ctx->getParam<int64_t>(2);

  SPU_ENFORCE(onehot.shape().size() == 2 && onehot.shape()[0] == 1,
              "one hot should be of shape {1, db_size}");
  SPU_ENFORCE(db.shape().size() == 2, "database should be 2D");
  SPU_ENFORCE(onehot.shape()[1] == db.shape()[0],
              "onehot and database shape mismatch");

  ctx->setOutput(
      WrapValue(proc(ctx, UnwrapValue(onehot), UnwrapValue(db), offset)));
}

}  // namespace spu::mpc
