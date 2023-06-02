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

  ArrayRef res = proc(ctx, shape.numel());

  ctx->setOutput(WrapValue(res, shape));
}

void UnaryKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam<Value>(0);
  auto [arr, shape, dtype] = UnwrapValue(in);

  ArrayRef res = proc(ctx, arr);

  ctx->setOutput(WrapValue(res, shape));
}

void RevealToKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam<Value>(0);
  const auto rank = ctx->getParam<size_t>(1);
  auto [arr, shape, dtype] = UnwrapValue(in);

  ArrayRef res = proc(ctx, arr, rank);

  ctx->setOutput(WrapValue(res, shape));
}

void ShiftKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam<Value>(0);
  size_t bits = ctx->getParam<size_t>(1);
  auto [arr, shape, dtype] = UnwrapValue(in);

  ArrayRef res = proc(ctx, arr, bits);

  ctx->setOutput(WrapValue(res, shape));
}

void BinaryKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& lhs = ctx->getParam<Value>(0);
  const auto& rhs = ctx->getParam<Value>(1);

  SPU_ENFORCE(lhs.shape() == rhs.shape(), "shape mismatch {} {}", lhs.shape(),
              rhs.shape());

  auto [x, shape, dtype] = UnwrapValue(lhs);
  auto [y, _, _1] = UnwrapValue(rhs);

  auto z = proc(ctx, x, y);

  ctx->setOutput(WrapValue(z, shape));
}

void MatmulKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& lhs = ctx->getParam<Value>(0);
  const auto& rhs = ctx->getParam<Value>(1);

  // TODO: drop (m, n, k)
  auto m = static_cast<int64_t>(ctx->getParam<size_t>(2));
  auto n = static_cast<int64_t>(ctx->getParam<size_t>(3));
  auto k = static_cast<int64_t>(ctx->getParam<size_t>(4));

  // SPU_ENFORCE(lhs.shape().size() == 2 && rhs.shape().size() == 2 &&
  //                 lhs.shape()[0] == m && lhs.shape()[1] == k &&
  //                 rhs.shape()[0] == k && rhs.shape()[1] == n,
  //             "invalid shape {} {}", lhs.shape(), rhs.shape());
  SPU_ENFORCE(
      calcNumel(lhs.shape()) == m * k && calcNumel(rhs.shape()) == k * n,
      "invalid shape {} {}", lhs.shape(), rhs.shape());

  auto [x, shape, dtype] = UnwrapValue(lhs);
  auto [y, _, _1] = UnwrapValue(rhs);

  auto z = proc(ctx, x, y, m, n, k);

  ctx->setOutput(WrapValue(z, Shape{m, n}));
}

void Conv2DKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& lhs = ctx->getParam<Value>(0);
  const auto& rhs = ctx->getParam<Value>(1);
  size_t N = ctx->getParam<size_t>(2);
  size_t H = ctx->getParam<size_t>(3);
  size_t W = ctx->getParam<size_t>(4);
  size_t C = ctx->getParam<size_t>(5);
  size_t O = ctx->getParam<size_t>(6);
  size_t h = ctx->getParam<size_t>(7);
  size_t w = ctx->getParam<size_t>(8);
  size_t stride_h = ctx->getParam<size_t>(9);
  size_t stride_w = ctx->getParam<size_t>(10);

  auto [x, shape, dtype] = UnwrapValue(lhs);
  auto [y, _, _1] = UnwrapValue(rhs);

  auto z = proc(ctx, x, y, N, H, W, C, O, h, w, stride_h, stride_w);

  ctx->setOutput(WrapValue(z, shape));
}

void BitrevKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam<Value>(0);
  size_t start = ctx->getParam<size_t>(1);
  size_t end = ctx->getParam<size_t>(2);

  auto [x, shape, dtype] = UnwrapValue(in);

  auto z = proc(ctx, x, start, end);

  ctx->setOutput(WrapValue(z, shape));
}

void TruncAWithSignKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam<Value>(0);
  size_t bits = ctx->getParam<size_t>(1);
  bool positive = ctx->getParam<bool>(2);

  auto [x, shape, dtype] = UnwrapValue(in);
  auto z = proc(ctx, x, bits, positive);

  ctx->setOutput(WrapValue(z, shape));
}

void BitSplitKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam<Value>(0);
  size_t stride = ctx->getParam<size_t>(1);
  auto [arr, shape, dtype] = UnwrapValue(in);

  ArrayRef res = proc(ctx, arr, stride);

  ctx->setOutput(WrapValue(res, shape));
}

void CastTypeKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& val = ctx->getParam<Value>(0);
  const auto& to_type = ctx->getParam<Type>(1);

  auto [arr, shape, dtype] = UnwrapValue(val);

  ArrayRef res = proc(ctx, arr, to_type);

  ctx->setOutput(WrapValue(res, shape));
}

}  // namespace spu::mpc
