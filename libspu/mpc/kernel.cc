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

void SimpleSortKernel::evaluate(KernelEvalContext* ctx) const {
  auto values = ctx->getParam<std::vector<Value>>(0);
  std::vector<NdArrayRef> inputs;

  for (const auto& val : values) {
    inputs.emplace_back(UnwrapValue(val));
  }

  auto res = proc(ctx, inputs);

  std::vector<Value> res_values;

  for (size_t i = 0; i < res.size(); ++i) {
    res_values.emplace_back(WrapValue(std::move(res[i])));
  }
  ctx->setOutput(res_values);
}

}  // namespace spu::mpc
