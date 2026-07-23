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

#include "experimental/squirrel/utils.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/prelude.h"
#include "libspu/core/type_util.h"
#include "libspu/kernel/hlo/basic_binary.h"
#include "libspu/kernel/hlo/basic_ternary.h"
#include "libspu/kernel/hlo/casting.h"
#include "libspu/kernel/hlo/const.h"
#include "libspu/kernel/hlo/geometrical.h"
#include "libspu/kernel/hlo/reduce.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/state.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace squirrel {

spu::Value ReduceSum(spu::SPUContext* ctx, const spu::Value& x, int axis,
                     bool keepdims) {
  namespace skh = spu::kernel::hlo;

  SPU_ENFORCE(axis >= 0 and axis < x.shape().ndim());
  auto init_val = skh::Constant(ctx, {0}, x.shape());
  auto ret = skh::Reduce(
      ctx, {&x, 1}, {&init_val, 1}, spu::Axes{axis},
      [&](absl::Span<spu::Value const> lhs, absl::Span<spu::Value const> rhs) {
        auto sum = skh::Add(ctx, lhs[0], rhs[0]);
        return std::vector<spu::Value>(1, sum);
      });

  if (not keepdims) {
    std::vector<int64_t> ret_shape;
    for (int i = 0; i < x.shape().ndim(); ++i) {
      if (i != axis) {
        ret_shape.push_back(x.shape()[i]);
      }
    }
    ret[0] = skh::Reshape(ctx, ret[0], spu::Shape(ret_shape));
  }

  return ret[0];
}
spu::Value ArgMaxWithValue(spu::SPUContext* ctx, const spu::Value& x, int axis,
                           spu::Value& max) {
  namespace skh = spu::kernel::hlo;
  SPU_ENFORCE(axis >= 0 and axis < x.shape().ndim());

  std::vector<int64_t> _ret_shape;
  for (int i = 0; i < x.shape().ndim(); ++i) {
    if (i != axis) {
      _ret_shape.push_back(x.shape()[i]);
    }
  }
  spu::Shape ret_shape(_ret_shape);
  auto index = skh::Broadcast(ctx, skh::Iota(ctx, spu::DT_I32, x.shape()[axis]),
                              x.shape(), spu::Axes{axis});
  index = skh::Seal(ctx, index);

  auto ret = spu::kernel::hlo::Reduce(
      ctx, {x, index}, {spu::Value(), spu::Value()}, spu::Axes{axis},
      [&](absl::Span<spu::Value const> lhs, absl::Span<spu::Value const> rhs) {
        // NOTE(lwj): we skip NaN check and return the 1st operand if the two
        // operands are identical
        auto lt = skh::Less(ctx, lhs[0], rhs[0]);
        // TODO(lwj): should we insert the _prefer_a for ABY3 here?
        auto max_val = skh::Select(ctx, lt, rhs[0], lhs[0]);
        auto max_idx = skh::Select(ctx, lt, rhs[1], lhs[1]);
        return std::vector<spu::Value>{max_val, max_idx};
      },
      /*ignore_init*/ true);

  ret[1] = skh::Reshape(ctx, ret[1], ret_shape);

  max = ret[0];
  return ret[1];
}

spu::Value ArgMax(spu::SPUContext* ctx, const spu::Value& x, int axis,
                  bool keepdims) {
  namespace skh = spu::kernel::hlo;
  SPU_ENFORCE(axis >= 0 and axis < x.shape().ndim());

  std::vector<int64_t> _ret_shape;
  for (int i = 0; i < x.shape().ndim(); ++i) {
    if (i != axis) {
      _ret_shape.push_back(x.shape()[i]);
    }
  }
  spu::Shape ret_shape(_ret_shape);
  auto index = skh::Broadcast(ctx, skh::Iota(ctx, spu::DT_I32, x.shape()[axis]),
                              x.shape(), spu::Axes{axis});
  index = skh::Seal(ctx, index);

  auto ret = spu::kernel::hlo::Reduce(
      ctx, {x, index}, {spu::Value(), spu::Value()}, spu::Axes{axis},
      [&](absl::Span<spu::Value const> lhs, absl::Span<spu::Value const> rhs) {
        // NOTE(lwj): we skip NaN check and return the 1st operand if the two
        // operands are identical
        auto lt = skh::Less(ctx, lhs[0], rhs[0]);
        // TODO(lwj): should we insert the _prefer_a for ABY3 here?
        auto max_val = skh::Select(ctx, lt, rhs[0], lhs[0]);
        auto max_idx = skh::Select(ctx, lt, rhs[1], lhs[1]);
        return std::vector<spu::Value>{max_val, max_idx};
      },
      /*ignore_init*/ true);

  if (not keepdims) {
    ret[1] = skh::Reshape(ctx, ret[1], ret_shape);
  }
  return ret[1];
}

spu::Value MulArithShareWithPrivateBoolean(spu::SPUContext* ctx,
                                           const spu::Value& ashr) {
  SPU_ENFORCE(ctx->config().protocol() == spu::ProtocolKind::CHEETAH);
  SPU_ENFORCE(ashr.isSecret());

  spu::KernelEvalContext kctx(ctx);
  auto out = spu::mpc::cheetah::TiledDispatchOTFunc(
      &kctx, ashr.data(),
      [&](const spu::NdArrayRef& input,
          const std::shared_ptr<spu::mpc::cheetah::BasicOTProtocols>& base_ot) {
        return base_ot->PrivateMulxSend(input);
      });
  return spu::Value(out, ashr.dtype());
}

spu::Value MulArithShareWithPrivateBoolean(
    spu::SPUContext* ctx, const spu::Value& ashr,
    absl::Span<const uint8_t> prv_boolean) {
  SPU_ENFORCE(ctx->config().protocol() == spu::ProtocolKind::CHEETAH);
  SPU_ENFORCE(ashr.isSecret());
  SPU_ENFORCE_EQ(ashr.numel(), (int64_t)prv_boolean.size());

  spu::KernelEvalContext kctx(ctx);
  auto out = spu::mpc::cheetah::TiledDispatchOTFunc(
      &kctx, ashr.data(), prv_boolean,
      [&](const spu::NdArrayRef& input, absl::Span<const uint8_t> choices,
          const std::shared_ptr<spu::mpc::cheetah::BasicOTProtocols>& base_ot) {
        return base_ot->PrivateMulxRecv(input, choices);
      });
  return spu::Value(out, ashr.dtype());
}

spu::Value MulPrivateArithWithPrivateBoolean(spu::SPUContext* ctx,
                                             const spu::Value& arith) {
  using namespace spu;
  SPU_ENFORCE(ctx->config().protocol() == spu::ProtocolKind::CHEETAH);
  spu::KernelEvalContext kctx(ctx);
  auto ft = ctx->config().field();
  auto out = mpc::cheetah::TiledDispatchOTFunc(
      &kctx, arith.data(),
      [&](const NdArrayRef& input,
          const std::shared_ptr<spu::mpc::cheetah::BasicOTProtocols>& base_ot) {
        return DISPATCH_ALL_FIELDS(ft, [&]() {
          NdArrayRef ot_out = spu::mpc::ring_zeros(ft, input.shape());
          auto inp = absl::MakeConstSpan(&input.at<ring2k_t>(0), input.numel());
          auto oup = absl::MakeSpan(&ot_out.at<ring2k_t>(0), ot_out.numel());
          base_ot->GetSenderCOT()->SendCAMCC(inp, oup);
          return ot_out;
        });
      });

  return spu::Value(out, arith.dtype());
}

spu::Value MulPrivateArithWithPrivateBoolean(spu::SPUContext* ctx,
                                             absl::Span<const uint8_t> boolean,
                                             const spu::DataType dtype,
                                             const spu::Shape& shape) {
  using namespace spu;
  SPU_ENFORCE(ctx->config().protocol() == spu::ProtocolKind::CHEETAH);
  SPU_ENFORCE_EQ(boolean.size(), (size_t)shape.numel());

  spu::KernelEvalContext kctx(ctx);
  auto ft = ctx->config().field();
  auto out = mpc::cheetah::TiledDispatchOTFunc(
      &kctx, boolean,
      [&](absl::Span<const uint8_t> input,
          const std::shared_ptr<spu::mpc::cheetah::BasicOTProtocols>& base_ot) {
        return DISPATCH_ALL_FIELDS(ft, [&]() {
          NdArrayRef ot_out = spu::mpc::ring_zeros(ft, {(int64_t)input.size()});
          auto oup = absl::MakeSpan(&ot_out.at<ring2k_t>(0), input.size());
          base_ot->GetReceiverCOT()->RecvCAMCC(input, oup);
          return ot_out;
        });
      });
  return spu::Value(out, dtype);
}

spu::Value MulArithShareWithANDBoolShare(spu::SPUContext* ctx,
                                         const spu::Value& ashr,
                                         absl::Span<const uint8_t> bshr) {
  using namespace spu;
  SPU_ENFORCE(ashr.isSecret());
  SPU_ENFORCE_EQ(ashr.numel(), (int64_t)bshr.size());

  SPU_ENFORCE(ctx->config().protocol() == spu::ProtocolKind::CHEETAH);

  spu::KernelEvalContext kctx(ctx);
  auto ft = ctx->config().field();
  int rank = ctx->lctx()->Rank();

  auto out = mpc::cheetah::TiledDispatchOTFunc(
      &kctx, ashr.data(), bshr,
      [&](const NdArrayRef& x, absl::Span<const uint8_t> y,
          std::shared_ptr<spu::mpc::cheetah::BasicOTProtocols> base_ot) {
        NdArrayRef out(x.eltype(), x.shape());

        DISPATCH_ALL_FIELDS(ft, [&]() {
          spu::NdArrayView<const ring2k_t> _ashr(x);
          auto oup = absl::MakeSpan(&out.at<ring2k_t>(0), y.size());
          std::vector<ring2k_t> corr(y.size());
          for (size_t i = 0; i < y.size(); ++i) {
            corr[i] = _ashr[i] * static_cast<ring2k_t>(y[i] & 1);
          }
          std::vector<ring2k_t> temp(y.size());

          if (rank == 0) {
            // Correlation x0*b0 on choice b1
            base_ot->GetSenderCOT()->SendCAMCC(absl::MakeConstSpan(corr),
                                               absl::MakeSpan(temp));
            base_ot->GetSenderCOT()->Flush();
            base_ot->GetReceiverCOT()->RecvCAMCC(y, oup);
          } else {
            base_ot->GetReceiverCOT()->RecvCAMCC(y, oup);
            // Correlation x1*b1 on choice b0
            base_ot->GetSenderCOT()->SendCAMCC(absl::MakeConstSpan(corr),
                                               absl::MakeSpan(temp));
            base_ot->GetSenderCOT()->Flush();
          }

          std::transform(oup.begin(), oup.end(), temp.data(), oup.begin(),
                         std::minus());
        });
        return out;
      });

  return spu::Value(out, ashr.dtype());
}

// a0 * b[0:batch_size]
// a1 * b[batch_size:batch_size*2]
// ...
// an * b[batch_size*(n-1):batch_size*n]
spu::Value BatchMulArithShareWithANDBoolShare(spu::SPUContext* ctx,
                                              const spu::Value& ashr,
                                              size_t batch_size,
                                              absl::Span<const uint8_t> bshr) {
  SPU_ENFORCE(ashr.isSecret());
  SPU_ENFORCE(batch_size > 0);
  SPU_ENFORCE_EQ(batch_size * static_cast<size_t>(ashr.numel()), bshr.size());

  auto flatten = spu::kernel::hlo::Reshape(ctx, ashr, {ashr.numel()});
  std::vector<spu::Value> batched(ashr.numel());
  for (int64_t i = 0; i < ashr.numel(); ++i) {
    auto ai = spu::kernel::hlo::Broadcast(
        ctx, spu::kernel::hlo::Slice(ctx, flatten, {i}, {i + 1}, {1}),
        {static_cast<int64_t>(batch_size)}, {});
    batched[i] = MulArithShareWithANDBoolShare(
        ctx, ai, bshr.subspan(i * batch_size, batch_size));
  }

  return spu::kernel::hlo::Concatenate(ctx, batched, 0);
}

}  // namespace squirrel
