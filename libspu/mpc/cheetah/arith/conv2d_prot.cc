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
//
#include "libspu/mpc/cheetah/arith/conv2d_prot.h"

#include "seal/evaluator.h"

#include "libspu/core/shape_util.h"
#include "libspu/core/xt_helper.h"
#include "libspu/mpc/cheetah/arith/conv2d_helper.h"
#include "libspu/mpc/cheetah/arith/tensor_encoder.h"
#include "libspu/mpc/cheetah/rlwe/utils.h"
#include "libspu/mpc/utils/ring_ops.h"

template <>
struct std::hash<spu::mpc::cheetah::Conv2DProtocol::Meta> {
  size_t operator()(
      const spu::mpc::cheetah::Conv2DProtocol::Meta &s) const noexcept {
    // FIXME(juhou): use a better way for hash
    using namespace spu::mpc::cheetah;
    size_t h = std::hash<std::string>()("Conv2DProtocol::Meta");
    h = (h << 1) ^ std::hash<int64_t>()(s.input_batch);
    h = (h << 1) ^ std::hash<int64_t>()(s.num_kernels);
    for (int i : {0, 1, 2}) {
      h = (h << 1) ^ std::hash<int64_t>()(s.input_shape[i]);
      h = (h << 1) ^ std::hash<int64_t>()(s.kernel_shape[i]);
    }
    h = (h << 1) ^ std::hash<int64_t>()(s.window_strides[0]);
    h = (h << 1) ^ std::hash<int64_t>()(s.window_strides[1]);
    return h;
  }
};

namespace spu::mpc::cheetah {
// Layout:
//   NxHxWxC for input
//   HxWxIxO for kernel
[[maybe_unused]] constexpr int kH = 0;
[[maybe_unused]] constexpr int kW = 1;
[[maybe_unused]] constexpr int kC = 2;
[[maybe_unused]] constexpr int kO = 3;

bool IsSameInputShape(const ArrayRef &base, const Shape3D &shape) {
  return base.numel() == calcNumel(shape);
}

bool IsSameKernelShape(const ArrayRef &base, const Shape3D &shape,
                       int64_t num_kernels) {
  return base.numel() == (num_kernels * calcNumel(shape));
}

bool operator==(const Conv2DProtocol::Meta &x, const Conv2DProtocol::Meta &y) {
  return 0 == std::memcmp(&x, &y, sizeof(Conv2DProtocol::Meta));
}

bool Conv2DProtocol::IsValidSubShape(const Shape3D &shape) const {
  int64_t n = calcNumel(absl::MakeSpan(shape));
  return (n > 0 && n <= poly_deg_);
}

size_t Conv2DProtocol::GetKernelSize(const Meta &meta,
                                     const Shape3D &subshape) const {
  SPU_ENFORCE(IsValidMeta(meta));
  SPU_ENFORCE(IsValidSubShape(subshape));
  return meta.num_kernels * CeilDiv(meta.kernel_shape[kC], subshape[kC]);
}

size_t Conv2DProtocol::GetInputSize(const Meta &meta,
                                    const Shape3D &subshape) const {
  SPU_ENFORCE(IsValidMeta(meta));
  SPU_ENFORCE(IsValidSubShape(subshape));
  Conv2DHelper helper(meta, subshape);
  return static_cast<size_t>(helper.num_slices());
}

size_t Conv2DProtocol::GetOutSize(const Meta &meta,
                                  const Shape3D &subshape) const {
  SPU_ENFORCE(IsValidMeta(meta));
  SPU_ENFORCE(IsValidSubShape(subshape));
  // input channel sums up. each output channel will increase the output size
  size_t wc = CeilDiv(meta.input_shape[kC], subshape[kC]);
  return GetInputSize(meta) / wc * meta.num_kernels;
}

Shape3D Conv2DProtocol::GetSubTensorShape(const Meta &meta) const {
  static std::shared_mutex lock_;
  static std::unordered_map<Meta, Shape3D> memo_;

  {
    std::shared_lock<std::shared_mutex> guard(lock_);
    auto val = memo_.find(meta);
    if (val != memo_.end()) return val->second;
  }
  std::unique_lock<std::shared_mutex> guard(lock_);
  auto val = memo_.find(meta);
  if (val != memo_.end()) return val->second;

  int64_t Cw = poly_deg_ / (meta.kernel_shape[kH] * meta.kernel_shape[kW]);
  SPU_ENFORCE_EQ(meta.input_shape[kC], meta.kernel_shape[kC],
                 "channel mismatch");
  SPU_ENFORCE(Cw > 0, "kernel size out-of-bound poly_deg={}", poly_deg_);

  Shape3D input_shape = meta.input_shape;
  Shape3D subshape = input_shape;
  const int64_t H = input_shape[kH];
  const int64_t W = input_shape[kW];
  const int64_t C = input_shape[kC];
  const int64_t HW = H * W;
  if (HW <= poly_deg_) {
    // H * W <= N
    subshape[kC] = std::min(Cw, std::min(poly_deg_ / HW, input_shape[kC]));
  } else {
    int64_t min_cost = std::numeric_limits<int64_t>::max();

    for (int64_t c = 1; c <= std::min(Cw, C); ++c) {
      for (int64_t h = 1; h <= H; ++h) {
        for (int64_t w = 1; w <= W; ++w) {
          if (c * h * w > poly_deg_) {
            continue;
          }

          int64_t encode_input = CeilDiv(input_shape[kH], h) *
                                 CeilDiv(input_shape[kW], w) * meta.input_batch;

          int64_t cost = CeilDiv(input_shape[kC], c) * encode_input;

          if (cost < min_cost) {
            subshape[kC] = c;
            subshape[kH] = h;
            subshape[kW] = w;
            min_cost = cost;
          }
        }
      }
    }
  }

  SPU_ENFORCE(
      calcNumel(subshape) <= poly_deg_,
      fmt::format("subshape {}x{}x{}", subshape[0], subshape[1], subshape[2]));

  memo_.insert({meta, subshape});
  return subshape;
}

Conv2DProtocol::Conv2DProtocol(const seal::SEALContext &context,
                               const ModulusSwitchHelper &ms_helper)
    : context_(context) {
  SPU_ENFORCE(context_.parameters_set());
  poly_deg_ = context.key_context_data()->parms().poly_modulus_degree();
  tencoder_ = std::make_unique<TensorEncoder>(context, ms_helper);
}

bool Conv2DProtocol::IsValidMeta(const Meta &meta) const {
  // santi check
  if (meta.input_shape[kC] != meta.kernel_shape[kC]) return false;
  for (size_t d = 0; d < 3; ++d) {
    if (meta.input_shape[d] < 0 || meta.kernel_shape[d] < 0) return false;
  }
  if (meta.num_kernels == 0) return false;
  if (meta.window_strides[0] < 0 || meta.window_strides[1] < 0) return false;

  // required by current implementation
  if (meta.kernel_shape[kH] * meta.kernel_shape[kW] > poly_deg_) {
    return false;
  }

  return true;
}

void Conv2DProtocol::EncodeInput(const ArrayRef &input, const Meta &meta,
                                 bool need_encrypt,
                                 absl::Span<RLWEPt> out) const {
  SPU_ENFORCE(IsSameInputShape(input, meta.input_shape));
  SPU_ENFORCE_EQ(out.size(), GetInputSize(meta));

  Shape3D subinput_shape = GetSubTensorShape(meta);

  Shape3D subkernel_shape = meta.kernel_shape;
  subkernel_shape[kC] = subinput_shape[kC];

  Conv2DHelper helper(meta, subinput_shape);
  int64_t poly_per_channel = helper.slice_size(kH) * helper.slice_size(kW);

  for (size_t i = 0; i < out.size(); ++i) {
    int64_t c = i / poly_per_channel;
    int64_t h = (i % poly_per_channel) / helper.slice_size(kW);
    int64_t w = i % helper.slice_size(kW);
    auto subinput = helper.Slice(input, meta.input_shape, {h, w, c});
    subinput.ZeroPadAs(subinput_shape);
    tencoder_->EncodeInput(subinput, subkernel_shape, need_encrypt, &out[i]);
  }
}

void Conv2DProtocol::EncodeKernels(const ArrayRef &kernels, const Meta &meta,
                                   bool need_encrypt,
                                   absl::Span<RLWEPt> out) const {
  SPU_ENFORCE(IsSameKernelShape(kernels, meta.kernel_shape, meta.num_kernels));
  SPU_ENFORCE_EQ(out.size(), GetKernelSize(meta));
  const int64_t kernel_sze = calcNumel(meta.kernel_shape);
  const size_t num_poly_per_kernel = out.size() / meta.num_kernels;
  const int64_t stride = meta.num_kernels;
  // H x W x C x O for kernel
  // slice on each O-channel
  for (int64_t m = 0; m < meta.num_kernels; ++m) {
    auto kernel = kernels.slice(m, m + kernel_sze * stride, stride);
    absl::Span<RLWEPt> dst = {out.data() + m * num_poly_per_kernel,
                              num_poly_per_kernel};
    EncodeSingleKernel(kernel, meta, need_encrypt, dst);
  }
}

void Conv2DProtocol::EncodeSingleKernel(const ArrayRef &kernel,
                                        const Meta &meta, bool need_encrypt,
                                        absl::Span<RLWEPt> out) const {
  SPU_ENFORCE_EQ(kernel.numel(), calcNumel(meta.kernel_shape));
  Shape3D subinput_shape = GetSubTensorShape(meta);
  Shape3D subkernel_shape = meta.kernel_shape;
  subkernel_shape[kC] = subinput_shape[kC];

  Meta kmeta;
  kmeta.input_shape = meta.kernel_shape;
  kmeta.kernel_shape = meta.kernel_shape;
  kmeta.window_strides = {meta.kernel_shape[0], meta.kernel_shape[1]};
  kmeta.num_kernels = 1;

  Conv2DHelper helper(kmeta, subkernel_shape);
  int64_t num_poly = helper.num_slices();
  int64_t poly_per_channel = helper.slice_size(kH) * helper.slice_size(kW);
  SPU_ENFORCE_EQ(static_cast<int64_t>(out.size()), num_poly);

  for (int64_t i = 0; i < num_poly; ++i) {
    int64_t c = i / poly_per_channel;
    int64_t h = (i % poly_per_channel) / helper.slice_size(kW);
    int64_t w = i % helper.slice_size(kW);
    auto subkernel = helper.Slice(kernel, meta.kernel_shape, {h, w, c});
    subkernel.ZeroPadAs(subkernel_shape);
    tencoder_->EncodeKernel(subkernel, subinput_shape, need_encrypt, &out[i]);
  }
}

inline absl::Span<const uint64_t> MakeSpan(const RLWEPt &pt) {
  return {pt.data(), pt.coeff_count()};
}

#define DEF_COMPUTE(I, K, O)                                                 \
  void Conv2DProtocol::Compute(absl::Span<const I> input,                    \
                               absl::Span<const K> kernel, const Meta &meta, \
                               absl::Span<O> out) const {                    \
    size_t input_poly = GetInputSize(meta);                                  \
    size_t kernel_poly = GetKernelSize(meta);                                \
    size_t out_size = GetOutSize(meta);                                      \
    SPU_ENFORCE_EQ(input.size(), input_poly);                                \
    SPU_ENFORCE_EQ(kernel.size(), kernel_poly);                              \
    SPU_ENFORCE_EQ(out.size(), out_size);                                    \
    size_t num_poly_per_I = kernel_poly / meta.num_kernels;                  \
    size_t num_poly_per_O = out_size / meta.num_kernels;                     \
    for (int64_t m = 0; m < meta.num_kernels; ++m) {                         \
      ComputeOneKernel<I, K, O>(                                             \
          input, {kernel.data() + m * num_poly_per_I, num_poly_per_I},       \
          {out.data() + m * num_poly_per_O, num_poly_per_O});                \
    }                                                                        \
  }

DEF_COMPUTE(RLWEPt, RLWEPt, RLWEPt)
DEF_COMPUTE(RLWECt, RLWEPt, RLWECt)
DEF_COMPUTE(RLWEPt, RLWECt, RLWECt)

template <>
void Conv2DProtocol::FusedMulAddInplace(RLWECt &acc, const RLWECt &lhs,
                                        const RLWEPt &rhs) const {
  SPU_ENFORCE(lhs.parms_id() == rhs.parms_id());
  auto cntxt_data = context_.get_context_data(lhs.parms_id());
  SPU_ENFORCE(cntxt_data != nullptr);

  if (acc.size() == 0) {
    acc.resize(context_, lhs.parms_id(), lhs.size());
    acc.is_ntt_form() = lhs.is_ntt_form();
  } else {
    SPU_ENFORCE_EQ(acc.size(), lhs.size());
    SPU_ENFORCE(acc.parms_id() == lhs.parms_id());
    SPU_ENFORCE(acc.is_ntt_form() && lhs.is_ntt_form());
  }

  auto parms = cntxt_data->parms();
  size_t coeff_count = parms.poly_modulus_degree();
  const auto &modulus = parms.coeff_modulus();

  for (size_t k = 0; k < lhs.size(); ++k) {
    using namespace seal::util;
    const auto *op0 = lhs.data(k);
    const auto *op1 = rhs.data();
    auto *dst = acc.data(k);
    for (const auto &prime : modulus) {
      for (size_t i = 0; i < coeff_count; ++i, ++dst) {
        *dst = multiply_add_uint_mod(*op0++, *op1++, *dst, prime);
      }
    }
  }
}

template <>
void Conv2DProtocol::FusedMulAddInplace(RLWECt &acc, const RLWEPt &lhs,
                                        const RLWECt &rhs) const {
  FusedMulAddInplace<RLWECt, RLWECt, RLWEPt>(acc, rhs, lhs);
}

template <>
void Conv2DProtocol::FusedMulAddInplace(RLWEPt &acc, const RLWEPt &lhs,
                                        const RLWEPt &rhs) const {
  SPU_ENFORCE(lhs.parms_id() == rhs.parms_id());
  SPU_ENFORCE(lhs.coeff_count() == rhs.coeff_count());
  auto cntxt = context_.get_context_data(lhs.parms_id());
  SPU_ENFORCE(cntxt != nullptr);

  if (acc.coeff_count() == 0) {
    // acc += lhs * rhs
    acc.parms_id() = seal::parms_id_zero;
    acc.resize(lhs.coeff_count());
    acc.parms_id() = lhs.parms_id();
  }

  size_t coeff_count = cntxt->parms().poly_modulus_degree();
  const auto &modulus = cntxt->parms().coeff_modulus();
  const auto *op0 = lhs.data();
  const auto *op1 = rhs.data();
  auto *dst = acc.data();
  for (const auto &prime : modulus) {
    using namespace seal::util;
    for (size_t i = 0; i < coeff_count; ++i, ++dst) {
      *dst = multiply_add_uint_mod(*op0++, *op1++, *dst, prime);
    }
  }
}

template <typename I, typename K, typename O>
void Conv2DProtocol::ComputeOneKernel(absl::Span<const I> input,
                                      absl::Span<const K> kernel,
                                      absl::Span<O> out) const {
  size_t out_size = input.size() / kernel.size();
  SPU_ENFORCE_EQ(out.size(), out_size);
  for (size_t m = 0; m < out_size; ++m) {
    out[m].release();
  }

  for (size_t c = 0; c < kernel.size(); ++c) {
    for (size_t m = 0; m < out_size; ++m) {
      size_t j = c * out_size + m;
      size_t o = (j % out_size);
      // out[o] += input[j] * kernel[c]
      FusedMulAddInplace<O, I, K>(out[o], input[j], kernel[c]);
    }
  }
}

void Conv2DProtocol::ExtractLWEsInplace(const Meta &meta,
                                        absl::Span<RLWECt> rlwe) const {
  auto subshape = GetSubTensorShape(meta);
  SPU_ENFORCE_EQ(rlwe.size(), GetOutSize(meta, subshape));
  Conv2DHelper helper(meta, subshape);
  size_t poly_per_channel = helper.slice_size(kH) * helper.slice_size(kW);

  seal::Evaluator evaluator(context_);
  for (int64_t m = 0; m < meta.num_kernels; ++m) {
    size_t poly_idx = m * poly_per_channel;
    std::vector<size_t> coefficients;
    for (int64_t h = 0; h < helper.slice_size(kH); ++h) {
      for (int64_t w = 0; w < helper.slice_size(kW); ++w) {
        helper.GetResultCoefficients({h, w, 0}, &coefficients);

        if (rlwe.at(poly_idx).is_ntt_form()) {
          evaluator.transform_from_ntt_inplace(rlwe[poly_idx]);
        }

        std::set<size_t> to_keep(coefficients.begin(), coefficients.end());
        KeepCoefficientsInplace(rlwe[poly_idx++], to_keep);
      }
    }
  }
}

ArrayRef Conv2DProtocol::ParseResult(FieldType field, const Meta &meta,
                                     absl::Span<const RLWEPt> rlwe) const {
  return ParseResult(field, meta, rlwe, tencoder_->ms_helper());
}

ArrayRef Conv2DProtocol::ParseResult(
    FieldType field, const Meta &meta, absl::Span<const RLWEPt> rlwe,
    const ModulusSwitchHelper &ms_helper) const {
  SPU_ENFORCE_EQ(rlwe.size(), GetOutSize(meta));
  size_t expected_n = poly_deg_ * ms_helper.coeff_modulus_size();
  SPU_ENFORCE(std::all_of(rlwe.data(), rlwe.data() + rlwe.size(),
                          [expected_n](const RLWEPt &pt) {
                            return pt.coeff_count() == expected_n;
                          }),
              "invalid RLWE to parse");

  Shape3D oshape;
  oshape[kC] = meta.num_kernels;
  for (int d : {kH, kW}) {
    oshape[d] =
        (meta.input_shape[d] - meta.kernel_shape[d] + meta.window_strides[d]) /
        meta.window_strides[d];
  }

  Conv2DHelper helper(meta, GetSubTensorShape(meta));
  size_t poly_per_channel = helper.slice_size(kH) * helper.slice_size(kW);

  NdArrayRef computed_tensor(makeType<RingTy>(field), oshape);
  for (int64_t m = 0; m < meta.num_kernels; ++m) {
    size_t poly_idx = m * poly_per_channel;

    Shape3D slice_shape;
    std::vector<size_t> coefficients;
    for (int64_t h = 0, hoffset = 0; h < helper.slice_size(kH); ++h) {
      for (int64_t w = 0, woffset = 0; w < helper.slice_size(kW); ++w) {
        helper.GetResultCoefficients({h, w, 0}, &coefficients, &slice_shape);
        SPU_ENFORCE_EQ(static_cast<int64_t>(coefficients.size()),
                       calcNumel(slice_shape));

        auto computed =
            ms_helper.ModulusDownRNS(field, MakeSpan(rlwe[poly_idx++]));
        auto *coeff_iter = coefficients.data();

        DISPATCH_ALL_FIELDS(field, "ParseResult", [&]() {
          auto xcomputed = xt_adapt<ring2k_t>(computed);
          auto xc = xt_mutable_adapt<ring2k_t>(computed_tensor);
          for (int64_t hh = 0; hh < slice_shape[kH]; ++hh) {
            for (int64_t ww = 0; ww < slice_shape[kW]; ++ww) {
              xc(hh + hoffset, ww + woffset, m) = xcomputed[*coeff_iter++];
            }
          }
        });
        woffset += slice_shape[kW];
      }
      hoffset += slice_shape[kH];
    }
  }

  return flatten(computed_tensor);
}

}  // namespace spu::mpc::cheetah
