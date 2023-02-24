// Copyright 2022 Ant Group Co., Ltd.
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

#include "libspu/mpc/cheetah/arith/tensor_encoder.h"

#include "libspu/core/xt_helper.h"
#include "libspu/mpc/cheetah/arith/conv2d_helper.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {

constexpr int kH = 0;
constexpr int kW = 1;
constexpr int kC = 2;

TensorEncoder::TensorEncoder(const seal::SEALContext &context,
                             const ModulusSwitchHelper &ms_helper)
    : ms_helper_(ms_helper) {
  SPU_ENFORCE(context.parameters_set());
  auto pid0 = context.first_parms_id();
  auto pid1 = ms_helper.parms_id();
  SPU_ENFORCE_EQ(0, std::memcmp(&pid0, &pid1, sizeof(seal::parms_id_type)),
                 fmt::format("parameter set mismatch"));
  poly_deg_ = context.first_context_data()->parms().poly_modulus_degree();
}

TensorEncoder::~TensorEncoder() = default;

void TensorEncoder::EncodeInput(const Sliced3DTensor &input,
                                const Shape3D &kernel_shape, bool need_encrypt,
                                RLWEPt *out) const {
  yacl::CheckNotNull(out);

  Shape3D input_shape = input.shape();
  SPU_ENFORCE_EQ(input_shape[kC], kernel_shape[kC], "channel mismatch");
  SPU_ENFORCE(poly_deg_ >= calcNumel(input.shape()));

  InputIndexer indexer(input_shape, kernel_shape);
  auto poly = Tensor2Poly(input_shape, kernel_shape, input, indexer);

  size_t num_modulus = ms_helper_.coeff_modulus_size();
  out->parms_id() = seal::parms_id_zero;
  out->resize(
      seal::util::mul_safe(static_cast<size_t>(poly_deg_), num_modulus));

  uint64_t *dst = out->data();
  for (size_t mod_idx = 0; mod_idx < num_modulus; ++mod_idx) {
    std::fill_n(dst, poly_deg_, 0);
    absl::Span<uint64_t> dst_wrap(dst, poly_deg_);
    if (need_encrypt) {
      ms_helper_.ModulusUpAt(poly, mod_idx, dst_wrap);
    } else {
      ms_helper_.CenteralizeAt(poly, mod_idx, dst_wrap);
    }
    dst += poly_deg_;
  }

  out->parms_id() = ms_helper_.parms_id();
  out->scale() = 1.;
}

void TensorEncoder::EncodeKernel(const Sliced3DTensor &kernel,
                                 const Shape3D &input_shape, bool need_encrypt,
                                 RLWEPt *out) const {
  yacl::CheckNotNull(out);

  Shape3D kernel_shape = kernel.shape();
  SPU_ENFORCE_EQ(input_shape[kC], kernel_shape[kC], "channel mismatch");
  SPU_ENFORCE(poly_deg_ >= calcNumel(kernel.shape()));

  KernelIndexer indexer(input_shape, kernel_shape);
  auto poly = Tensor2Poly(input_shape, kernel_shape, kernel, indexer);

  size_t num_modulus = ms_helper_.coeff_modulus_size();
  out->parms_id() = seal::parms_id_zero;
  out->resize(
      seal::util::mul_safe(static_cast<size_t>(poly_deg_), num_modulus));

  uint64_t *dst = out->data();
  for (size_t mod_idx = 0; mod_idx < num_modulus; ++mod_idx) {
    std::fill_n(dst, poly_deg_, 0);
    absl::Span<uint64_t> dst_wrap(dst, poly_deg_);
    if (need_encrypt) {
      ms_helper_.ModulusUpAt(poly, mod_idx, dst_wrap);
    } else {
      ms_helper_.CenteralizeAt(poly, mod_idx, dst_wrap);
    }
    dst += poly_deg_;
  }

  out->parms_id() = ms_helper_.parms_id();
  out->scale() = 1.;
}

template <class Indexer>
ArrayRef TensorEncoder::Tensor2Poly(const Shape3D &input_shape,
                                    const Shape3D &kernel_shape,
                                    const Sliced3DTensor &tensor,
                                    const Indexer &indexer) const {
  int64_t isze = calcNumel(input_shape);
  int64_t ksze = calcNumel(kernel_shape);
  int64_t n_elt = tensor.numel();
  int64_t N = poly_deg_;
  SPU_ENFORCE(isze > 0 && ksze > 0, "invalid shapes");
  SPU_ENFORCE(n_elt == isze || n_elt == ksze, "shape mismatch");
  SPU_ENFORCE(n_elt <= N, "too large tensor to encode as one poly");

  Shape3D shape = isze == n_elt ? input_shape : kernel_shape;

  const auto field = tensor.field();
  return DISPATCH_ALL_FIELDS(field, "Tensor2Poly", [&]() {
    ArrayRef _flatten = ring_zeros(field, N);
    auto flatten = xt_mutable_adapt<ring2k_t>(_flatten);
    for (long c = 0; c < shape[kC]; ++c) {
      for (long h = 0; h < shape[kH]; ++h) {
        for (long w = 0; w < shape[kW]; ++w) {
          long coeff_index = indexer(h, w, c);
          SPU_ENFORCE(coeff_index >= 0 && coeff_index < N,
                      fmt::format("invalid index at ({}, {}, {})", h, w, c));
          flatten[coeff_index] = tensor.at<ring2k_t>({h, w, c});
        }
      }
    }
    return _flatten;
  });
}

}  // namespace spu::mpc::cheetah
