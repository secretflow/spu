// Copyright 2023 Ant Group Co., Ltd.
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

#include "libspu/mpc/generic/kernels.h"

#include <set>

#include "libspu/core/bit_utils.h"
#include "libspu/core/memref.h"
#include "libspu/mpc/ab_api.h"

namespace spu::mpc::generic {

namespace {

// Get a secret bit share of zero
MemRef wrap_bit_zero(SPUContext* ctx, SemanticType st, const Shape& shape) {
  auto zero = dynDispatch(ctx, "make_p", static_cast<uint128_t>(0), st, shape);
  return p2b(ctx, zero);
}

bool is_packed(const Type& ty) { return ty.as<BoolShare>()->is_packed(); }

MemRef getShare(const MemRef& in, int64_t share_idx) {
  SPU_ENFORCE_GE(share_idx, 0);

  auto new_strides = in.strides();
  int64_t factor = in.elsize() / SizeOf(in.eltype().storage_type());
  std::transform(new_strides.cbegin(), new_strides.cend(), new_strides.begin(),
                 [factor](int64_t s) { return factor * s; });

  const auto& in_ty = in.eltype();
  auto out_ty =
      makeType<RingTy>(in_ty.semantic_type(), SizeOf(in_ty.storage_type()) * 8);
  return MemRef(in.buf(), out_ty, in.shape(), new_strides,
                in.offset() + share_idx * out_ty.size());
}

}  // namespace

// Compact threshold heuristic, try to make it same as L1 cache size
#define COMPACT_THRESHOLD (32 * 1024)  // 32K
#define COMPACT_SIZE_LOWER_BOUND 10

SPU_ALWAYS_INLINE MemRef _try_compact(const MemRef& in) {
  // If in data is not compact after some shape ops and small enough, make it
  // compact
  if (in.numel() > COMPACT_SIZE_LOWER_BOUND &&
      in.numel() * in.elsize() <= COMPACT_THRESHOLD && !in.isCompact()) {
    return in.clone();
  }
  return in;
}

MemRef Broadcast::proc(KernelEvalContext* ctx, const MemRef& in,
                       const Shape& to_shape, const Axes& in_dims) const {
  return in.broadcast_to(to_shape, in_dims);
}

MemRef Reshape::proc(KernelEvalContext* ctx, const MemRef& in,
                     const Shape& to_shape) const {
  return _try_compact(in.reshape(to_shape));
}

MemRef ExtractSlice::proc(KernelEvalContext* ctx, const MemRef& in,
                          const Index& offsets, const Shape& sizes,
                          const Strides& strides) const {
  return _try_compact(in.slice(offsets, sizes, strides));
}

MemRef InsertSlice::proc(KernelEvalContext* ctx, const MemRef& in,
                         const MemRef& update, const Index& offsets,
                         const Strides& strides, bool prefer_in_place) const {
  SPU_ENFORCE(in.eltype() == update.eltype(),
              "Element type mismatch, in = {}, update ={}", in.eltype(),
              update.eltype());

  MemRef ret;
  if (prefer_in_place) {
    ret = in;
  } else {
    ret = in.clone();
  }
  ret.insert_slice(update, offsets, strides);
  return ret;
}

MemRef Transpose::proc(KernelEvalContext* ctx, const MemRef& in,
                       const Axes& permutation) const {
  Axes perm = permutation;
  if (perm.empty()) {
    // by default, transpose the data in reverse order.
    perm.resize(in.shape().size());
    std::iota(perm.rbegin(), perm.rend(), 0);
  }

  // sanity check.
  SPU_ENFORCE_EQ(perm.size(), in.shape().size());
  std::set<int64_t> uniq(perm.begin(), perm.end());
  SPU_ENFORCE_EQ(uniq.size(), perm.size(), "perm={} is not unique", perm);

  // fast path, if identity permutation, return it.
  Axes no_perm(in.shape().size());
  std::iota(no_perm.begin(), no_perm.end(), 0);
  if (perm == no_perm) {
    return in;
  }

  return _try_compact(in.transpose(perm));
}

MemRef Reverse::proc(KernelEvalContext* ctx, const MemRef& in,
                     const Axes& dimensions) const {
  return in.reverse(dimensions);
}

MemRef Fill::proc(KernelEvalContext* ctx, const MemRef& in,
                  const Shape& to_shape) const {
  return in.expand(to_shape);
}

MemRef Pad::proc(KernelEvalContext* ctx, const MemRef& in,
                 const MemRef& padding_value, const Sizes& edge_padding_low,
                 const Sizes& edge_padding_high) const {
  SPU_ENFORCE(in.eltype() == padding_value.eltype(),
              "Element type mismatch, in = {}, pad_value ={}", in.eltype(),
              padding_value.eltype());
  return in.pad(padding_value, edge_padding_low, edge_padding_high);
}

MemRef Concate::proc(KernelEvalContext* ctx, const std::vector<MemRef>& values,
                     int64_t axis) const {
  return values.front().concatenate(
      absl::MakeSpan(&values[1], values.size() - 1), axis);
}

// Todo: move this directly to Value
MemRef LShift::proc(KernelEvalContext* ctx, const MemRef& in,
                    const Sizes& shift) const {
  bool is_splat = shift.size() == 1;
  auto out_ty = in.eltype();
  int64_t out_nbits =
      in.eltype().as<BaseRingType>()->valid_bits() +
      (shift.empty() ? 0 : *std::max_element(shift.begin(), shift.end()));
  int64_t field = ctx->sctx()->config().protocol().field();
  out_nbits =
      std::clamp(out_nbits, 0L, static_cast<int64_t>(SizeOf(field) * 8));

  // fast path for packed bshares
  if (is_packed(in.eltype())) {
    const_cast<Type&>(out_ty).as<BaseRingType>()->set_valid_bits(out_nbits);
    const_cast<Type&>(out_ty).as<BaseRingType>()->set_storage_type(
        GetStorageType(field));

    int64_t n_share = in.elsize() / SizeOf(in.eltype().storage_type());
    MemRef out(out_ty, in.shape());
    DISPATCH_ALL_STORAGE_TYPES(in.eltype().storage_type(), [&]() {
      using IT = ScalarT;
      DISPATCH_ALL_STORAGE_TYPES(out_ty.storage_type(), [&]() {
        using OT = ScalarT;
        for (int64_t i = 0; i < n_share; ++i) {
          auto out_ = getShare(out, i);
          auto in_ = getShare(in, i);
          MemRefView<OT> _out(out_);
          MemRefView<IT> _in(in_);
          pforeach(0, in.numel(), [&](int64_t idx) {
            _out[idx] = static_cast<OT>(_in[idx])
                        << (is_splat ? shift[0] : shift[idx]);
          });
        }
      });
    });
    return out;
  }

  auto shift_fn = [&](const MemRef& t, int64_t offset) {
    auto bits = bit_decompose_b(ctx->sctx(), t);
    auto zero = wrap_bit_zero(ctx->sctx(), out_ty.semantic_type(), t.shape());
    bits.insert(bits.begin(), offset, zero);
    if (bits.size() < static_cast<size_t>(out_nbits)) {
      bits.insert(bits.end(), out_nbits - bits.size(), zero);
    }
    auto out = bit_compose_b(
        ctx->sctx(),
        std::vector<MemRef>(bits.begin(), bits.begin() + out_nbits));
    return out;
  };

  if (is_splat) {
    return shift_fn(in, shift[0]);
  }
  const auto numel = in.numel();
  SPU_ENFORCE_EQ(shift.size(), static_cast<size_t>(numel));
  auto flat_in = in.reshape({numel});
  std::vector<MemRef> outs(numel);
  pforeach(0, numel, [&](int64_t idx) {
    auto in_i = flat_in.slice({idx}, {1}, {1});
    outs[idx] = shift_fn(in_i, shift[idx]);
  });
  auto out =
      outs.front().concatenate(absl::MakeSpan(&outs[1], outs.size() - 1), 0);
  return out.reshape(in.shape());
}

MemRef RShift::proc(KernelEvalContext* ctx, const MemRef& in,
                    const Sizes& shift) const {
  bool is_splat = shift.size() == 1;
  auto out_ty = in.eltype();
  int64_t nbits = in.eltype().as<BaseRingType>()->valid_bits();
  int64_t out_nbits =
      nbits -
      std::min(
          nbits,
          (shift.empty() ? 0 : *std::min_element(shift.begin(), shift.end())));
  int64_t field = ctx->sctx()->config().protocol().field();
  out_nbits =
      std::clamp(out_nbits, 0L, static_cast<int64_t>(SizeOf(field) * 8));

  if (is_packed(in.eltype())) {
    const_cast<Type&>(out_ty).as<BaseRingType>()->set_valid_bits(out_nbits);
    const_cast<Type&>(out_ty).as<BaseRingType>()->set_storage_type(
        GetStorageType(field));

    int64_t n_share = in.elsize() / SizeOf(in.eltype().storage_type());
    MemRef out(out_ty, in.shape());
    DISPATCH_ALL_STORAGE_TYPES(in.eltype().storage_type(), [&]() {
      using IT = ScalarT;
      DISPATCH_ALL_STORAGE_TYPES(out_ty.storage_type(), [&]() {
        using OT = ScalarT;
        for (int64_t i = 0; i < n_share; ++i) {
          auto out_ = getShare(out, i);
          auto in_ = getShare(in, i);
          MemRefView<OT> _out(out_);
          MemRefView<IT> _in(in_);
          pforeach(0, in.numel(), [&](int64_t idx) {
            _out[idx] =
                static_cast<OT>(_in[idx]) >> (is_splat ? shift[0] : shift[idx]);
          });
        }
      });
    });
    return out;
  }

  if (out_nbits == 0) {
    return wrap_bit_zero(ctx->sctx(), out_ty.semantic_type(), in.shape());
  }
  auto shift_fn = [&](const MemRef& t, int64_t offset) {
    auto bits = bit_decompose_b(ctx->sctx(), t);
    if (static_cast<int64_t>(bits.size()) - offset < out_nbits) {
      auto zero = wrap_bit_zero(ctx->sctx(), out_ty.semantic_type(), t.shape());
      bits.insert(bits.end(), out_nbits - bits.size() + offset, zero);
    }
    auto out = bit_compose_b(
        ctx->sctx(), std::vector<MemRef>(bits.begin() + offset, bits.end()));
    return out;
  };

  if (is_splat) {
    return shift_fn(in, shift[0]);
  }
  const auto numel = in.numel();
  SPU_ENFORCE_EQ(shift.size(), static_cast<size_t>(numel));
  auto flat_in = in.reshape({numel});
  std::vector<MemRef> outs(numel);
  pforeach(0, numel, [&](int64_t idx) {
    auto in_i = flat_in.slice({idx}, {1}, {1});
    outs[idx] = shift_fn(in_i, shift[idx]);
  });
  auto out =
      outs.front().concatenate(absl::MakeSpan(&outs[1], outs.size() - 1), 0);
  return out.reshape(in.shape());
}

MemRef ARShift::proc(KernelEvalContext* ctx, const MemRef& in,
                     const Sizes& shift) const {
  bool is_splat = shift.size() == 1;
  auto st = in.eltype().storage_type();
  int64_t out_nbits = SizeOf(st) * 8;

  if (is_packed(in.eltype())) {
    MemRef out(in.eltype(), in.shape());
    const_cast<Type&>(out.eltype())
        .as<BaseRingType>()
        ->set_valid_bits(SizeOf(st) * 8);

    DISPATCH_ALL_STORAGE_TYPES(st, [&]() {
      int64_t n_share = in.elsize() / SizeOf(st);
      for (int64_t i = 0; i < n_share; ++i) {
        using T = std::make_signed_t<ScalarT>;
        auto out_ = getShare(out, i);
        auto in_ = getShare(in, i);
        MemRefView<T> _out(out_);
        MemRefView<T> _in(in_);
        pforeach(0, in.numel(), [&](int64_t idx) {
          _out[idx] = _in[idx] >> (is_splat ? shift[0] : shift[idx]);
        });
      }
    });
    return out;
  }

  auto shift_fn = [&](const MemRef& t, int64_t offset) {
    auto bits = bit_decompose_b(ctx->sctx(), t);
    if (static_cast<int64_t>(bits.size()) - offset < out_nbits) {
      auto msb = bits.back();
      bits.insert(bits.end(), out_nbits - bits.size() + offset, msb);
    }
    auto out = bit_compose_b(
        ctx->sctx(), std::vector<MemRef>(bits.begin() + offset, bits.end()));
    return out;
  };

  if (is_splat) {
    return shift_fn(in, shift[0]);
  }
  const auto numel = in.numel();
  SPU_ENFORCE_EQ(shift.size(), static_cast<size_t>(numel));
  auto flat_in = in.reshape({numel});
  std::vector<MemRef> outs(numel);
  pforeach(0, numel, [&](int64_t idx) {
    auto in_i = flat_in.slice({idx}, {1}, {1});
    outs[idx] = shift_fn(in_i, shift[idx]);
  });
  auto out =
      outs.front().concatenate(absl::MakeSpan(&outs[1], outs.size() - 1), 0);
  return out.reshape(in.shape());
}

MemRef BitDeintl::proc(KernelEvalContext* ctx, const MemRef& in,
                       size_t stride) const {
  if (is_packed(in.eltype())) {
    MemRef out(in.eltype(), in.shape());
    auto st = in.eltype().storage_type();
    int64_t nbits = in.eltype().as<BaseRingType>()->valid_bits();

    DISPATCH_ALL_STORAGE_TYPES(st, [&]() {
      int64_t n_share = in.elsize() / SizeOf(st);
      for (int64_t i = 0; i < n_share; ++i) {
        auto out_ = getShare(out, i);
        auto in_ = getShare(in, i);
        MemRefView<ScalarT> _out(out_);
        MemRefView<ScalarT> _in(in_);
        pforeach(0, in.numel(), [&](int64_t idx) {
          _out[idx] = spu::BitDeintl<ScalarT>(_in[idx], stride, nbits);
        });
      }
    });
    return out;
  }

  auto bits = bit_decompose_b(ctx->sctx(), in);
  int64_t nbits = bits.size();
  int64_t offset = 1 << stride;
  int64_t half_bits = nbits / 2;
  int64_t idx = 0;
  std::vector<MemRef> out_bits(nbits);
  while (idx < half_bits) {
    for (int j = 0; j < offset; ++j) {
      out_bits[idx] = bits[idx + j];
    }
    for (int j = 0; j < offset; ++j) {
      out_bits[idx + half_bits] = bits[idx + j];
    }
    idx += offset;
  }
  SPU_ENFORCE_EQ(idx, nbits);
  auto out = bit_compose_b(ctx->sctx(), out_bits);
  return out;
}

MemRef BitIntl::proc(KernelEvalContext* ctx, const MemRef& in,
                     size_t stride) const {
  if (is_packed(in.eltype())) {
    MemRef out(in.eltype(), in.shape());
    auto st = in.eltype().storage_type();
    int64_t nbits = in.eltype().as<BaseRingType>()->valid_bits();

    DISPATCH_ALL_STORAGE_TYPES(st, [&]() {
      int64_t n_share = in.elsize() / SizeOf(st);
      for (int64_t i = 0; i < n_share; ++i) {
        auto out_ = getShare(out, i);
        auto in_ = getShare(in, i);
        MemRefView<ScalarT> _out(out_);
        MemRefView<ScalarT> _in(in_);
        pforeach(0, in.numel(), [&](int64_t idx) {
          _out[idx] = spu::BitIntl<ScalarT>(_in[idx], stride, nbits);
        });
      }
    });
    return out;
  }

  auto bits = bit_decompose_b(ctx->sctx(), in);
  int64_t nbits = bits.size();
  int64_t offset = 1 << stride;
  int64_t half_bits = nbits / 2;
  int64_t idx = 0;
  std::vector<MemRef> out_bits(nbits);
  for (int64_t i = 0; i < half_bits; i += offset) {
    for (int j = 0; j < offset; ++j) {
      out_bits[idx++] = bits[i + j];
    }
    for (int j = 0; j < offset; ++j) {
      out_bits[idx++] = bits[i + j + half_bits];
    }
  }
  SPU_ENFORCE_EQ(idx, nbits);
  auto out = bit_compose_b(ctx->sctx(), out_bits);
  return out;
}

MemRef Bitrev::proc(KernelEvalContext* ctx, const MemRef& in, size_t start,
                    size_t end) const {
  SPU_ENFORCE(start <= end && end <= 128);

  if (is_packed(in.eltype())) {
    auto out_ty = in.eltype();
    int64_t nbits = in.eltype().as<BaseRingType>()->valid_bits();
    int64_t out_nbits = std::max(nbits, static_cast<int64_t>(end));
    int64_t field = ctx->sctx()->config().protocol().field();
    out_nbits =
        std::clamp(out_nbits, 0L, static_cast<int64_t>(SizeOf(field) * 8));
    const_cast<Type&>(out_ty).as<BaseRingType>()->set_valid_bits(out_nbits);
    const_cast<Type&>(out_ty).as<BaseRingType>()->set_storage_type(
        GetStorageType(field));

    int64_t n_share = in.elsize() / SizeOf(in.eltype().storage_type());
    MemRef out(out_ty, in.shape());
    DISPATCH_ALL_STORAGE_TYPES(in.eltype().storage_type(), [&]() {
      using IT = ScalarT;
      DISPATCH_ALL_STORAGE_TYPES(out_ty.storage_type(), [&]() {
        using OT = ScalarT;

        auto bitrev_fn = [&](OT el) {
          OT tmp = 0U;
          for (size_t idx = start; idx < end; idx++) {
            if (el & ((OT)1 << idx)) {
              tmp |= (OT)1 << (end - 1 - idx + start);
            }
          }

          OT mask = ((OT)1U << end) - ((OT)1U << start);
          return (el & ~mask) | tmp;
        };

        for (int64_t i = 0; i < n_share; ++i) {
          auto out_ = getShare(out, i);
          auto in_ = getShare(in, i);
          MemRefView<OT> _out(out_);
          MemRefView<IT> _in(in_);
          pforeach(0, in.numel(), [&](int64_t idx) {
            _out[idx] = bitrev_fn(static_cast<OT>(_in[idx]));
          });
        }
      });
    });
    return out;
  }

  auto bits = bit_decompose_b(ctx->sctx(), in);
  auto nbits = bits.size();
  if (start > nbits || start == end) {
    return in;
  }
  std::vector<MemRef> out_bits(nbits);
  if (nbits < end) {
    auto zero =
        wrap_bit_zero(ctx->sctx(), in.eltype().semantic_type(), in.shape());
    out_bits.insert(out_bits.end(), end - nbits, zero);
  }
  for (size_t i = 0; i < start; ++i) {
    out_bits[i] = bits[i];
  }
  for (size_t i = end; i < nbits; ++i) {
    out_bits[i] = bits[i];
  }
  for (size_t i = 0; i + start < end; ++i) {
    out_bits[end - 1 - i] = bits[i + start];
  }
  auto out = bit_compose_b(ctx->sctx(), out_bits);
  return out;
}

}  // namespace spu::mpc::generic
