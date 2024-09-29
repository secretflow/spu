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

#include "libspu/device/intrinsics/dispatcher.h"

#include "libspu/core/context.h"
#include "libspu/device/executor.h"
#include "libspu/device/intrinsic_table.h"
#include "libspu/dialect/ring/IR/types.h"
#include "libspu/kernel/hal/debug.h"
#include "libspu/kernel/hal/permute.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/prot_wrapper.h"
#include "libspu/kernel/hal/ring.h"

namespace spu::device {

MemRef iabs(SPUContext* ctx, const MemRef& x) {
  return kernel::hal::_mul(ctx, kernel::hal::_sign(ctx, x), x);
}

MemRef ipower(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  // ref:
  //
  // https://github.com/openxla/stablehlo/blob/main/stablehlo/reference/Element.cpp#L912
  // Although there are some "strange" semantics in stablehlo, we still
  // follow
  // them yet:
  //   1. when x is int, then the return value must be int type.
  //   2. if x is int, then y must be int
  //   3. if x is int and y<0, then
  //      a. when |x|!=1, then always return 0;
  //      b. when |x|=1, then y=|y|;
  //
  // However, for jax.numpy.power, it behaves differently:
  //   1. if any x or y is float, then both x and y will be upcast to float.
  //   2. if both x and y are int, then y must be non-negative.
  auto k0 =
      kernel::hal::_constant(ctx, 0, x.eltype().semantic_type(), x.shape());
  auto k1 =
      kernel::hal::_constant(ctx, 1, x.eltype().semantic_type(), x.shape());

  auto y_b = kernel::hal::_prefer_b(ctx, y);
  const int64_t bit_width = SizeOf(y_b.eltype().storage_type()) * 8;
  auto msb_y = kernel::hal::_rshift(ctx, y_b, {bit_width - 1});
  auto x_abs1 = kernel::hal::_equal(ctx, iabs(ctx, x), k1);

  auto ret = kernel::hal::_constant(ctx, 1, x.eltype().semantic_type(),
                                    x.shape());  // To compute ret = x^y,
  // although y has `bit_width` bits, we only consider `y_bits` bits here.
  // The reason are two folds (recall that both x and y are int):
  //   1. if |x|>1, then `ret` will OVERFLOW/UNDERFLOW if y>63 (e.g. FM64),
  //   which means the valid bits of y can't exceed `log(bit_width - 1)` .
  //   2. if |x|=1:
  //      a). x=1, then we always get `ret`=1;
  //      b). x=-1, then the sign of `ret` is decided on the LSB of y;
  // So we can "truncate" y to `y_bits` bits safely.
  const int64_t y_bits = Log2Ceil(bit_width - 1);

  auto base = x;
  // TODO: do this in parallel
  // To compute x^y, it is necessary to compute all x^(2^idx), we use base
  // (init as `x`) to store it, update base to base*base till last
  // iteration, and multiply all these numbers according to y_{idx}.
  // e.g. y=0101, then ret = (x) * (1) * (x^(2^2)) * (1) = x^5
  for (int64_t idx = 0; idx < y_bits; idx++) {
    // x^(2^idx) * y_{idx}
    auto cur_pow = kernel::hal::_mux(
        ctx, kernel::hal::_and(ctx, kernel::hal::_rshift(ctx, y_b, {idx}), k1),
        base, k1);
    ret = kernel::hal::_mul(ctx, cur_pow, ret);
    if (idx < y_bits - 1) {
      base = kernel::hal::_mul(ctx, base, base);
    }
  }

  // promote msb to target type
  msb_y = kernel::hal::_ring_cast(ctx, msb_y, x_abs1.eltype().semantic_type());
  // when x=-1 and y<0, we can still get a correct result
  return kernel::hal::_mux(
      ctx, kernel::hal::_and(ctx, msb_y, kernel::hal::_not(ctx, x_abs1)), k0,
      ret);
}

using Rank1dFn = std::function<std::vector<spu::MemRef>(const spu::MemRef&)>;

std::vector<spu::MemRef> TopkApply(SPUContext* ctx, const spu::MemRef& input,
                                   const Rank1dFn& apply_fn) {
  const Shape& shape = input.shape();

  // Topk always deals last-dimension
  // - N is the number of vector to permute
  // - W is the vector length.
  const int64_t W = shape.back();
  const int64_t N = shape.numel() / W;

  // First, reshape the input to (N, W)
  auto reshaped = kernel::hal::_reshape(ctx, input, {N, W});

  // Then, do topk in last dimension
  std::vector<std::vector<spu::MemRef>> topk1d;
  topk1d.reserve(N);
  for (int64_t i = 0; i < N; ++i) {
    // TODO: how to do these parallelly?
    auto input_i = kernel::hal::_reshape(
        ctx, kernel::hal::_extract_slice(ctx, reshaped, {i, 0}, {1, W}, {}),
        {W});
    topk1d.push_back(apply_fn(input_i));
  }

  const bool include_index = topk1d[0].size() == 2;

  // the output shape is (..., k)
  Shape new_shape(shape.begin(), shape.end());
  const auto k = topk1d[0][0].numel();
  new_shape.back() = k;

  // Finally, Reshape back to shape
  std::vector<spu::MemRef> ret;
  ret.reserve(2);

  std::vector<spu::MemRef> value2d;
  value2d.reserve(N);
  for (int64_t i = 0; i < N; ++i) {
    value2d.push_back(kernel::hal::_reshape(ctx, topk1d[i][0], {1, k}));
  }
  auto ret_val = kernel::hal::_concatenate(ctx, value2d, 0);
  ret.push_back(kernel::hal::_reshape(ctx, ret_val, new_shape));
  if (include_index) {
    std::vector<spu::MemRef> index2d;
    index2d.reserve(N);
    for (int64_t i = 0; i < N; ++i) {
      index2d.push_back(kernel::hal::_reshape(ctx, topk1d[i][1], {1, k}));
    }
    auto ret_inx = kernel::hal::_concatenate(ctx, index2d, 0);
    ret.push_back(kernel::hal::_reshape(ctx, ret_inx, new_shape));
  }
  return ret;
}

std::vector<spu::MemRef> TopK(SPUContext* ctx, const spu::MemRef& input,
                              int64_t k_lo, int64_t k_hi, bool largest,
                              bool value_only) {
  const Shape& shape = input.shape();
  SPU_ENFORCE(shape.numel() > 0, "input must non-empty.");
  SPU_ENFORCE(
      k_lo <= shape.back() && k_lo > 0,
      "k_lo should be larger than 0 and smaller than the last dimension.");

  if (k_hi == -1) {
    k_hi = k_lo;
  }

  SPU_ENFORCE(k_lo <= k_hi,
              "k_lo should be smaller than k_hi, got k_lo={}, k_hi={}", k_lo,
              k_hi);

  auto scalar_cmp_fn = [largest](spu::SPUContext* ctx, const spu::MemRef& lhs,
                                 const spu::MemRef& rhs) {
    if (largest) {
      return kernel::hal::_less(ctx, rhs, lhs);
    } else {
      return kernel::hal::_less(ctx, lhs, rhs);
    }
  };

  kernel::hal::TopKConfig config = {value_only, false, k_lo, k_hi};

  auto topk_fn = [&](const spu::MemRef& input) {
    return kernel::hal::topk_1d(ctx, input, scalar_cmp_fn, config);
  };

  return TopkApply(ctx, input, topk_fn);
}

// <<<<--------------------------------------------------------------
// HLO Instrinsics, clean them later
std::vector<spu::MemRef> Shuffle(SPUContext* ctx,
                                 absl::Span<const spu::MemRef> inputs,
                                 int64_t axis) {
  SPU_ENFORCE_GT(inputs.size(), 0U);
  SPU_ENFORCE(ctx->hasKernel("rand_perm_m") && ctx->hasKernel("perm_am"));

  if (inputs[0].numel() == 0) {
    return std::vector<spu::MemRef>(inputs.begin(), inputs.end());
  }

  auto input_shape = inputs[0].shape();

  auto _2s = [](SPUContext* ctx, const MemRef& x) {
    if (x.isPublic()) {
      return kernel::hal::_p2s(ctx, x);
    } else if (x.isPrivate()) {
      return kernel::hal::_v2s(ctx, x);
    }
    return x;
  };

  auto shuffle_fn = [&](absl::Span<const spu::MemRef> input) {
    auto rand_perm = kernel::hal::_rand_perm_s(ctx, input_shape);

    std::vector<spu::MemRef> rets;
    rets.reserve(input.size());

    for (size_t i = 0; i < input.size(); ++i) {
      rets.emplace_back(
          kernel::hal::_perm_ss(ctx, _2s(ctx, input[i]), rand_perm));
    }

    return rets;
  };

  return kernel::hal::permute(ctx, inputs, axis, shuffle_fn);
}

spu::MemRef FilterByMask(SPUContext*, const spu::MemRef& operand,
                         absl::Span<const uint8_t> mask) {
  // Sanity
  SPU_ENFORCE(operand.shape().size() == 1, "Operand must be a vector");
  SPU_ENFORCE(mask.size() == (size_t)operand.shape()[0],
              "filter must be same length as operand");

  // Count result size
  int64_t num_true = 0;
  for (auto m : mask) {
    if (m != 0) {
      ++num_true;
    }
  }

  Index indices(num_true);
  int64_t indices_counter = 0;
  for (int64_t mask_idx = 0; mask_idx != static_cast<int64_t>(mask.size());
       ++mask_idx) {
    if (mask[mask_idx] != 0) {
      indices[indices_counter++] = mask_idx;
    }
  }

  return operand.linear_gather(indices).as(operand.eltype());
}

spu::MemRef LinearScatterInPlace(SPUContext* ctx, spu::MemRef in,
                                 const spu::MemRef& update,
                                 const Index& indices) {
  if (in.eltype() != update.eltype()) {
    auto common_type =
        kernel::hal::_common_type(ctx, in.eltype(), update.eltype());

    return LinearScatterInPlace(
        ctx, kernel::hal::_cast_type(ctx, in, common_type),
        kernel::hal::_cast_type(ctx, update, common_type), indices);
  }

  return spu::MemRef(in.linear_scatter(update, indices));
}
// -------------------------------------------------------------->>>>

std::vector<MemRef> dispatcher(OpExecutor* executor, SPUContext* ctx,
                               mlir::func::CallOp& call,
                               absl::Span<const MemRef> inputs) {
  // DO-NOT-EDIT: Add_DISPATCH_CODE
  auto call_name = demangle_fcn_name(call.getCallee());

  if (call_name == DBG_PRINT) {
    SPDLOG_INFO(kernel::hal::dbg_print<int128_t>(ctx, inputs[0]));
    return {};
  }

  if (call_name == TOPK) {
    SPU_ENFORCE(inputs.size() == 1);
    auto attr =
        mlir::dyn_cast<mlir::DictionaryAttr>(call->getAttr("mhlo.attributes"));
    auto k = mlir::dyn_cast<mlir::IntegerAttr>(attr.get("k")).getInt();
    auto largest =
        mlir::dyn_cast<mlir::BoolAttr>(attr.get("largest")).getValue();

    auto value_only = false;

    if (auto value_only_attr = attr.get("value_only")) {
      value_only = mlir::dyn_cast<mlir::BoolAttr>(value_only_attr).getValue();
    }

    if (auto k_hi_attr = attr.get("k_hi")) {
      auto k_hi = mlir::dyn_cast<mlir::IntegerAttr>(k_hi_attr).getInt();
      return TopK(ctx, inputs[0], k, k_hi, largest, value_only);
    }

    return TopK(ctx, inputs[0], k, -1, largest, value_only);
  }

  if (call_name == SIMPLE_SORT) {
    auto attr = mlir::dyn_cast<mlir::DictionaryAttr>(
        call->getAttr("spu.sort.attributes"));
    auto sort_dim = mlir::cast<mlir::IntegerAttr>(attr.get("dim")).getInt();
    auto num_keys =
        mlir::cast<mlir::IntegerAttr>(attr.get("num_keys")).getInt();
    bool is_ascending =
        mlir::cast<mlir::BoolAttr>(attr.get("is_ascending")).getValue();

    kernel::hal::SortDirection direction =
        is_ascending ? kernel::hal::SortDirection::Ascending
                     : kernel::hal::SortDirection::Descending;

    auto sort_fn = [&](absl::Span<const spu::MemRef> input) {
      return kernel::hal::simple_sort1d(ctx, input, direction, num_keys, -1);
    };

    return kernel::hal::permute(ctx, inputs, sort_dim, sort_fn);
  }

  if (call_name == MAKE_CACHED_VAR) {
    if (ctx->hasKernel("beaver_cache")) {
      SPU_ENFORCE(inputs.size() == 1);
      dynDispatch(ctx, "beaver_cache", inputs[0], true);
    }

    return {inputs[0]};
  }

  if (call_name == DROP_CACHED_VAR) {
    if (ctx->hasKernel("beaver_cache")) {
      SPU_ENFORCE(inputs.size() > 0);
      dynDispatch(ctx, "beaver_cache", inputs[0], false);
    }

    return {inputs[0]};
  }

  if (call_name == IPOW) {
    auto ret = ipower(ctx, inputs[0], inputs[1]);
    return {ret};
  }

  if (call_name == PREFER_A) {
    return {kernel::hal::_prefer_a(ctx, inputs[0])};
  }

  if (call_name == ROUND_NE) {
    return {kernel::hal::round_tne(ctx, inputs[0])};
  }

  if (call_name == GENERIC_SORT) {
    // get sort kernel
    auto comparator_name =
        mlir::cast<mlir::StringAttr>(call->getAttr("comparator")).strref();
    auto* module = mlir::SymbolTable::getNearestSymbolTable(call);
    auto fcn = mlir::cast<mlir::func::FuncOp>(
        mlir::SymbolTable::lookupSymbolIn(module, comparator_name));

    SPU_ENFORCE(fcn, "Unable to find comparator");

    auto sort_dim =
        mlir::cast<mlir::IntegerAttr>(call->getAttr("dimension")).getInt();
    auto is_stable =
        mlir::cast<mlir::BoolAttr>(call->getAttr("is_stable")).getValue();

    SymbolScope root;

    auto comparator = [&](absl::Span<const spu::MemRef> inputs) {
      auto ret = runRegion(executor, ctx, &root, fcn.getBody(), inputs);
      return ret[0];
    };

    auto comparator_return = fcn.getFunctionType().getResult(0);
    auto secret_sort = mlir::isa<mlir::spu::ring::SecretType>(
        mlir::getElementTypeOrSelf(comparator_return));

    const spu::Visibility spu_return_vis =
        secret_sort ? spu::Visibility::VIS_SECRET : spu::Visibility::VIS_PUBLIC;

    // NOTE(junfeng):
    // https://github.com/google/jax/blob/e5b2c5ea44b44439bf574cbdc0944c36b167c10c/jax/_src/numpy/lax_numpy.py#L3439
    // 'kind' is ignored in jax.numpy.sort and fixed to 'quicksort'. In order
    // to to accommodate this situation, we need to modify 'is_stable' here.
    if (is_stable && spu_return_vis == spu::Visibility::VIS_SECRET) {
      SPDLOG_WARN("only unstable sort is supported for secret returns.");
      is_stable = false;
    }

    auto sort_fn = [&](absl::Span<const MemRef> input) {
      return kernel::hal::sort1d(ctx, input, comparator, spu_return_vis,
                                 is_stable);
    };

    return kernel::hal::permute(ctx, inputs, sort_dim, sort_fn);
  }

  // <<<<--------------------------------------------------------------
  // HLO Instrinsics, clean them later
  if (call_name == "hlo.shuffle") {
    auto attr =
        mlir::dyn_cast<mlir::DictionaryAttr>(call->getAttr("mhlo.attributes"));
    auto axis = mlir::dyn_cast<mlir::IntegerAttr>(attr.get("axis")).getInt();

    return Shuffle(ctx, inputs, axis);
  }

  if (call_name == "hlo.filter_by_mask") {
    auto attr =
        mlir::dyn_cast<mlir::DictionaryAttr>(call->getAttr("mhlo.attributes"));
    auto mask =
        mlir::dyn_cast<mlir::DenseI8ArrayAttr>(attr.get("mask")).asArrayRef();

    return {
        FilterByMask(ctx, inputs[0],
                     absl::Span(reinterpret_cast<const uint8_t*>(mask.data()),
                                mask.size()))};
  }

  if (call_name == "hlo.linear_gather") {
    auto attr =
        mlir::dyn_cast<mlir::DictionaryAttr>(call->getAttr("mhlo.attributes"));
    auto indices = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(attr.get("indices"))
                       .asArrayRef();

    return {MemRef(inputs[0].linear_gather(indices).as(inputs[0].eltype()))};
  }

  if (call_name == "hlo.linear_scatter") {
    auto attr =
        mlir::dyn_cast<mlir::DictionaryAttr>(call->getAttr("mhlo.attributes"));
    auto indices = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(attr.get("indices"))
                       .asArrayRef();

    return {LinearScatterInPlace(ctx, inputs[0], inputs[1], indices)};
  }
  // -------------------------------------------------------------->>>>

  SPU_THROW("Unhandled intrinsic call {}", call_name.str());
}

}  // namespace spu::device
