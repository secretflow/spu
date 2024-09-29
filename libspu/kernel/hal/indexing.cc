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

#include "libspu/kernel/hal/indexing.h"

#include <cstring>

#include "llvm/ADT/STLExtras.h"

#include "libspu/core/memref.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/kernel/hal/shape_ops.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/hal/utils.h"

// forward
namespace spu::kernel::hal::detail {

void hintNumberOfBits(const MemRef &a, size_t nbits) {
  if (a.eltype().isa<BoolShare>()) {
    const_cast<Type &>(a.eltype()).as<BaseRingType>()->set_valid_bits(nbits);
  }
}

}  // namespace spu::kernel::hal::detail

namespace {
struct IndexIterationSpace {
  spu::Index index_base;
  spu::Index index_count;
  spu::Index index_incr;
};

spu::MemRef SecretLinearUpdateIndexing(spu::SPUContext *ctx,
                                       const spu::MemRef &operand,
                                       const spu::MemRef &update,
                                       const spu::MemRef &linear_idx) {
  // TODO: Consider utilizing DLP to improve performance
  SPU_ENFORCE(operand.shape().size() == 1, "operand must be a 1D tensor");
  SPU_ENFORCE(linear_idx.numel() == 1, "index must be a 1D indexing");
  SPU_ENFORCE(update.numel() == 1, "update must be a scalar");

  // Basic idea here:
  // eq(iota, idx) * update + !eq(iota, idx) * operand
  auto linear_idx_broadcasted =
      spu::kernel::hal::broadcast_to(ctx, linear_idx, {operand.numel()}, {});
  spu::MemRef idx_iota =
      spu::kernel::hal::iota(ctx, spu::PT_I64, operand.numel());
  auto mask = spu::kernel::hal::_equal(ctx, linear_idx_broadcasted, idx_iota);

  auto reverse_mask = spu::kernel::hal::logical_not(ctx, mask);

  auto broadcast_update =
      spu::kernel::hal::broadcast_to(ctx, update, operand.shape(), {0});

  return spu::kernel::hal::_add(
      ctx, spu::kernel::hal::_mul(ctx, operand, reverse_mask),
      spu::kernel::hal::_mul(ctx, broadcast_update, mask));
}

std::vector<spu::MemRef> ClampAndFlattenIndex(
    spu::SPUContext *ctx, absl::Span<const spu::MemRef> start_indices,
    const spu::Shape &iterate_shape, const spu::Shape &limit_shape) {
  // Transform start_indices
  // start_indices[i] = clamp(start_indices[i], 0, operand.dimension_size[i]
  // - size_indices[i])

  std::vector<spu::MemRef> clamped_start(start_indices.size());
  {
    std::vector<spu::MemRef> reshaped_start_indices;
    std::transform(start_indices.cbegin(), start_indices.cend(),
                   std::back_inserter(reshaped_start_indices),
                   [&](const spu::MemRef &x) {
                     return spu::kernel::hal::unsqueeze(ctx, x);
                   });

    auto concat_idx =
        spu::kernel::hal::concatenate(ctx, reshaped_start_indices, 0);
    auto lower_bound = spu::kernel::hal::constant(ctx, static_cast<int64_t>(0),
                                                  concat_idx.shape());

    std::vector<int64_t> upper_bound_pt(start_indices.size());
    for (size_t idx = 0; idx < upper_bound_pt.size(); ++idx) {
      upper_bound_pt[idx] = limit_shape[idx] - iterate_shape[idx];
    }
    auto upper_bound =
        spu::kernel::hal::constant(ctx, upper_bound_pt, concat_idx.shape());

    auto c = spu::kernel::hal::clamp(ctx, concat_idx, lower_bound, upper_bound);
    for (int64_t idx = 0; idx < static_cast<int64_t>(clamped_start.size());
         ++idx) {
      clamped_start[idx] = spu::kernel::hal::squeeze(
          ctx, spu::kernel::hal::slice(ctx, c, {idx}, {1}, {1}));
    }
  }

  // Now flatten start index
  auto linear_idx = spu::kernel::hal::constant(ctx, static_cast<int64_t>(0));
  int64_t stride = 1;
  for (int64_t idx = iterate_shape.size() - 1; idx >= 0; --idx) {
    linear_idx = spu::kernel::hal::_add(
        ctx, linear_idx,
        spu::kernel::hal::_mul(ctx, clamped_start[idx],
                               spu::kernel::hal::constant(ctx, stride)));
    stride *= limit_shape[idx];
  }

  // Now compute offsets of each index
  spu::Index base(iterate_shape.size(), 0);
  spu::Index incr(iterate_shape.size(), 1);

  spu::Index flatten_idx;
  spu::kernel::hal::forEachIndex(
      limit_shape, base, iterate_shape, incr,
      [&flatten_idx, &limit_shape](const spu::Index &idx) {
        flatten_idx.emplace_back(spu::flattenIndex(idx, limit_shape));
      });

  auto num_index = iterate_shape.numel();
  std::vector<spu::MemRef> linear_indices;
  linear_indices.reserve(num_index);
  auto added = spu::kernel::hal::_add(
      ctx,
      spu::kernel::hal::broadcast_to(
          ctx, spu::kernel::hal::unsqueeze(ctx, linear_idx), {num_index}, {0}),
      spu::kernel::hal::constant(ctx, flatten_idx, {num_index}));
  for (int64_t idx = 0; idx < num_index; ++idx) {
    linear_indices.emplace_back(spu::kernel::hal::squeeze(
        ctx, spu::kernel::hal::slice(ctx, added, {idx}, {1}, {1})));
  }
  return linear_indices;
}

}  // namespace

namespace spu::kernel::hal {

spu::MemRef DynamicUpdateSlice(SPUContext *ctx, const spu::MemRef &operand,
                               const spu::MemRef &update,
                               absl::Span<const spu::MemRef> start_indices,
                               bool prefer_in_place) {
  SPU_ENFORCE(!operand.isComplex());
  // Basic idea here, get a ref slice and update the whole slice..
  SPU_ENFORCE_EQ(start_indices.size(), operand.shape().size());
  SPU_ENFORCE_EQ(start_indices.size(), update.shape().size());
  SPU_ENFORCE(!start_indices.empty());

  if (start_indices[0].isSecret()) {
    // flatten first
    spu::MemRef flattened_operand =
        hal::reshape(ctx, operand, {operand.numel()});

    spu::MemRef flattened_update = hal::reshape(ctx, update, {update.numel()});

    auto flattened_indices = ClampAndFlattenIndex(
        ctx, start_indices, update.shape(), operand.shape());

    spu::MemRef ret = flattened_operand;

    for (int64_t n = 0; n < static_cast<int64_t>(flattened_indices.size());
         ++n) {
      auto update_slice = hal::slice(ctx, flattened_update, {n}, {1}, {1});
      ret = SecretLinearUpdateIndexing(ctx, ret, update_slice,
                                       flattened_indices[n]);
    }

    return hal::reshape(ctx, ret, operand.shape());

  } else {
    // Start indices
    Index start_indices_i64(start_indices.size());
    for (const auto &idx : llvm::enumerate(start_indices)) {
      auto v_idx = idx.value();
      if (v_idx.isSecret()) {
        v_idx = hal::reveal(ctx, v_idx);
        SPDLOG_WARN("Reveal {}th start index of DynamicUpdateSlice",
                    idx.index());
      }
      start_indices_i64[idx.index()] =
          hal::dump_public_as<int64_t>(ctx, v_idx)[0];
      // Transform start_indices
      // start_indices[i] = clamp(start_indices[i], 0, operand.dimension_size[i]
      // - update.dimension_size[i])
      start_indices_i64[idx.index()] = std::min(
          std::max(start_indices_i64[idx.index()], static_cast<int64_t>(0)),
          operand.shape()[idx.index()] - update.shape()[idx.index()]);
    }

    return hal::insert_slice(ctx, operand, update, start_indices_i64,
                             Strides(start_indices_i64.size(), 1),
                             prefer_in_place);
  }
}

spu::MemRef SecretDynamicSliceImpl(
    SPUContext *ctx, const spu::MemRef &operand, const Sizes &slice_size,
    absl::Span<const spu::MemRef> start_indices) {
  if (slice_size[0] == operand.shape()[0]) {
    if (slice_size.size() == 1) {
      return operand;
    }

    // Full dimension
    Index start(operand.shape().size(), 0);
    Shape sizes = operand.shape();
    sizes[0] = 1;
    Strides strides(operand.shape().size(), 1);

    std::vector<spu::MemRef> results(operand.shape()[0]);
    for (int64_t idx = 0; idx < operand.shape()[0]; ++idx) {
      start[0] = idx;
      // Slice one...
      auto sliced = hal::slice(ctx, operand, start, sizes, strides);
      // Remove leading one
      auto reshaped = hal::squeeze(ctx, sliced);
      // Do indexing
      auto indexed = SecretDynamicSliceImpl(
          ctx, reshaped, {slice_size.begin() + 1, slice_size.end()},
          start_indices.subspan(1));
      // Add leading one dimension back
      results[idx] = hal::unsqueeze(ctx, indexed);
    }

    if (results.size() == 1) {
      return results[0];
    }
    return hal::concatenate(ctx, results, 0);
  }

  // equal(adjusted, iota)
  spu::MemRef mask;
  spu::MemRef idx_iota = hal::iota(ctx, PT_I32, operand.shape()[0]);
  idx_iota =
      hal::_ring_cast(ctx, idx_iota, start_indices[0].eltype().semantic_type());

  mask = hal::_equal(ctx,
                     hal::broadcast_to(ctx, start_indices[0], idx_iota.shape()),
                     idx_iota);

  if (slice_size[0] >= 1) {
    auto pad_value = hal::seal(ctx, hal::constant(ctx, false));
    mask = hal::pad(ctx, mask, pad_value, {slice_size[0]}, {0});
    // FIXME(juhou): we should avoid setting the BShr here
    // However mask.eltype().as<BShare>->nbits() is not 1 after the
    // padding. We implicitly set mask as a 1-bit BShr so that the following
    // hal::matmul can use a much lighter B2A proc for both ABY3 and CHEETAH.
    hal::detail::hintNumberOfBits(mask, 1);
  }

  // foreach
  std::vector<spu::MemRef> results(slice_size[0]);

  // Do collapse inner dims when necessary
  auto collapsed_operand = operand;
  if (collapsed_operand.shape().size() > 2) {
    // Reshape from XxYxZ to Xx(Y*Z)
    collapsed_operand = hal::reshape(
        ctx, collapsed_operand,
        {operand.shape()[0],
         Shape(operand.shape().begin() + 1, operand.shape().end()).numel()});
  }

  Shape indexed_shape = operand.shape();
  indexed_shape[0] = 1;

  for (int64_t idx = 0; idx < slice_size[0]; ++idx) {
    auto mask_slice =
        hal::slice(ctx, mask, {mask.numel() - idx - operand.shape()[0]},
                   {operand.shape()[0]}, {1});
    mask_slice = hal::unsqueeze(ctx, mask_slice);

    results[idx] = hal::_mmul(ctx, mask_slice, collapsed_operand);

    results[idx] = hal::reshape(ctx, results[idx], indexed_shape);
  }

  if (slice_size.size() > 1) {
    // Keep indexing deeper
    for (int64_t idx = 0; idx < slice_size[0]; ++idx) {
      results[idx] = hal::squeeze(ctx, results[idx]);
      results[idx] = SecretDynamicSliceImpl(
          ctx, results[idx], {slice_size.begin() + 1, slice_size.end()},
          start_indices.subspan(1));
      results[idx] = hal::unsqueeze(ctx, results[idx]);
    }
  }

  if (results.size() == 1) {
    return results[0];
  }

  return hal::concatenate(ctx, results, 0);
}

spu::MemRef SecretDynamicSliceOramImpl(
    SPUContext *ctx, const spu::MemRef &operand, const Sizes &slice_size,
    absl::Span<const spu::MemRef> start_indices) {
  if (slice_size[0] == operand.shape()[0]) {
    if (slice_size.size() == 1) {
      return operand;
    }

    // Full dimension
    Index start(operand.shape().size(), 0);
    Shape sizes = operand.shape();
    sizes[0] = 1;
    Strides strides(operand.shape().size(), 1);

    std::vector<spu::MemRef> results(operand.shape()[0]);
    for (int64_t idx = 0; idx < operand.shape()[0]; ++idx) {
      start[0] = idx;
      // Slice one...
      auto sliced = hal::slice(ctx, operand, start, sizes, strides);
      // Remove leading one
      auto reshaped = hal::reshape(
          ctx, sliced, {sliced.shape().begin() + 1, sliced.shape().end()});
      // Do indexing
      auto indexed = SecretDynamicSliceOramImpl(
          ctx, reshaped, {slice_size.begin() + 1, slice_size.end()},
          start_indices.subspan(1));
      // Add leading one dimension back
      Shape result_shape(indexed.shape().size() + 1, 1);
      for (size_t idx = 0; idx < indexed.shape().size(); ++idx) {
        result_shape[idx + 1] = indexed.shape()[idx];
      }
      results[idx] = hal::reshape(ctx, indexed, result_shape);
    }

    if (results.size() == 1) {
      return results[0];
    }
    return hal::concatenate(ctx, results, 0);
  }

  if (start_indices[0].isPublic()) {
    return SecretDynamicSliceImpl(ctx, operand, slice_size, start_indices);
  }

  // try oram impl
  auto opt_onehot = hal::oramonehot(ctx, start_indices[0], operand.shape()[0],
                                    operand.isPublic());
  if (!opt_onehot.has_value()) {
    // fall back
    return SecretDynamicSliceImpl(ctx, operand, slice_size, start_indices);
  }

  auto onehot = opt_onehot.value();

  // Do collapse inner dims when necessary
  auto collapsed_operand = operand;
  if (collapsed_operand.shape().size() > 2) {
    // Reshape from XxYxZ to Xx(Y*Z)
    collapsed_operand = hal::reshape(
        ctx, collapsed_operand,
        {operand.shape()[0],
         Shape(operand.shape().begin() + 1, operand.shape().end()).numel()});
  }

  // slice each
  std::vector<spu::MemRef> results(slice_size[0]);
  Shape indexed_shape = operand.shape();
  indexed_shape[0] = 1;

  for (int64_t idx = 0; idx < slice_size[0]; ++idx) {
    results[idx] = hal::oramread(ctx, onehot, collapsed_operand, idx);
    results[idx] = hal::reshape(ctx, results[idx], indexed_shape);
  }

  if (slice_size.size() > 1) {
    Shape result_shape(slice_size.begin(), slice_size.end());
    result_shape[0] = 1;
    // Keep indexing deeper
    for (int64_t idx = 0; idx < slice_size[0]; ++idx) {
      results[idx] = hal::reshape(
          ctx, results[idx],
          {results[idx].shape().begin() + 1, results[idx].shape().end()});
      // slice next dim
      results[idx] = SecretDynamicSliceOramImpl(
          ctx, results[idx], {slice_size.begin() + 1, slice_size.end()},
          start_indices.subspan(1));
      results[idx] = hal::reshape(ctx, results[idx], result_shape);
    }
  }

  if (results.size() == 1) {
    return results[0];
  }

  return hal::concatenate(ctx, results, 0);
}

spu::MemRef SecretDynamicSlice(SPUContext *ctx, const spu::MemRef &operand,
                               const Sizes &slice_size,
                               absl::Span<const spu::MemRef> start_indices) {
  // Prune public indexed dimensions
  if (std::any_of(start_indices.begin(), start_indices.end(),
                  [](const spu::MemRef &v) { return v.isPublic(); })) {
    Index start(operand.shape().size(), 0);
    Shape sizes = operand.shape();
    std::vector<spu::MemRef> new_start_indices(start_indices.size());
    auto zero_s = hal::seal(
        ctx, hal::_ring_cast(ctx, hal::zeros(ctx, PT_I64),
                             start_indices[0].eltype().semantic_type()));

    for (size_t rank = 0; rank < operand.shape().size(); ++rank) {
      if (start_indices[rank].isPublic()) {
        const MemRef &idx = start_indices[rank];
        start[rank] = hal::getScalarValue<int64_t>(ctx, idx);
        start[rank] =
            std::min(operand.shape()[rank] - slice_size[rank], start[rank]);
        sizes[rank] = slice_size[rank];
        new_start_indices[rank] = zero_s;
      } else {
        new_start_indices[rank] = start_indices[rank];
      }
    }

    auto pruned_operand = hal::slice(ctx, operand, start, sizes);

    return SecretDynamicSlice(ctx, pruned_operand, slice_size,
                              new_start_indices);
  }
  // Clamp all indices
  auto lower_bound =
      hal::constant(ctx, std::vector<int64_t>(slice_size.size(), 0),
                    {static_cast<int64_t>(slice_size.size())});

  spu::Shape limit = operand.shape();
  for (size_t idx = 0; idx < limit.size(); ++idx) {
    limit[idx] -= slice_size[idx];
  }
  auto upper_bound =
      hal::constant(ctx, limit, {static_cast<int64_t>(slice_size.size())});

  // Reshape from scalar to {1} to make concat happy
  std::vector<spu::MemRef> adjusted_start_indices;
  std::transform(start_indices.cbegin(), start_indices.cend(),
                 std::back_inserter(adjusted_start_indices),
                 [&](const MemRef &x) { return hal::unsqueeze(ctx, x); });

  lower_bound = hal::_ring_cast(
      ctx, lower_bound, adjusted_start_indices[0].eltype().semantic_type());
  upper_bound = hal::_ring_cast(
      ctx, upper_bound, adjusted_start_indices[0].eltype().semantic_type());

  auto adjusted_all_indices =
      hal::clamp(ctx, hal::concatenate(ctx, adjusted_start_indices, 0),
                 lower_bound, upper_bound);

  for (int64_t idx = 0;
       idx < static_cast<int64_t>(adjusted_start_indices.size()); ++idx) {
    adjusted_start_indices[idx] =
        hal::slice(ctx, adjusted_all_indices, {idx}, {1}, {1});
  }

  return SecretDynamicSliceOramImpl(ctx, operand, slice_size,
                                    adjusted_start_indices);
}

spu::MemRef DynamicSlice(SPUContext *ctx, const spu::MemRef &operand,
                         const Sizes &slice_size,
                         absl::Span<const spu::MemRef> start_indices) {
  SPU_ENFORCE_EQ(slice_size.size(), start_indices.size());
  SPU_ENFORCE_EQ(slice_size.size(), operand.shape().size());
  SPU_ENFORCE(!start_indices.empty());
  SPU_ENFORCE(!operand.isComplex());

  if (std::all_of(start_indices.begin(), start_indices.end(),
                  [](const spu::MemRef &v) { return v.isPublic(); })) {
    // Start indices
    Index start_indices_i64(start_indices.size());
    for (const auto &idx : llvm::enumerate(start_indices)) {
      auto v_idx = idx.value();
      start_indices_i64[idx.index()] =
          hal::dump_public_as<int64_t>(ctx, v_idx)[0];
      // Transform start_indices
      // start_indices[i] = clamp(start_indices[i], 0,
      // operand.dimension_size[i]
      // - size_indices[i])
      start_indices_i64[idx.index()] = std::min(
          std::max(start_indices_i64[idx.index()], static_cast<int64_t>(0)),
          operand.shape()[idx.index()] - slice_size[idx.index()]);
    }

    // Strides is always 1
    Strides strides(slice_size.size(), 1);

    return hal::slice(ctx, operand, start_indices_i64, Shape(slice_size),
                      strides);
  }

  return SecretDynamicSlice(ctx, operand, slice_size, start_indices);
}

}  // namespace spu::kernel::hal
