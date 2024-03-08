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

#include "libspu/kernel/hlo/indexing.h"

#include <cstring>

#include "llvm/ADT/STLExtras.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/value.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/kernel/hal/shape_ops.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/hal/utils.h"
#include "libspu/kernel/hlo/basic_unary.h"
#include "libspu/kernel/hlo/const.h"
#include "libspu/kernel/hlo/utils.h"

// forward
namespace spu::kernel::hal::detail {
void hintNumberOfBits(const Value &a, size_t nbits);
}

namespace {
spu::Value SecretLinearUpdateIndexing(spu::SPUContext *ctx,
                                      const spu::Value &operand,
                                      const spu::Value &update,
                                      const spu::Value &linear_idx) {
  // TODO: Consider utilizing DLP to improve performance
  SPU_ENFORCE(operand.shape().size() == 1, "operand must be a 1D tensor");
  SPU_ENFORCE(linear_idx.numel() == 1, "index must be a 1D indexing");
  SPU_ENFORCE(update.numel() == 1, "update must be a scalar");

  // Basic idea here:
  // eq(iota, idx) * update + !eq(iota, idx) * operand
  auto linear_idx_broadcasted =
      spu::kernel::hal::broadcast_to(ctx, linear_idx, {operand.numel()}, {});
  spu::Value idx_iota =
      spu::kernel::hal::iota(ctx, spu::DT_I64, operand.numel());
  auto mask = spu::kernel::hal::equal(ctx, linear_idx_broadcasted, idx_iota);

  auto c0 = spu::kernel::hlo::Constant(ctx, static_cast<int64_t>(0), {});
  auto i0 = spu::kernel::hal::dtype_cast(ctx, c0, operand.dtype());

  auto reverse_mask = spu::kernel::hlo::Not(ctx, mask);

  auto broadcast_update =
      spu::kernel::hal::broadcast_to(ctx, update, operand.shape(), {0});

  return spu::kernel::hal::add(
      ctx, spu::kernel::hal::mul(ctx, operand, reverse_mask),
      spu::kernel::hal::mul(ctx, broadcast_update, mask));
}

std::vector<spu::Value> ClampAndFlattenIndex(
    spu::SPUContext *ctx, absl::Span<const spu::Value> start_indices,
    const spu::Shape &iterate_shape, const spu::Shape &limit_shape) {
  // Transform start_indices
  // start_indices[i] = clamp(start_indices[i], 0, operand.dimension_size[i]
  // - size_indices[i])

  std::vector<spu::Value> clamped_start(start_indices.size());
  {
    std::vector<spu::Value> reshaped_start_indices;
    std::transform(start_indices.cbegin(), start_indices.cend(),
                   std::back_inserter(reshaped_start_indices),
                   [&](const spu::Value &x) {
                     return spu::kernel::hal::unsqueeze(ctx, x);
                   });

    auto concat_idx =
        spu::kernel::hal::concatenate(ctx, reshaped_start_indices, 0);
    auto lower_bound = spu::kernel::hlo::Constant(ctx, static_cast<int64_t>(0),
                                                  concat_idx.shape());
    lower_bound =
        spu::kernel::hal::dtype_cast(ctx, lower_bound, concat_idx.dtype());

    std::vector<int64_t> upper_bound_pt(start_indices.size());
    for (size_t idx = 0; idx < upper_bound_pt.size(); ++idx) {
      upper_bound_pt[idx] = limit_shape[idx] - iterate_shape[idx];
    }
    auto upper_bound =
        spu::kernel::hlo::Constant(ctx, upper_bound_pt, concat_idx.shape());
    upper_bound =
        spu::kernel::hal::dtype_cast(ctx, upper_bound, concat_idx.dtype());

    auto c = spu::kernel::hal::clamp(ctx, concat_idx, lower_bound, upper_bound);
    for (int64_t idx = 0; idx < static_cast<int64_t>(clamped_start.size());
         ++idx) {
      clamped_start[idx] = spu::kernel::hal::squeeze(
          ctx, spu::kernel::hal::slice(ctx, c, {idx}, {idx + 1}, {1}));
    }
  }

  // Now flatten start index
  auto linear_idx =
      spu::kernel::hlo::Constant(ctx, static_cast<int64_t>(0), {});
  int64_t stride = 1;
  for (int64_t idx = iterate_shape.size() - 1; idx >= 0; --idx) {
    linear_idx = spu::kernel::hal::add(
        ctx, linear_idx,
        spu::kernel::hal::mul(ctx, clamped_start[idx],
                              spu::kernel::hlo::Constant(ctx, stride, {})));
    stride *= limit_shape[idx];
  }

  // Now compute offsets of each index
  spu::Index base(iterate_shape.size(), 0);
  spu::Index incr(iterate_shape.size(), 1);

  spu::Index flatten_idx;
  spu::kernel::forEachIndex(
      limit_shape, base, iterate_shape, incr,
      [&flatten_idx, &limit_shape](const spu::Index &idx) {
        flatten_idx.emplace_back(spu::flattenIndex(idx, limit_shape));
      });

  auto num_index = iterate_shape.numel();
  std::vector<spu::Value> linear_indices;
  linear_indices.reserve(num_index);
  auto added = spu::kernel::hal::add(
      ctx,
      spu::kernel::hal::broadcast_to(
          ctx, spu::kernel::hal::unsqueeze(ctx, linear_idx), {num_index}, {0}),
      spu::kernel::hlo::Constant(ctx, flatten_idx, {num_index}));
  for (int64_t idx = 0; idx < num_index; ++idx) {
    linear_indices.emplace_back(spu::kernel::hal::squeeze(
        ctx, spu::kernel::hal::slice(ctx, added, {idx}, {idx + 1}, {1})));
  }
  return linear_indices;
}

}  // namespace

namespace spu::kernel::hlo {

spu::Value DynamicUpdateSlice(SPUContext *ctx, const spu::Value &operand,
                              const spu::Value &update,
                              absl::Span<const spu::Value> start_indices) {
  SPU_ENFORCE(!operand.isComplex());
  // Basic idea here, get a ref slice and update the whole slice..
  SPU_ENFORCE_EQ(start_indices.size(), operand.shape().size());
  SPU_ENFORCE_EQ(start_indices.size(), update.shape().size());
  SPU_ENFORCE(!start_indices.empty());

  if (start_indices[0].isSecret()) {
    // flatten first
    spu::Value flattened_operand =
        hal::reshape(ctx, operand, {operand.numel()});

    spu::Value flattened_update = hal::reshape(ctx, update, {update.numel()});

    auto flattened_indices = ClampAndFlattenIndex(
        ctx, start_indices, update.shape(), operand.shape());

    spu::Value ret = flattened_operand;

    for (int64_t n = 0; n < static_cast<int64_t>(flattened_indices.size());
         ++n) {
      auto update_slice = hal::slice(ctx, flattened_update, {n}, {n + 1}, {1});
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
      start_indices_i64[idx.index()] = getIndices(ctx, v_idx)[0];
      // Transform start_indices
      // start_indices[i] = clamp(start_indices[i], 0, operand.dimension_size[i]
      // - update.dimension_size[i])
      start_indices_i64[idx.index()] = std::min(
          std::max(start_indices_i64[idx.index()], static_cast<int64_t>(0)),
          operand.shape()[idx.index()] - update.shape()[idx.index()]);
    }

    return UpdateSlice(ctx, operand, update, start_indices_i64);
  }
}

spu::Value UpdateSlice(SPUContext *ctx, const spu::Value &in,
                       const spu::Value &update, const Index &start_indices) {
  return hal::update_slice(ctx, in, update, start_indices);
}

spu::Value SecretDynamicSliceImpl(SPUContext *ctx, const spu::Value &operand,
                                  const Sizes &slice_size,
                                  absl::Span<const spu::Value> start_indices) {
  if (slice_size[0] == operand.shape()[0]) {
    if (slice_size.size() == 1) {
      return operand;
    }

    // Full dimension
    Index start(operand.shape().size(), 0);
    Index limit(operand.shape().begin(), operand.shape().end());
    Strides strides(operand.shape().size(), 1);

    std::vector<spu::Value> results(operand.shape()[0]);
    for (int64_t idx = 0; idx < operand.shape()[0]; ++idx) {
      start[0] = idx;
      limit[0] = idx + 1;
      // Slice one...
      auto sliced = hal::slice(ctx, operand, start, limit, strides);
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
  spu::Value mask;
  spu::Value idx_iota =
      hal::iota(ctx, start_indices[0].dtype(), operand.shape()[0]);

  mask = hal::equal(ctx,
                    hal::broadcast_to(ctx, start_indices[0], idx_iota.shape()),
                    idx_iota);

  if (slice_size[0] >= 1) {
    auto pad_value = hal::seal(ctx, hal::constant(ctx, false, mask.dtype()));
    pad_value = hal::_cast_type(ctx, pad_value, mask.storage_type());
    mask = hal::pad(ctx, mask, pad_value, {slice_size[0]}, {0}, {0});
    // FIXME(juhou): we should avoid setting the BShr here
    // However mask.storage_type().as<BShare>->nbits() is not 1 after the
    // padding. We implicitly set mask as a 1-bit BShr so that the following
    // hal::matmul can use a much lighter B2A proc for both ABY3 and CHEETAH.
    hal::detail::hintNumberOfBits(mask, 1);
  }

  // foreach
  std::vector<spu::Value> results(slice_size[0]);

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
                   {mask.numel() - idx}, {1});
    mask_slice = hal::unsqueeze(ctx, mask_slice);

    results[idx] = hal::matmul(ctx, mask_slice, collapsed_operand);

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

spu::Value SecretDynamicSlice(SPUContext *ctx, const spu::Value &operand,
                              const Sizes &slice_size,
                              absl::Span<const spu::Value> start_indices) {
  // Prune public indexed dimensions
  if (std::any_of(start_indices.begin(), start_indices.end(),
                  [](const spu::Value &v) { return v.isPublic(); })) {
    Index start(operand.shape().size(), 0);
    Index limit(operand.shape());
    std::vector<spu::Value> new_start_indices(start_indices.size());
    auto zero_s =
        hal::seal(ctx, hal::zeros(ctx, start_indices.front().dtype()));

    for (size_t rank = 0; rank < operand.shape().size(); ++rank) {
      if (start_indices[rank].isPublic()) {
        start[rank] = hal::getScalarValue<int64_t>(ctx, start_indices[rank]);
        start[rank] = std::min(limit[rank] - slice_size[rank], start[rank]);
        limit[rank] = start[rank] + slice_size[rank];
        new_start_indices[rank] = zero_s;
      } else {
        new_start_indices[rank] = start_indices[rank];
      }
    }

    auto pruned_operand = hal::slice(ctx, operand, start, limit);

    return SecretDynamicSlice(ctx, pruned_operand, slice_size,
                              new_start_indices);
  }
  // Clamp all indices
  auto lower_bound =
      hlo::Constant(ctx, std::vector<int64_t>(slice_size.size(), 0),
                    {static_cast<int64_t>(slice_size.size())});

  spu::Shape limit = operand.shape();
  for (size_t idx = 0; idx < limit.size(); ++idx) {
    limit[idx] -= slice_size[idx];
  }
  auto upper_bound =
      hlo::Constant(ctx, limit, {static_cast<int64_t>(slice_size.size())});

  // Cast to proper type
  lower_bound = hal::dtype_cast(ctx, lower_bound, start_indices[0].dtype());
  upper_bound = hal::dtype_cast(ctx, upper_bound, start_indices[0].dtype());

  // Reshape from scalar to {1} to make concat happy
  std::vector<spu::Value> adjusted_start_indices;
  std::transform(start_indices.cbegin(), start_indices.cend(),
                 std::back_inserter(adjusted_start_indices),
                 [&](const Value &x) { return hal::unsqueeze(ctx, x); });

  auto adjusted_all_indices =
      hal::clamp(ctx, hal::concatenate(ctx, adjusted_start_indices, 0),
                 lower_bound, upper_bound);

  for (int64_t idx = 0;
       idx < static_cast<int64_t>(adjusted_start_indices.size()); ++idx) {
    adjusted_start_indices[idx] =
        hal::slice(ctx, adjusted_all_indices, {idx}, {idx + 1}, {1});
  }

  return SecretDynamicSliceImpl(ctx, operand, slice_size,
                                adjusted_start_indices);
}

spu::Value DynamicSlice(SPUContext *ctx, const spu::Value &operand,
                        const Sizes &slice_size,
                        absl::Span<const spu::Value> start_indices) {
  SPU_ENFORCE_EQ(slice_size.size(), start_indices.size());
  SPU_ENFORCE_EQ(slice_size.size(), operand.shape().size());
  SPU_ENFORCE(!start_indices.empty());
  SPU_ENFORCE(!operand.isComplex());

  if (std::all_of(start_indices.begin(), start_indices.end(),
                  [](const spu::Value &v) { return v.isPublic(); })) {
    // Start indices
    Index start_indices_i64(start_indices.size());
    for (const auto &idx : llvm::enumerate(start_indices)) {
      auto v_idx = idx.value();
      start_indices_i64[idx.index()] = getIndices(ctx, v_idx)[0];
      // Transform start_indices
      // start_indices[i] = clamp(start_indices[i], 0, operand.dimension_size[i]
      // - size_indices[i])
      start_indices_i64[idx.index()] = std::min(
          std::max(start_indices_i64[idx.index()], static_cast<int64_t>(0)),
          operand.shape()[idx.index()] - slice_size[idx.index()]);
    }

    // Limit
    Index limit(start_indices_i64);
    for (size_t idx = 0; idx < limit.size(); ++idx) {
      limit[idx] += slice_size[idx];
    }

    // Strides is always 1
    Strides strides(limit.size(), 1);

    return hal::slice(ctx, operand, start_indices_i64, limit, strides);
  }

  return SecretDynamicSlice(ctx, operand, slice_size, start_indices);
}

spu::Value FilterByMask(SPUContext *, const spu::Value &operand,
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

  return Value(operand.data().linear_gather(indices), operand.dtype());
}

spu::Value LinearGather(SPUContext *, const spu::Value &in,
                        const Index &indices) {
  return Value(in.data().linear_gather(indices), in.dtype());
}

void LinearScatterInPlace(SPUContext *ctx, spu::Value &in,
                          const spu::Value &update, const Index &indices) {
  if (in.data().eltype() != update.data().eltype()) {
    auto common_type =
        hal::_common_type(ctx, in.data().eltype(), update.data().eltype());
    in = hal::_cast_type(ctx, in, common_type).setDtype(in.dtype());
    LinearScatterInPlace(
        ctx, in,
        hal::_cast_type(ctx, update, common_type).setDtype(update.dtype()),
        indices);
    return;
  }
  in.data().linear_scatter(update.data(), indices);
}

}  // namespace spu::kernel::hlo
