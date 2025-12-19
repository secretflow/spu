// Copyright 2025 Ant Group Co., Ltd.
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

#include "libspu/kernel/hal/group_by_agg.h"

#include "libspu/core/trace.h"
#include "libspu/kernel/hal/fxp_base.h"
#include "libspu/kernel/hal/permute.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/prot_wrapper.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/hal/utils.h"

namespace spu::kernel::hal {

namespace {

inline int64_t _get_owner(const Value &x) {
  return x.storage_type().as<Private>()->owner();
}

inline bool _has_same_owner(const Value &x, const Value &y) {
  return _get_owner(x) == _get_owner(y);
}

Value _group_mark(SPUContext *ctx, absl::Span<Value const> inputs,
                  bool end_group_mark) {
  if (inputs[0].isPublic()) {
    SPU_ENFORCE(std::all_of(inputs.begin(), inputs.end(),
                            [](const Value &v) { return v.isPublic(); }),
                "inputs should be all public");
    return _group_mark_p(ctx, inputs, end_group_mark);
  } else if (inputs[0].isPrivate()) {
    SPU_ENFORCE(std::all_of(inputs.begin(), inputs.end(),
                            [&inputs](const Value &v) {
                              return v.isPrivate() &&
                                     _has_same_owner(v, inputs[0]);
                            }),
                "inputs should have a same owner");
    return _group_mark_v(ctx, inputs, end_group_mark);
  } else if (inputs[0].isSecret()) {
    SPU_THROW("Secret group mark computation is currently not supported");
  } else {
    SPU_THROW("should not be here");
  }
}

// circular right shift the input with 1 position
// Note: the first element will be padded with 0
Value _circular_right_shift_1d(SPUContext *ctx, const Value &input) {
  const auto n = input.shape()[0];
  auto padding_value = hal::zeros(ctx, input.dtype(), {1});
  return hal::concatenate(
      ctx, {padding_value, hal::slice(ctx, input, {0}, {n - 1})}, 0);
}

// Warning: after this call, payloads can not be used anymore
void _inplace_merge_keys_and_payloads(std::vector<Value> &keys,
                                      std::vector<Value> &&payloads) {
  keys.reserve(keys.size() + payloads.size());
  keys.insert(keys.end(), std::make_move_iterator(payloads.begin()),
              std::make_move_iterator(payloads.end()));
}

}  // namespace

std::vector<Value> private_groupby_sum_1d(
    SPUContext *ctx, absl::Span<spu::Value const> keys,
    absl::Span<spu::Value const> payloads) {
  SPU_TRACE_HAL_DISP(ctx, keys.size(), payloads.size());

  auto private_perm = gen_inv_perm_1d(ctx, keys, SortDirection::Ascending);
  auto sorted_keys = apply_inv_permute_1d(ctx, keys, private_perm);
  auto group_marks =
      _group_mark(ctx, absl::MakeSpan(sorted_keys), /*end_group_mark=*/true);

  // the permutation that makes valid key appear first
  auto group_mark_perm =
      gen_inv_perm_1d(ctx, {group_marks}, SortDirection::Descending);
  auto output_order_keys =
      apply_inv_permute_1d(ctx, sorted_keys, group_mark_perm);

  // inv_perm_xv called here, x relies on the visibility of payloads
  auto permuted_payloads = apply_inv_permute_1d(ctx, payloads, private_perm);

  std::vector<Value> prefix_sum_payloads;
  prefix_sum_payloads.reserve(payloads.size());

  // use zero to mark the temporay dummy value
  for (uint64_t i = 0; i < permuted_payloads.size(); ++i) {
    auto w = associative_scan(hal::add, ctx, permuted_payloads[i]);
    // inv_perm_xv called here
    //
    // multiple calls of inv_perm_sv(value) = single call of
    // inv_perm_sv(vector[value]), so we just do it in the loop
    auto y = apply_inv_permute_1d(ctx, {w}, group_mark_perm)[0];
    auto s = hal::_sub(ctx, y, _circular_right_shift_1d(ctx, y));
    prefix_sum_payloads.push_back(s.setDtype(payloads[i].dtype()));
  }

  // now, we always return both keys and payloads
  _inplace_merge_keys_and_payloads(output_order_keys,
                                   std::move(prefix_sum_payloads));
  return output_order_keys;
}

// most of the code is same as private_groupby_sum_1d except we need to compute
// a helper counter to divide the sum results
std::vector<Value> private_groupby_avg_1d(SPUContext *ctx,
                                          absl::Span<spu::Value const> keys,
                                          absl::Span<spu::Value const> payloads,
                                          bool unsafe_drop_rest) {
  SPU_TRACE_HAL_DISP(ctx, keys.size(), payloads.size());

  auto private_perm = gen_inv_perm_1d(ctx, keys, SortDirection::Ascending);
  auto sorted_keys = apply_inv_permute_1d(ctx, keys, private_perm);
  auto group_marks =
      _group_mark(ctx, absl::MakeSpan(sorted_keys), /*end_group_mark=*/true);
  SPU_ENFORCE(group_marks.shape().ndim() == 1, "group marks should be 1d");

  // the permutation that makes valid key appear first
  auto group_mark_perm =
      gen_inv_perm_1d(ctx, {group_marks}, SortDirection::Descending);
  auto output_order_keys =
      apply_inv_permute_1d(ctx, sorted_keys, group_mark_perm);

  // compute the group sizes
  auto one_payload = hal::constant(ctx, 1.0F, DT_F32, group_marks.shape());
  auto one_prefix_sum = associative_scan(hal::add, ctx, one_payload);
  // inv_perm_pv called here, free
  one_prefix_sum =
      apply_inv_permute_1d(ctx, {one_prefix_sum}, group_mark_perm)[0];
  auto count = hal::_sub(ctx, one_prefix_sum,
                         _circular_right_shift_1d(ctx, one_prefix_sum))
                   .setDtype(DT_F32);

  // drop the rest count for better performance
  Value unique_key_value_count;
  int64_t key_count = 0;
  if (unsafe_drop_rest) {
    auto group_dtype = ctx->getField() == FieldType::FM32 ? DT_I32 : DT_I64;
    unique_key_value_count = hal::associative_reduce(
        hal::add, ctx, group_marks.setDtype(group_dtype));

    if (!unique_key_value_count.isPublic()) {
      unique_key_value_count = hal::reveal(ctx, unique_key_value_count);
    }

    // INFO LEAKAGE HERE: reveal the unique key count to all parties
    key_count = getScalarValue<int64_t>(ctx, unique_key_value_count);
    SPU_ENFORCE(
        key_count > 0,
        "key_count must be greater than 0 when unsafe_drop_rest is true");

    // update the shapes
    count = hal::slice(ctx, count, {0}, {key_count});
    for (auto &k : output_order_keys) {
      k = hal::slice(ctx, k, {0}, {key_count});
    }
  }

  // inv_perm_xv called here, x relies on the visibility of payloads
  auto permuted_payloads = apply_inv_permute_1d(ctx, payloads, private_perm);

  std::vector<Value> prefix_sum_payloads;
  prefix_sum_payloads.reserve(payloads.size());

  // use zero to mark the temporay dummy value
  for (uint64_t i = 0; i < permuted_payloads.size(); ++i) {
    auto w = associative_scan(hal::add, ctx, permuted_payloads[i]);
    // inv_perm_xv called here
    //
    // multiple calls of inv_perm_sv(value) = single call of
    // inv_perm_sv(vector[value]), so we just do it in the loop
    auto y = apply_inv_permute_1d(ctx, {w}, group_mark_perm)[0];
    auto s = hal::_sub(ctx, y, _circular_right_shift_1d(ctx, y))
                 .setDtype(payloads[i].dtype());
    if (unsafe_drop_rest) {
      s = hal::slice(ctx, s, {0}, {key_count});
    }

    auto dtype = std::max(count.dtype(), s.dtype());
    // count is always positive; private or public
    count = dtype_cast(ctx, count, dtype);
    s = dtype_cast(ctx, s, dtype);

    // we use mul (will do dtype dispatch inside) + internal reciprocal to do
    // division to increase performance at the cost of slight precision loss
    // when the count and sums are both large.
    prefix_sum_payloads.push_back(
        hal::mul(ctx, s, detail::reciprocal_goldschmidt_positive(ctx, count)));
  }

  // now, we always return both keys and payloads
  _inplace_merge_keys_and_payloads(output_order_keys,
                                   std::move(prefix_sum_payloads));
  return output_order_keys;
}

}  // namespace spu::kernel::hal
