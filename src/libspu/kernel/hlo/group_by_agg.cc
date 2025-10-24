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

#include "libspu/kernel/hlo/group_by_agg.h"

#include "magic_enum.hpp"

namespace spu::kernel::hlo {

namespace {

inline int64_t _get_owner(const Value& x) {
  return x.storage_type().as<Private>()->owner();
}

bool _all_pub_or_pri_with_same_owner(absl::Span<spu::Value const> values) {
  bool flag = true;

  int64_t pri_rank = -1;
  for (const auto& v : values) {
    if (v.isPrivate()) {
      if (pri_rank == -1) {
        pri_rank = _get_owner(v);
      } else if (pri_rank != _get_owner(v)) {
        flag = false;
        break;
      }
    } else if (v.isSecret()) {
      flag = false;
      break;
    }
  }
  return flag;
}
}  // namespace

std::vector<Value> GroupByAgg(SPUContext* ctx,
                              absl::Span<spu::Value const> keys,
                              absl::Span<spu::Value const> payloads,
                              AggFunc agg_func, absl::Span<int64_t> valid_bits,
                              const GroupByAggOptions& options) {
  // normal sanity checks
  // TODO(zjj): support more valid_bits hint after radix sort supporting.
  SPU_ENFORCE(valid_bits.size() <= 1);
  {
    SPU_ENFORCE(!keys.empty(), "keys should not be empty");
    SPU_ENFORCE(keys[0].shape().ndim() == 1,
                "Keys should be 1-d but actually have {} dimensions",
                keys[0].shape().ndim());
    SPU_ENFORCE(std::all_of(keys.begin(), keys.end(),
                            [&keys](const spu::Value& v) {
                              return v.shape() == keys[0].shape();
                            }),
                "Keys shape mismatched");

    SPU_ENFORCE(!payloads.empty(), "payloads should not be empty");
    SPU_ENFORCE(payloads[0].shape().ndim() == 1,
                "Payloads should be 1-d but actually have {} dimensions",
                payloads[0].shape().ndim());
    SPU_ENFORCE(std::all_of(payloads.begin(), payloads.end(),
                            [&payloads](const spu::Value& v) {
                              return v.shape() == payloads[0].shape();
                            }),
                "Payloads shape mismatched");

    SPU_ENFORCE(keys[0].numel() == payloads[0].numel(),
                "Keys and payloads shape mismatched");

    SPU_ENFORCE(std::all_of(keys.begin(), keys.end(),
                            [&keys](const spu::Value& v) {
                              return v.vtype() == keys[0].vtype();
                            }),
                "Keys visibility mismatched");
    SPU_ENFORCE(std::all_of(payloads.begin(), payloads.end(),
                            [&payloads](const spu::Value& v) {
                              return v.vtype() == payloads[0].vtype();
                            }),
                "Payloads visibility mismatched");
  }
  // empty
  if (keys[0].numel() == 0) {
    std::vector<Value> rets;
    rets.reserve(keys.size() + payloads.size());
    rets.insert(rets.end(), keys.begin(), keys.end());
    rets.insert(rets.end(), payloads.begin(), payloads.end());
    return rets;
  }

  // TODO(zjj): only support the private groupby sum for now.
  // maybe we should implement a switch function to dispatch different
  // groupby-agg implementations.
  switch (agg_func) {
    case AggFunc::Sum:
      if ((options.mode != GroupByAggMode::PrefixSumMode) &&
          (options.output_format == OutputFormat::OutputOrder)) {
        SPU_ENFORCE(_all_pub_or_pri_with_same_owner(keys),
                    "keys should be all public or private with the same owner");
        return hal::private_groupby_sum_1d(ctx, keys, payloads);
      } else {
        SPU_THROW(
            "groupby sum with mode {} and output format {} is not "
            "supported now",
            magic_enum::enum_name(options.mode),
            magic_enum::enum_name(options.output_format));
      }
    default:
      SPU_THROW("groupby agg func {} is not supported now",
                magic_enum::enum_name(agg_func));
  }

  SPU_THROW("should not reach here");
}

}  // namespace spu::kernel::hlo
