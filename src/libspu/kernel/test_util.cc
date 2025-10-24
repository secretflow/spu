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

#include "libspu/kernel/test_util.h"

#include "libspu/core/config.h"
#include "libspu/core/encoding.h"
#include "libspu/core/trace.h"
#include "libspu/kernel/hal/constants.h"  // bad reference
#include "libspu/mpc/factory.h"

namespace spu::kernel::test {

SPUContext makeSPUContext(RuntimeConfig config,
                          const std::shared_ptr<yacl::link::Context> &lctx) {
  populateRuntimeConfig(config);

  SPUContext ctx(config, lctx);
  mpc::Factory::RegisterProtocol(&ctx, lctx);

  return ctx;
}

SPUContext makeSPUContext(ProtocolKind prot_kind, FieldType field,
                          const std::shared_ptr<yacl::link::Context> &lctx) {
  RuntimeConfig cfg;
  cfg.protocol = prot_kind;
  cfg.field = field;
  cfg.enable_action_trace = false;

  return makeSPUContext(cfg, lctx);
}

Value makeValue(SPUContext *ctx, PtBufferView init, Visibility vtype,
                DataType dtype, const Shape &shape, int64_t owner) {
  if (dtype == DT_INVALID) {
    dtype = getEncodeType(init.pt_type);
  }
  auto res = hal::constant(ctx, init, dtype, shape);
  switch (vtype) {
    case VIS_PUBLIC:
      return res;
    case VIS_SECRET:
      return hal::_p2s(ctx, res).setDtype(res.dtype());
    case VIS_PRIVATE:
      SPU_ENFORCE(owner >= 0, "owner rank should be >=0 for private value");
      return hal::_p2v(ctx, res, owner).setDtype(res.dtype());
    default:
      SPU_THROW("not supported vtype={}", vtype);
  }
}

namespace {
struct ActionKey {
  std::string_view name;
  int64_t flag;
  bool operator<(const ActionKey &other) const {
    return std::tie(name, flag) < std::tie(other.name, other.flag);
  }
};

struct ActionStats {
  // number of actions executed.
  size_t count = 0;
  // total duration time.
  Duration total_time = {};
  // total send bytes.
  size_t send_bytes = 0;
  // total recv bytes.
  size_t recv_bytes = 0;
  // total send actions.
  size_t send_actions = 0;
  // total recv actions.
  size_t recv_actions = 0;

  inline double getTotalTimeInSecond() const {
    return std::chrono::duration_cast<std::chrono::duration<double>>(total_time)
        .count();
  }
};
}  // namespace

void printProfileData(SPUContext *sctx) {
  std::map<ActionKey, ActionStats> stats;

  const auto &tracer = GET_TRACER(sctx);
  const auto &records = tracer->getProfState()->getRecords();

  for (const auto &rec : records) {
    auto &stat = stats[{rec.name, rec.flag}];
    stat.count++;
    stat.total_time +=
        std::chrono::duration_cast<Duration>(rec.end - rec.start);
    stat.send_bytes += (rec.send_bytes_end - rec.send_bytes_start);
    stat.recv_bytes += (rec.recv_bytes_end - rec.recv_bytes_start);
    stat.send_actions += (rec.send_actions_end - rec.send_actions_start);
    stat.recv_actions += (rec.recv_actions_end - rec.recv_actions_start);
  }

  static std::map<int64_t, std::string> kModules = {
      {TR_HLO, "HLO"}, {TR_HAL, "HAL"}, {TR_MPC, "MPC"}};

  for (const auto &[mod_flag, mod_name] : kModules) {
    if ((tracer->getFlag() & mod_flag) == 0) {
      continue;
    }
    double total_time = 0.0;
    std::vector<ActionKey> sorted_by_time;
    for (const auto &[key, stat] : stats) {
      if ((key.flag & mod_flag) != 0) {
        total_time += stat.getTotalTimeInSecond();
        sorted_by_time.emplace_back(key);
      }
    }

    std::sort(sorted_by_time.begin(), sorted_by_time.end(),
              [&](const auto &k0, const auto &k1) {
                return stats.find(k0)->second.getTotalTimeInSecond() >
                       stats.find(k1)->second.getTotalTimeInSecond();
              });

    SPDLOG_INFO("{} profiling: total time {}", mod_name, total_time);
    for (const auto &key : sorted_by_time) {
      const auto &stat = stats.find(key)->second;
      SPDLOG_INFO(
          "- {}, executed {} times, duration {}s, send bytes {} recv "
          "bytes {}, send actions {}, recv actions {}",
          key.name, stat.count, stat.getTotalTimeInSecond(), stat.send_bytes,
          stat.recv_bytes, stat.send_actions, stat.recv_actions);
    }
  }
}

}  // namespace spu::kernel::test
