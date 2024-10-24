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

#include <string>
#include <vector>

#include "gflags/gflags.h"
#include "spdlog/spdlog.h"
#include "yacl/link/link.h"

#include "libspu/core/prelude.h"
#include "libspu/psi/bucket_psi.h"
#include "libspu/psi/utils/resource.h"

DEFINE_string(input_path, "", "input csv file path");
DEFINE_string(
    psi_keys, "id",
    "which keys in input file should be used in psi, separated by \",\"");
DEFINE_string(output_path, "psi_demo_out.csv", "output csv file path");
DEFINE_string(
    party_ips, "127.0.0.1:9307,127.0.0.1:9308",
    "all parties ip, separated by \",\", the order must be consistent");
DEFINE_uint32(rank, 0, "self rank in link::Context");
DEFINE_string(psi_protocol, "ECDH_PSI_2PC", "psi protocol, see PsiType");
DEFINE_bool(output_sort, true, "whether output file should be sorted");
DEFINE_bool(ic_mode, false, "run in interconnection mode");

std::shared_ptr<yacl::link::Context> CreateLinkContext(
    const std::string& party_ips, size_t self_rank) {
  std::vector<std::string> ip_list = absl::StrSplit(party_ips, ',');
  SPU_ENFORCE(ip_list.size() > 1);

  yacl::link::ContextDesc ctx_desc;
  for (size_t i = 0; i < ip_list.size(); ++i) {
    ctx_desc.parties.push_back({std::to_string(i), ip_list[i]});
  }
  if (FLAGS_ic_mode) {
    ctx_desc.brpc_channel_protocol = "h2:grpc";
  }

  return yacl::link::FactoryBrpc().CreateContext(ctx_desc, self_rank);
}

std::vector<std::string> GetPsiKeys(const std::string& psi_keys) {
  return absl::StrSplit(psi_keys, ',');
}

int main(int argc, char* argv[]) {
  gflags::AllowCommandLineReparsing();
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  try {
    auto link_ctx = CreateLinkContext(FLAGS_party_ips, FLAGS_rank);
    link_ctx->ConnectToMesh();

    auto fields = GetPsiKeys(FLAGS_psi_keys);

    spu::psi::BucketPsiConfig config;
    config.mutable_input_params()->set_path(FLAGS_input_path);
    config.mutable_input_params()->mutable_select_fields()->Add(fields.begin(),
                                                                fields.end());
    config.mutable_input_params()->set_precheck(false);
    config.mutable_output_params()->set_path(FLAGS_output_path);
    config.mutable_output_params()->set_need_sort(FLAGS_output_sort);
    spu::psi::PsiType psi_type;
    SPU_ENFORCE(spu::psi::PsiType_Parse(FLAGS_psi_protocol, &psi_type),
                "Parse psi_protocol {} fail", FLAGS_psi_protocol);
    config.set_psi_type(psi_type);
    config.set_broadcast_result(true);
    config.set_curve_type(spu::psi::CurveType::CURVE_25519);

    SPDLOG_INFO("Run BucketPsi with config: {}", config.ShortDebugString());
    SPDLOG_INFO("Ic mode: {}", FLAGS_ic_mode);
    spu::psi::BucketPsi ctx(config, link_ctx, FLAGS_ic_mode);
    auto report = ctx.Run();

    SPDLOG_INFO("psi intersection_count={}, original_count={}",
                report.intersection_count(), report.original_count());

    double peak_mem_kb = 0;
    try {
      peak_mem_kb = spu::psi::GetPeakKbMemUsage();
    } catch (const std::exception& ex) {
      SPDLOG_WARN("GetPeakKbMemUsage throw exception {}", ex.what());
    }
    SPDLOG_INFO("psi finished, peak mem usage {} GB",
                peak_mem_kb / 1024 / 1024);

  } catch (const std::exception& ex) {
    SPDLOG_ERROR("Failed to run psi, error={}", ex.what());
    return -1;
  }

  return 0;
}
