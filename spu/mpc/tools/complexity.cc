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

#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <string>

#include "google/protobuf/util/json_util.h"
#include "llvm/Support/CommandLine.h"
#include "spdlog/spdlog.h"

#include "spu/mpc/factory.h"
#include "spu/mpc/interfaces.h"
#include "spu/mpc/object.h"
#include "spu/mpc/util/communicator.h"
#include "spu/mpc/util/simulate.h"

#include "spu/mpc/tools/complexity.pb.h"

namespace spu::mpc {

internal::SingleComplexityReport dumpComplexityReport(
    const std::string& protocol_str, ProtocolKind protocol, size_t party_cnt) {
  internal::SingleComplexityReport single_report;
  single_report.set_protocol(protocol_str);

  // the interested kernel whitelist.
  const std::vector<std::string> kWhitelist = {
      "a2b",    "b2a",    "a2p",    "b2p",     "add_bb",  "add_aa",
      "add_ap", "mul_aa", "mul_ap", "mmul_aa", "mmul_ap", "truncpr_a",
      "xor_bb", "xor_bp", "and_bb", "and_bp",
  };
  std::cout << protocol_str << std::endl;
  // print header
  fmt::print("{:<15}, {:<20}, {:<20}\n", "name", "latency", "comm");

  util::simulate(
      party_cnt, [&](const std::shared_ptr<yasl::link::Context>& lctx) -> void {
        auto prot = Factory::CreateCompute(protocol, lctx);
        if (lctx->Rank() != 0) {
          return;
        }

        for (auto name : kWhitelist) {
          auto* kernel = prot->getKernel(name);
          auto latency = kernel->latency();
          auto comm = kernel->comm();

          const std::string latency_str = latency ? latency->expr() : "TODO";
          const std::string comm_str = comm ? comm->expr() : "TODO";

          fmt::print("{:<15}, {:<20}, {:<20}\n", name, latency_str, comm_str);

          auto* entry = single_report.add_entries();
          entry->set_kernel(name);
          entry->set_comm(comm_str);
          entry->set_latency(latency_str);
        }

        return;
      });

  return single_report;
}

}  // namespace spu::mpc

llvm::cl::opt<std::string> OutputFilename(
    "out", llvm::cl::desc("Specify output json filename"),
    llvm::cl::value_desc("filename"));

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  // suppress all link logs.
  spdlog::set_level(spdlog::level::off);

  spu::mpc::internal::ComplexityReport report;

  *(report.add_reports()) =
      spu::mpc::dumpComplexityReport("Semi2k", spu::ProtocolKind::SEMI2K, 2);
  *(report.add_reports()) =
      spu::mpc::dumpComplexityReport("Aby3", spu::ProtocolKind::ABY3, 3);

  if (!OutputFilename.empty()) {
    std::string json;
    google::protobuf::util::JsonPrintOptions json_options;
    json_options.preserve_proto_field_names = true;

    YASL_ENFORCE(
        google::protobuf::util::MessageToJsonString(report, &json, json_options)
            .ok());

    std::ofstream out(OutputFilename.getValue());

    out << json;
  }
}
