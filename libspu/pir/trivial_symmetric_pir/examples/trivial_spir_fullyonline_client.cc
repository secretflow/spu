// build 'trivial_spir_fullyonline_client'
// > bazel build
// //libspu/pir/trivial_symmetric_pir/examples:trivial_spir_fullyonline_client
// -c opt
//
// To run the example, start terminals:
// ./bazel-bin/libspu/pir/trivial_symmetric_pir/examples/trivial_spir_fullyonline_client -in_path ./examples/data/pir_client_data.csv -out_path ./dump/client_query_output.csv -key_columns id

#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

#include "absl/strings/str_split.h"
#include "examples/cpp/utils.h"
#include "spdlog/spdlog.h"
#include "yacl/link/link.h"

#include "libspu/pir/trivial_symmetric_pir/trivial_spir.h"

#include "libspu/pir/pir.pb.h"

llvm::cl::opt<std::string> InPathOpt(
    "in_path", llvm::cl::init("data.csv"),
    llvm::cl::desc("[in] pir data in file path"));

llvm::cl::opt<std::string> KeyColumnsOpt("key_columns", llvm::cl::init("id"),
                                         llvm::cl::desc("key columns"));

llvm::cl::opt<std::string> OutPathOpt(
    "out_path", llvm::cl::init("."),
    llvm::cl::desc("[out] pir query output path"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  SPDLOG_INFO("Trivial symmetric PIR, [mode]: fully online, [role]: client");
  std::string ip0 = "127.0.0.1:17268";
  std::string ip1 = "127.0.0.1:17269";
  yacl::link::ContextDesc lctx_desc;
  lctx_desc.parties.push_back({"id_0", ip0});
  lctx_desc.parties.push_back({"id_1", ip1});
  lctx_desc.recv_timeout_ms = 2 * 60 * 1000;
  lctx_desc.connect_retry_times = 180;

  auto link_ctx = yacl::link::FactoryBrpc().CreateContext(lctx_desc, 1);
  link_ctx->ConnectToMesh();

  std::vector<std::string> ids = absl::StrSplit(KeyColumnsOpt.getValue(), ',');

  SPDLOG_INFO("in_path: {}", InPathOpt.getValue());
  SPDLOG_INFO("id columns: {}", KeyColumnsOpt.getValue());

  spu::pir::PirClientConfig config;

  config.set_pir_protocol(spu::pir::PirProtocol::KEYWORD_PIR_LABELED_PSI);
  config.set_input_path(InPathOpt.getValue());

  config.mutable_key_columns()->Add(ids.begin(), ids.end());
  config.set_output_path(OutPathOpt.getValue());

  spu::pir::PirResultReport report =
      spu::pir::TrivialSpirFullyOnlineClient(link_ctx, config);

  SPDLOG_INFO("data count:{}", report.data_count());

  return 0;
}