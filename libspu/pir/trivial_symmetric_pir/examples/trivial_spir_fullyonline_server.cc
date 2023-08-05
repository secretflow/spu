// build 'trivial_spir_fullyonline_server'
// > bazel build
// //libspu/pir/trivial_symmetric_pir/examples:trivial_spir_fullyonline_server
// -c opt
//
// To generate ecc oprf secret key, start terminals:
// > dd if=/dev/urandom of=secret_key.bin-id bs=32 count=1
// > dd if=/dev/urandom of=secret_key.bin-label bs=32 count=1
//
// To run the example, start terminals:
// >
// ./bazel-bin/libspu/pir/trivial_symmetric_pir/examples/trivial_spir_fullyonline_server -in_path ./examples/data/pir_server_data.csv -oprf_key_path ./dump/secret_key.bin  -key_columns id -label_length 72

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

llvm::cl::opt<std::string> OprfKeyPathOpt(
    "oprf_key_path", llvm::cl::init("oprf_key.bin"),
    llvm::cl::desc("[in] ecc oprf secretkey file path, 32bytes binary file"));

llvm::cl::opt<std::string> KeyColumnsOpt("key_columns", llvm::cl::init("id"),
                                         llvm::cl::desc("key columns"));

llvm::cl::opt<int> LabelPadLengthOpt(
    "label_length", llvm::cl::init(16),
    llvm::cl::desc("the maximum byte length of the label (labels are "
                   "expected to have the same lengths)"));

namespace {
constexpr uint32_t kLinkRecvTimeout = 30 * 60 * 1000;
}

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  SPDLOG_INFO("Trivial symmetric PIR, [mode]: fully online, [role]: server");
  SPDLOG_INFO("in_path: {}", InPathOpt.getValue());
  SPDLOG_INFO("id columns: {}", KeyColumnsOpt.getValue());
  SPDLOG_INFO("oprf key path: {}", OprfKeyPathOpt.getValue());
  std::string ip0 = "127.0.0.1:17268";
  std::string ip1 = "127.0.0.1:17269";
  yacl::link::ContextDesc lctx_desc;
  lctx_desc.parties.push_back({"id_0", ip0});
  lctx_desc.parties.push_back({"id_1", ip1});
  lctx_desc.recv_timeout_ms = 2 * 60 * 1000;
  lctx_desc.connect_retry_times = 180;

  auto link_ctx = yacl::link::FactoryBrpc().CreateContext(lctx_desc, 0);
  link_ctx->ConnectToMesh();

  link_ctx->SetRecvTimeout(kLinkRecvTimeout);
  SPDLOG_INFO("Link setup finished!");

  std::vector<std::string> ids = absl::StrSplit(KeyColumnsOpt.getValue(), ',');

  spu::pir::PirSetupConfig config;

  config.set_pir_protocol(spu::pir::PirProtocol::KEYWORD_PIR_LABELED_PSI);
  config.set_input_path(InPathOpt.getValue());

  config.mutable_key_columns()->Add(ids.begin(), ids.end());

  config.set_label_max_len(LabelPadLengthOpt.getValue());
  config.set_oprf_key_path(OprfKeyPathOpt.getValue());

  spu::pir::PirResultReport report =
      spu::pir::TrivialSpirFullyOnlineServer(link_ctx, config);

  SPDLOG_INFO("data count:{}", report.data_count());

  return 0;
}