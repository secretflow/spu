// Copyright 2023 Ant Group Co., Ltd.
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
#include <optional>

#include "absl/strings/ascii.h"
#include "butil/file_util.h"
#include "gflags/gflags.h"
#include "google/protobuf/util/json_util.h"
#include "yacl/crypto/key_utils.h"

#include "libspu/core/logging.h"
#include "libspu/core/prelude.h"
#include "libspu/mpc/semi2k/beaver/beaver_impl/ttp_server/beaver_server.h"

#include "libspu/mpc/semi2k/beaver/beaver_impl/ttp_server/config.pb.h"

using spu::mpc::semi2k::beaver::ttp_server::TTPServerConfig;

namespace ttp_server_config {
DEFINE_bool(
    gen_key, false,
    "if true, gen a pair of asym_crypto_schema key in base64, then exit.");
DEFINE_string(asym_crypto_schema, "sm2",
              "asym_crypto_schema: support [\"SM2\"]");
DEFINE_string(public_key_out, "sm2-key.pub", "file path to save public key");
DEFINE_string(private_key_out, "sm2-key", "file path to save private key");
DEFINE_string(config_file, "/home/admin/server-config.json",
              "server config file, json format, see config.proto");
DEFINE_string(private_key_file, "/home/admin/server-private-key",
              "private key file path");
DEFINE_int32(port, 9449, "TCP Port of this server");
DEFINE_string(log_dir, "logs", "log directory");
DEFINE_bool(enable_console_logger, true,
            "whether logging to stdout while logging to file");
DEFINE_int64(max_log_file_size, 100 * 1024 * 1024,
             "max file size for each log file");
DEFINE_int64(max_log_file_count, 10, "max rotated log files save in dir");
}  // namespace ttp_server_config

void SetupLogging() {
  spu::logging::LogOptions options;
  options.enable_console_logger =
      ttp_server_config::FLAGS_enable_console_logger;
  std::filesystem::path log_path(ttp_server_config::FLAGS_log_dir);
  options.system_log_path = log_path / "ttp_server.log";
  options.log_level = spu::logging::LogLevel::Debug;
  options.max_log_file_size = ttp_server_config::FLAGS_max_log_file_size;
  options.max_log_file_count = ttp_server_config::FLAGS_max_log_file_count;
  spu::logging::SetupLogging(options);
}

void GenKeyPair(const std::string& asym_crypto_schema) {
  auto lower_schema = absl::AsciiStrToLower(asym_crypto_schema);

  yacl::crypto::openssl::UniquePkey asym_crypto_key;
  if (lower_schema == "sm2") {
    asym_crypto_key = yacl::crypto::GenSm2KeyPair();
  } else {
    SPU_THROW("not support asym_crypto_schema {}", asym_crypto_schema);
  }

  yacl::crypto::ExportPublicKeyToPemFile(
      asym_crypto_key, ttp_server_config::FLAGS_public_key_out);
  yacl::crypto::ExportSecretKeyToDerFile(
      asym_crypto_key, ttp_server_config::FLAGS_private_key_out);
}

std::optional<TTPServerConfig> ReadServerConfig() {
  std::string json;
  if (!butil::ReadFileToString(
          butil::FilePath(ttp_server_config::FLAGS_config_file), &json)) {
    return std::nullopt;
  }

  TTPServerConfig config;
  auto status = google::protobuf::util::JsonStringToMessage(json, &config);
  SPU_ENFORCE(status.ok(), status.ToString());

  return config;
}

yacl::Buffer ReadPrivateKey() {
  auto private_key =
      yacl::crypto::LoadKeyFromFile(ttp_server_config::FLAGS_private_key_file);
  return yacl::crypto::ExportSecretKeyToPemBuf(private_key);
}

int main(int argc, char* argv[]) {
  // Parse gflags. We recommend you to use gflags as well.
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  SetupLogging();

  if (ttp_server_config::FLAGS_gen_key) {
    GenKeyPair(ttp_server_config::FLAGS_asym_crypto_schema);
    return 0;
  }

  spu::mpc::semi2k::beaver::ttp_server::ServerOptions ops;
  ops.server_private_key = ReadPrivateKey();
  auto config = ReadServerConfig();
  if (config.has_value()) {
    ops.port = config.value().server_port();
    ops.asym_crypto_schema = config.value().asym_crypto_schema();
    if (config->has_ssl()) {
      brpc::ServerSSLOptions ssl_options;
      ssl_options.default_cert.certificate = config.value().ssl().cert_file();
      ssl_options.default_cert.private_key = config.value().ssl().key_file();
      ssl_options.verify.ca_file_path = config.value().ssl().ca_file();
      ssl_options.verify.verify_depth = config.value().ssl().verify_depth();
      ops.brpc_ssl_options = std::move(ssl_options);
    }
  } else {
    SPDLOG_INFO("Failed to read config file, use command line options");
    ops.port = ttp_server_config::FLAGS_port;
    ops.asym_crypto_schema = ttp_server_config::FLAGS_asym_crypto_schema;
  }

  return spu::mpc::semi2k::beaver::ttp_server::RunUntilAskedToQuit(ops);
}
