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

#include "absl/strings/ascii.h"
#include "butil/base64.h"
#include "gflags/gflags.h"
#include "yacl/crypto/key_utils.h"

#include "libspu/core/logging.h"
#include "libspu/core/prelude.h"
#include "libspu/mpc/semi2k/beaver/beaver_impl/ttp_server/beaver_server.h"

namespace ttp_server_config {
DEFINE_bool(
    gen_key, false,
    "if true, gen a pair of asym_crypto_schema key in base64, then exit.");
DEFINE_string(asym_crypto_schema, "sm2",
              "asym_crypto_schema: support [\"SM2\"]");
DEFINE_string(server_private_key, "", "base64ed server_private_key");
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

  std::pair<yacl::Buffer, yacl::Buffer> asym_crypto_key;
  if (lower_schema == "sm2") {
    asym_crypto_key = yacl::crypto::GenSm2KeyPairToPemBuf();
  } else {
    SPU_THROW("not support asym_crypto_schema {}", asym_crypto_schema);
  }

  std::string base64_pk;
  std::string base64_sk;

  butil::Base64Encode(std::string(asym_crypto_key.first.data<char>(),
                                  asym_crypto_key.first.size()),
                      &base64_pk);
  butil::Base64Encode(std::string(asym_crypto_key.second.data<char>(),
                                  asym_crypto_key.second.size()),
                      &base64_sk);
  SPDLOG_INFO("\nbase64ed public key:\n{}\n\nbase64ed private key:\n{}\n",
              base64_pk, base64_sk);
}

int main(int argc, char* argv[]) {
  // Parse gflags. We recommend you to use gflags as well.
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  SetupLogging();

  if (ttp_server_config::FLAGS_gen_key) {
    GenKeyPair(ttp_server_config::FLAGS_asym_crypto_schema);
    return 0;
  }

  yacl::Buffer decode_private_key;
  {
    std::string key;
    SPU_ENFORCE(
        butil::Base64Decode(ttp_server_config::FLAGS_server_private_key, &key));
    decode_private_key =
        yacl::Buffer(decode_private_key.data(), decode_private_key.size());
  }

  spu::mpc::semi2k::beaver::ttp_server::ServerOptions ops{
      .port = ttp_server_config::FLAGS_port,
      .asym_crypto_schema = ttp_server_config::FLAGS_asym_crypto_schema,
      .server_private_key = std::move(decode_private_key),
  };

  return spu::mpc::semi2k::beaver::ttp_server::RunUntilAskedToQuit(ops);
}
