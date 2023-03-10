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

#include "gflags/gflags.h"

#include "libspu/core/logging.h"
#include "libspu/mpc/semi2k/beaver/ttp_server/beaver_server.h"

namespace ttp_server_config {

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

int main(int argc, char* argv[]) {
  // Parse gflags. We recommend you to use gflags as well.
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  SetupLogging();

  return spu::mpc::semi2k::beaver::ttp_server::RunUntilAskedToQuit(
      ttp_server_config::FLAGS_port);
}
