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

#pragma once

#include <string>

#include "spdlog/spdlog.h"
#include "yacl/link/trace.h"

namespace spu::logging {

namespace internal {
class SpuTraceLogger : public yacl::link::TraceLogger {
 public:
  SpuTraceLogger(const std::shared_ptr<spdlog::logger>& logger,
                 size_t content_length);

 private:
  void LinkTraceImpl(std::string_view event, std::string_view tag,
                     std::string_view content) override;

  std::shared_ptr<spdlog::logger> logger_;
  size_t content_length_;
};
}  // namespace internal

enum class LogLevel {
  Debug = 0,
  Info = 1,
  Warn = 2,
  Error = 3,
};

struct LogOptions {
  bool enable_console_logger = true;
  std::string system_log_path = "spu.log";
  std::string trace_log_path;

  LogLevel log_level = LogLevel::Info;

  size_t max_log_file_size = 500 * 1024 * 1024;
  size_t max_log_file_count = 10;
  size_t trace_content_length = 100;  // Byte
};

void SetupLogging(const LogOptions& options = {});

}  // namespace spu::logging
