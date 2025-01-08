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

#include "libspu/core/logging.h"

#include <filesystem>

#include "absl/strings/escaping.h"

// TODO: we should not include brpc here.
#include "butil/logging.h"
#include "fmt/format.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#include "libspu/core/prelude.h"

namespace spu::logging {

namespace internal {

SpuTraceLogger::SpuTraceLogger(const std::shared_ptr<spdlog::logger>& logger,
                               size_t content_length)
    : logger_(logger), content_length_(content_length) {}

void SpuTraceLogger::LinkTraceImpl(std::string_view event, std::string_view tag,
                                   std::string_view content) {
  if (logger_) {
    SPDLOG_LOGGER_INFO(
        logger_, "[spu link] key={}, tag={}, value={}", event, tag,
        absl::BytesToHexString(content.substr(0, content_length_)));
  }
}
}  // namespace internal

namespace {
// clang-format off
/// custom formatting:
/// https://github.com/gabime/spdlog/wiki/3.-Custom-formatting
/// example: ```2019-11-29 06:58:54.633 [info] [log_test.cpp:TestBody:16] The answer is 42.```
// clang-format on
const char* kFormatPattern = "%Y-%m-%d %H:%M:%S.%e [%l] [%s:%!:%#] %v";

const char* kBrpcUnknownFuncname = "BRPC";

const size_t kDefaultMaxLogFileSize = 500 * 1024 * 1024;
const size_t kDefaultMaxLogFileCount = 3;

spdlog::level::level_enum FromBrpcLogSeverity(int severity) {
  spdlog::level::level_enum level = spdlog::level::off;
  if (severity == ::logging::BLOG_INFO) {
    level = spdlog::level::debug;
  } else if (severity == ::logging::BLOG_NOTICE) {
    level = spdlog::level::info;
  } else if (severity == ::logging::BLOG_WARNING) {
    level = spdlog::level::warn;
  } else if (severity == ::logging::BLOG_ERROR) {
    level = spdlog::level::err;
  } else if (severity == ::logging::BLOG_FATAL) {
    level = spdlog::level::critical;
  } else {
    level = spdlog::level::warn;
  }
  return level;
}

spdlog::level::level_enum FromSpuLogLevel(LogLevel spu_log_level) {
  spdlog::level::level_enum level = spdlog::level::off;
  if (spu_log_level == LogLevel::Debug) {
    level = spdlog::level::debug;
  } else if (spu_log_level == LogLevel::Info) {
    level = spdlog::level::info;
  } else if (spu_log_level == LogLevel::Warn) {
    level = spdlog::level::warn;
  } else if (spu_log_level == LogLevel::Error) {
    level = spdlog::level::err;
  } else {
    level = spdlog::level::info;
  }
  return level;
}

class SpuLogSink : public ::logging::LogSink {
 public:
  bool OnLogMessage(int severity, const char* file, int line,
                    const butil::StringPiece& log_content) override {
    spdlog::level::level_enum log_level = FromBrpcLogSeverity(severity);
    spdlog::log(spdlog::source_loc{file, line, kBrpcUnknownFuncname}, log_level,
                "{}", fmt::string_view(log_content.data(), log_content.size()));
    return true;
  }
};

void SinkBrpcLogWithDefaultLogger() {
  static SpuLogSink log_sink;
  ::logging::SetLogSink(&log_sink);
  ::logging::SetMinLogLevel(::logging::BLOG_ERROR);
}

}  // namespace

void SetupLogging(const LogOptions& options) {
  spdlog::level::level_enum level = FromSpuLogLevel(options.log_level);

  auto log_dir = std::filesystem::path(options.system_log_path).parent_path();
  if (!log_dir.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(log_dir, ec);
    SPU_ENFORCE(ec.value() == 0, "failed to create dir={}, reason = {}",
                log_dir.string(), ec.message());
  }

  auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
      options.system_log_path, options.max_log_file_size,
      options.max_log_file_count);

  std::vector<spdlog::sink_ptr> sinks = {file_sink};

  if (options.enable_console_logger) {
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    sinks.emplace_back(std::move(console_sink));
  }

  auto root_logger = std::make_shared<spdlog::logger>(
      "system_root", sinks.begin(), sinks.end());
  root_logger->set_level(level);
  root_logger->set_pattern(kFormatPattern);
  root_logger->flush_on(level);
  spdlog::set_default_logger(root_logger);

  SinkBrpcLogWithDefaultLogger();

  if (!options.trace_log_path.empty()) {
    auto trace_logger = spdlog::rotating_logger_mt(
        "trace", options.trace_log_path, kDefaultMaxLogFileSize,
        kDefaultMaxLogFileCount);
    yacl::link::TraceLogger::SetLogger(
        std::make_shared<internal::SpuTraceLogger>(
            trace_logger, options.trace_content_length));
  }
}

}  // namespace spu::logging
