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

#include "spu/psi/io/io.h"

#include "yasl/io/rw/csv_reader.h"
#include "yasl/io/rw/csv_writer.h"
#include "yasl/io/stream/file_io.h"
#include "yasl/io/stream/mem_io.h"

namespace spu::psi::io {

std::unique_ptr<InputStream> BuildInputStream(const std::any& io_options) {
  std::unique_ptr<InputStream> is;
  if (io_options.type() == typeid(MemIoOptions)) {
    auto op = std::any_cast<MemIoOptions>(io_options);
    is.reset(new yasl::io::MemInputStream(*op.mem_io_buffer));
  } else if (io_options.type() == typeid(FileIoOptions)) {
    auto op = std::any_cast<FileIoOptions>(io_options);
    is.reset(new yasl::io::FileInputStream(op.file_name));
  } else {
    YASL_THROW("unknow io_options type {}", io_options.type().name());
  }

  return is;
}

std::unique_ptr<OutputStream> BuildOutputStream(const std::any& io_options) {
  std::unique_ptr<OutputStream> os;
  if (io_options.type() == typeid(MemIoOptions)) {
    auto op = std::any_cast<MemIoOptions>(io_options);
    os.reset(new yasl::io::MemOutputStream(op.mem_io_buffer));
  } else if (io_options.type() == typeid(FileIoOptions)) {
    auto op = std::any_cast<FileIoOptions>(io_options);
    os.reset(new yasl::io::FileOutputStream(op.file_name,
                                            op.exit_for_fail_in_destructor));
  } else {
    YASL_THROW("unknow io_options type {}", io_options.type().name());
  }
  return os;
}

std::unique_ptr<Reader> BuildReader(const std::any& io_options,
                                    const std::any& format_options) {
  auto is = BuildInputStream(io_options);
  std::unique_ptr<Reader> ret;
  if (format_options.type() == typeid(CsvOptions)) {
    auto op = std::any_cast<CsvOptions>(format_options);
    ret.reset(new yasl::io::CsvReader(op.read_options, std::move(is),
                                      op.field_delimiter, op.line_delimiter));
  } else {
    YASL_THROW("unknow format_options type {}", format_options.type().name());
  }
  ret->Init();
  return ret;
}

std::unique_ptr<Writer> BuildWriter(const std::any& io_options,
                                    const std::any& format_options) {
  auto os = BuildOutputStream(io_options);
  std::unique_ptr<Writer> ret;
  if (format_options.type() == typeid(CsvOptions)) {
    auto op = std::any_cast<CsvOptions>(format_options);
    ret.reset(new yasl::io::CsvWriter(op.writer_options, std::move(os),
                                      op.field_delimiter, op.line_delimiter));
  } else {
    YASL_THROW("unknow format_options type {}", format_options.type().name());
  }
  ret->Init();
  return ret;
}

}  // namespace spu::psi::io