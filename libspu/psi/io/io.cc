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

#include "libspu/psi/io/io.h"

#include <memory>
#include <utility>

#include "yacl/io/rw/csv_reader.h"
#include "yacl/io/rw/csv_writer.h"
#include "yacl/io/stream/file_io.h"
#include "yacl/io/stream/mem_io.h"

#include "libspu/core/prelude.h"

namespace spu::psi::io {

std::unique_ptr<InputStream> BuildInputStream(const std::any& io_options) {
  std::unique_ptr<InputStream> is;
  if (io_options.type() == typeid(MemIoOptions)) {
    auto op = std::any_cast<MemIoOptions>(io_options);
    is = std::make_unique<yacl::io::MemInputStream>(*op.mem_io_buffer);
  } else if (io_options.type() == typeid(FileIoOptions)) {
    auto op = std::any_cast<FileIoOptions>(io_options);
    is = std::make_unique<yacl::io::FileInputStream>(op.file_name);
  } else {
    SPU_THROW("unknow io_options type {}", io_options.type().name());
  }

  return is;
}

std::unique_ptr<OutputStream> BuildOutputStream(const std::any& io_options) {
  std::unique_ptr<OutputStream> os;
  if (io_options.type() == typeid(MemIoOptions)) {
    auto op = std::any_cast<MemIoOptions>(io_options);
    os = std::make_unique<yacl::io::MemOutputStream>(op.mem_io_buffer);
  } else if (io_options.type() == typeid(FileIoOptions)) {
    auto op = std::any_cast<FileIoOptions>(io_options);
    os = std::make_unique<yacl::io::FileOutputStream>(
        op.file_name, op.exit_for_fail_in_destructor);
  } else {
    SPU_THROW("unknow io_options type {}", io_options.type().name());
  }
  return os;
}

std::unique_ptr<Reader> BuildReader(const std::any& io_options,
                                    const std::any& format_options) {
  auto is = BuildInputStream(io_options);
  std::unique_ptr<Reader> ret;
  if (format_options.type() == typeid(CsvOptions)) {
    auto op = std::any_cast<CsvOptions>(format_options);
    ret = std::make_unique<yacl::io::CsvReader>(
        op.read_options, std::move(is), op.field_delimiter, op.line_delimiter);
  } else {
    SPU_THROW("unknow format_options type {}", format_options.type().name());
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
    ret = std::make_unique<yacl::io::CsvWriter>(
        op.writer_options, std::move(os), op.field_delimiter,
        op.line_delimiter);
  } else {
    SPU_THROW("unknow format_options type {}", format_options.type().name());
  }
  ret->Init();
  return ret;
}

}  // namespace spu::psi::io
