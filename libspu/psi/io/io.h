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

#pragma once
#include <any>
#include <memory>
#include <string>

#include "yacl/io/rw/reader.h"
#include "yacl/io/rw/writer.h"
#include "yacl/io/stream/interface.h"
namespace spu::psi::io {

using Schema = yacl::io::Schema;

using InputStream = yacl::io::InputStream;
using OutputStream = yacl::io::OutputStream;

using Reader = yacl::io::Reader;
using Writer = yacl::io::Writer;
using ReaderOptions = yacl::io::ReaderOptions;
using WriterOptions = yacl::io::WriterOptions;

using ColumnVectorBatch = yacl::io::ColumnVectorBatch;
using FloatColumnVector = yacl::io::FloatColumnVector;
using DoubleColumnVector = yacl::io::DoubleColumnVector;
using StringColumnVector = yacl::io::StringColumnVector;

template <class S>
using ColumnVector = yacl::io::ColumnVector<S>;

struct MemIoOptions {
  // IO buffer
  // for reader, it's mock input data.
  // for writer, it recv data when out stream close/.
  std::string* mem_io_buffer;
};

struct FileIoOptions {
  FileIoOptions() : exit_for_fail_in_destructor(true) {}
  explicit FileIoOptions(const std::string& f)
      : file_name(f), exit_for_fail_in_destructor(true) {}
  // filename for read / write.
  // IF read:
  //    PLS make sure file exist.
  // IF write:
  //    FileIo always trunc file if target file exist.
  std::string file_name;
  // FileIo will try Close() in destructor if file not closed.
  // IF false:
  //    FileIo action likes std::ofstream, ignores Close()'s io exception.
  //    output file may be damaged.
  //    Use for online service should not exit.
  // IF true:
  //    FileIo will log & call _exit if Close() throw any io exception.
  //
  // SO, !!! PLS manually Close writer before release it !!!
  bool exit_for_fail_in_destructor;
};

struct CsvOptions {
  // for BuildReader
  ReaderOptions read_options;
  // for BuildWriter
  WriterOptions writer_options;
  char field_delimiter = ',';
  char line_delimiter = '\n';
};

/**
 * Writer & OutputStream Always trunc file if target file exist.
 *
 * Please read io_test.cc as examples of how to use.
 */

// Formatted IO
std::unique_ptr<Reader> BuildReader(const std::any& io_options,
                                    const std::any& format_options);
// !!! PLS manually call Close before release Writer !!!
std::unique_ptr<Writer> BuildWriter(const std::any& io_options,
                                    const std::any& format_options);

// Raw IO
std::unique_ptr<InputStream> BuildInputStream(const std::any& io_options);
// !!! PLS manually call Close before release OutputStream !!!
std::unique_ptr<OutputStream> BuildOutputStream(const std::any& io_options);

}  // namespace spu::psi::io
