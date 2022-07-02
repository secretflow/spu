// Copyright (c) 2021 Ant Financial. All rights reserved.

#pragma once
#include <any>
#include <memory>

#include "yasl/io/rw/reader.h"
#include "yasl/io/rw/writer.h"
#include "yasl/io/stream/interface.h"

namespace spu::psi::io {

using Schema = yasl::io::Schema;

using InputStream = yasl::io::InputStream;
using OutputStream = yasl::io::OutputStream;

using Reader = yasl::io::Reader;
using Writer = yasl::io::Writer;
using ReaderOptions = yasl::io::ReaderOptions;
using WriterOptions = yasl::io::WriterOptions;

using ColumnVectorBatch = yasl::io::ColumnVectorBatch;
using FloatColumnVector = yasl::io::FloatColumnVector;
using DoubleColumnVector = yasl::io::DoubleColumnVector;
using StringColumnVector = yasl::io::StringColumnVector;

template <class S>
using ColumnVector = yasl::io::ColumnVector<S>;

struct MemIoOptions {
  // IO buffer
  // for reader, it's mock input data.
  // for writer, it recv data when out stream close/.
  std::string* mem_io_buffer;
};

struct FileIoOptions {
  FileIoOptions() : exit_for_fail_in_destructor(true) {}
  FileIoOptions(const std::string& f)
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