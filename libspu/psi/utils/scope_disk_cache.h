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

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "spdlog/spdlog.h"

#include "libspu/psi/io/io.h"

namespace spu::psi {

class ScopedTempDir {
 private:
  std::filesystem::path dir_;

 public:
  bool CreateUniqueTempDirUnderPath(const std::filesystem::path& dir);

  const std::filesystem::path& path() const { return dir_; }

  ~ScopedTempDir() {
    if (!dir_.empty()) {
      std::error_code ec;
      std::filesystem::remove_all(dir_, ec);
      // Leave error as it is, do nothing
    }
  }
};

/// An utility for creating temporary hash bin files.
class ScopeDiskCache {
 public:
  static std::unique_ptr<ScopeDiskCache> Create(
      const std::filesystem::path& parent_path) {
    auto scoped_cache =
        std::unique_ptr<ScopeDiskCache>(new ScopeDiskCache(parent_path));
    if (!scoped_cache->scoped_temp_dir_.CreateUniqueTempDirUnderPath(
            parent_path)) {
      return nullptr;
    }
    return scoped_cache;
  }

  ScopeDiskCache(const ScopeDiskCache&) = delete;
  ScopeDiskCache& operator=(const ScopeDiskCache&) = delete;

  std::string GetBinPath(size_t index) const;

  void CreateHashBinStreams(
      size_t num_bins,
      std::vector<std::unique_ptr<io::OutputStream>>* bin_outs) const;

  std::unique_ptr<io::InputStream> CreateHashBinInputStream(size_t index) const;

 private:
  explicit ScopeDiskCache(std::filesystem::path cache_dir)
      : cache_dir_(std::move(cache_dir)) {}

  const std::filesystem::path cache_dir_;
  ScopedTempDir scoped_temp_dir_;
};

}  // namespace spu::psi
