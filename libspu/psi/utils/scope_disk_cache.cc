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

#include "libspu/psi/utils/scope_disk_cache.h"

#include "llvm/Support/FileSystem.h"

#include "libspu/core/prelude.h"
namespace spu::psi {

std::string ScopeDiskCache::GetBinPath(size_t index) const {
  return fmt::format("{}/{}", scoped_temp_dir_.path().string(), index);
}

void ScopeDiskCache::CreateHashBinStreams(
    size_t num_bins,
    std::vector<std::unique_ptr<io::OutputStream>>* bin_outs) const {
  SPU_ENFORCE(num_bins != 0, "bad num_bins={}", num_bins);
  for (size_t i = 0; i < num_bins; ++i) {
    bin_outs->push_back(
        io::BuildOutputStream(io::FileIoOptions(GetBinPath(i))));
  }
}

std::unique_ptr<io::InputStream> ScopeDiskCache::CreateHashBinInputStream(
    size_t index) const {
  return io::BuildInputStream(io::FileIoOptions(GetBinPath(index)));
}

bool ScopedTempDir::CreateUniqueTempDirUnderPath(
    const std::filesystem::path& dir) {
  llvm::SmallVector<char, 128> file;
  llvm::sys::fs::createUniquePath("psi-disk-cache-%%%%%%", file, false);
  dir_ = dir / llvm::Twine(file).str();
  return std::filesystem::create_directory(dir_);
}

}  // namespace spu::psi
