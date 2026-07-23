// Copyright 2024 Ant Group Co., Ltd.
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
#include <unistd.h>

#include <ctime>
#include <filesystem>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <unordered_map>

#include "fmt/format.h"
#include "leveldb/db.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/mpc/semi2k/beaver/beaver_interface.h"

namespace spu::mpc::semi2k {

class BeaverCache {
 public:
  // clang-format off
  BeaverCache()
      : cache_db_(fmt::format("BeaverCache.{}.{}.{}", getpid(), fmt::ptr(this),
                              std::random_device()())) {};
  // clang-format on
  ~BeaverCache() {
    db_.reset();
    try {
      // try remove.
      std::filesystem::remove_all(cache_db_);
    } catch (const std::exception&) {
      // if error. do nothing.
    }
  };
  BeaverCache(const BeaverCache&) = delete;
  BeaverCache& operator=(const BeaverCache&) = delete;

  void EnableCache(const NdArrayRef&);

  struct Cache {
    bool enabled{false};
    Beaver::ReplayDesc replay_desc;
    NdArrayRef open_cache;
  };

  Cache GetCache(const NdArrayRef&, bool allow_transpose = true) const;

  void SetCache(const NdArrayRef&, const Beaver::ReplayDesc&,
                const NdArrayRef&);

  void DisableCache(const NdArrayRef&);

 private:
  void LazyInitCacheDB();

  struct CacheMeta {
    Beaver::ReplayDesc replay;
    // data_shape_strides_idx
    std::vector<std::string> leveldb_keys;
  };

  NdArrayRef ReadCache(const CacheMeta&, const NdArrayRef&) const;
  void WriteCache(const NdArrayRef&, const NdArrayRef&, CacheMeta&);
  void DropCache(const CacheMeta&);

  const std::string cache_db_;
  std::once_flag db_lazy_once_flag_;
  mutable std::shared_mutex mutex_;
  std::unique_ptr<leveldb::DB> db_;
  const size_t cache_slice_size_ = 32 * 1024 * 1024;
  // for all NdArrayRef share same buffer (cache enabled NdArrayRef and slice of
  // this NdArrayRef) use NdArrayRef(data ptr with offset + shape + strides) to
  // track the specific cache.
  using BufferCacheMeta = std::unordered_map<NdArrayRef, CacheMeta>;
  // use NdArrayRef's under-layer buffer data ptr as first map's key to mark if
  // we need keep cache for a NdArrayRef.
  std::unordered_map<const void*, BufferCacheMeta> cache_meta_;
};

}  // namespace spu::mpc::semi2k
