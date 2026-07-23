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

#include "libspu/mpc/semi2k/beaver/beaver_cache.h"

namespace spu::mpc::semi2k {

namespace {
void Check(const NdArrayRef& x, const Beaver::ReplayDesc& replay) {
  SPU_ENFORCE(x.eltype().as<Ring2k>()->field() == replay.field);
  SPU_ENFORCE(x.shape().numel() == replay.size);
}

std::string KeyPrefix(const NdArrayRef& x) {
  // data_shape_strides
  const size_t shape_size = x.shape().size() * sizeof(Shape::value_type);
  const size_t strides_size = x.strides().size() * sizeof(Strides::value_type);
  std::string key_prefix(sizeof(x.data()) + shape_size + strides_size, 0);

  const void* data = x.data();
  std::memcpy(key_prefix.data(), &data, sizeof(data));
  size_t pos = sizeof(data);
  std::memcpy(key_prefix.data() + pos, x.shape().data(), shape_size);
  pos += shape_size;
  std::memcpy(key_prefix.data() + pos, x.strides().data(), strides_size);

  return key_prefix;
}

std::string Key(const std::string& prefix, size_t idx) {
  std::string ret(prefix.size() + sizeof(idx), 0);

  std::memcpy(ret.data(), prefix.data(), prefix.size());
  std::memcpy(ret.data() + prefix.size(), &idx, sizeof(idx));

  return ret;
}
}  // namespace

void BeaverCache::EnableCache(const NdArrayRef& x) {
  std::unique_lock lock(mutex_);
  if (cache_meta_.find(x.buf()->data()) != cache_meta_.end()) {
    return;
  }
  cache_meta_.insert({x.buf()->data(), BufferCacheMeta()});
}

BeaverCache::Cache BeaverCache::GetCache(const NdArrayRef& x,
                                         bool allow_transpose) const {
  Cache ret;

  std::shared_lock lock(mutex_);
  const auto cit = cache_meta_.find(x.buf()->data());
  if (cit == cache_meta_.end()) {
    return ret;
  }
  ret.enabled = true;

  const auto& buf_cache = cit->second;
  auto bit = buf_cache.find(x);
  if (bit != buf_cache.end()) {
    const auto& meta = bit->second;
    Check(x, meta.replay);
    ret.replay_desc = meta.replay;
    ret.replay_desc.status = Beaver::Replay;
    ret.open_cache = ReadCache(meta, x);
    return ret;
  }

  if (!allow_transpose) {
    return ret;
  }

  auto x_t = x.transpose();
  bit = buf_cache.find(x_t);
  if (bit != buf_cache.end()) {
    const auto& meta = bit->second;
    Check(x, meta.replay);
    ret.replay_desc = meta.replay;
    ret.replay_desc.status = Beaver::TransposeReplay;
    ret.open_cache = ReadCache(meta, x_t).transpose();
    return ret;
  }

  return ret;
}

void BeaverCache::SetCache(const NdArrayRef& x,
                           const Beaver::ReplayDesc& replay,
                           const NdArrayRef& open_cache) {
  LazyInitCacheDB();
  std::unique_lock lock(mutex_);

  const auto cit = cache_meta_.find(x.buf()->data());
  SPU_ENFORCE(cit != cache_meta_.end());

  auto& buf_cache = cit->second;
  const auto bit = buf_cache.find(x);
  SPU_ENFORCE(bit == buf_cache.end());

  auto mit = buf_cache.emplace(x, CacheMeta{replay, {}}).first;
  auto& meta = mit->second;

  WriteCache(x, open_cache, meta);
}

void BeaverCache::DisableCache(const NdArrayRef& x) {
  std::unique_lock lock(mutex_);

  const auto cit = cache_meta_.find(x.buf()->data());
  if (cit == cache_meta_.end()) {
    return;
  }

  for (auto& buf_cache : cit->second) {
    DropCache(buf_cache.second);
  }

  cache_meta_.erase(cit);
}

NdArrayRef BeaverCache::ReadCache(const CacheMeta& meta,
                                  const NdArrayRef& x) const {
  Beaver::Array ret_buf;
  ret_buf.resize(meta.replay.size * SizeOf(meta.replay.field));
  size_t read_size = 0;

  for (const auto& k : meta.leveldb_keys) {
    std::string cache_slice;
    auto status =
        db_->Get(leveldb::ReadOptions{.fill_cache = false}, k, &cache_slice);
    SPU_ENFORCE(status.ok());
    SPU_ENFORCE(cache_slice.size() <= ret_buf.size() - read_size);
    std::memcpy(ret_buf.data<std::byte>() + read_size, cache_slice.data(),
                cache_slice.size());
    read_size += cache_slice.size();
  }
  SPU_ENFORCE(read_size == static_cast<size_t>(ret_buf.size()));

  return NdArrayRef(std::make_shared<yacl::Buffer>(std::move(ret_buf)),
                    x.eltype(), x.shape());
}

void BeaverCache::WriteCache(const NdArrayRef& x, const NdArrayRef& open_cache,
                             CacheMeta& meta) {
  SPU_ENFORCE(open_cache.isCompact());
  const auto key_prefix = KeyPrefix(x);
  const size_t elsize = open_cache.elsize();
  size_t remain_size = elsize * open_cache.numel();
  size_t slice_idx = 0;

  do {
    auto key = Key(key_prefix, slice_idx);
    size_t start_pos = slice_idx * cache_slice_size_;
    size_t write_size = std::min(remain_size, cache_slice_size_);

    const char* data = open_cache.data<char>();
    leveldb::Slice slice{data + start_pos, write_size};
    auto status = db_->Put(leveldb::WriteOptions(), key, slice);
    SPU_ENFORCE(status.ok());

    meta.leveldb_keys.emplace_back(std::move(key));
    remain_size -= write_size;
    slice_idx++;
  } while (remain_size != 0);
}

void BeaverCache::DropCache(const CacheMeta& meta) {
  for (const auto& k : meta.leveldb_keys) {
    auto status = db_->Delete(leveldb::WriteOptions(), k);
    SPU_ENFORCE(status.ok());
  }
}

void BeaverCache::LazyInitCacheDB() {
  std::call_once(db_lazy_once_flag_, [&]() {
    leveldb::DB* db = nullptr;
    leveldb::Options options;
    options.create_if_missing = true;
    options.error_if_exists = true;
    options.compression = leveldb::CompressionType::kNoCompression;
    options.block_size = 4 << 20;
    auto status = leveldb::DB::Open(options, cache_db_, &db);
    SPU_ENFORCE(status.ok());
    db_.reset(db);
  });
}

}  // namespace spu::mpc::semi2k