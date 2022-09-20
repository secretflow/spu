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

#include "spu/psi/bucket_psi.h"

#include <filesystem>
#include <numeric>
#include <type_traits>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "spdlog/spdlog.h"
#include "yasl/base/exception.h"
#include "yasl/crypto/hash_util.h"
#include "yasl/utils/scope_guard.h"
#include "yasl/utils/serialize.h"

#include "spu/psi/core/ecdh_psi.h"
#include "spu/psi/cryptor/cryptor_selector.h"
#include "spu/psi/io/io.h"
#include "spu/psi/operator/operator.h"
#include "spu/psi/utils/batch_provider.h"
#include "spu/psi/utils/cipher_store.h"
#include "spu/psi/utils/csv_checker.h"
#include "spu/psi/utils/csv_header_analyzer.h"
#include "spu/psi/utils/serialize.h"
#include "spu/psi/utils/utils.h"

namespace spu::psi {

namespace {

constexpr size_t kCsvHeaderLineCount = 1;

constexpr size_t kBucketSize = 1 << 20;

bool HashListEqualTest(const std::vector<yasl::Buffer>& hash_list) {
  YASL_ENFORCE(!hash_list.empty(), "unsupported hash_list size={}",
               hash_list.size());
  for (size_t idx = 1; idx < hash_list.size(); idx++) {
    if (hash_list[idx] == hash_list[0]) {
      continue;
    }
    return false;
  }
  return true;
}

}  // namespace

BucketPsi::BucketPsi(BucketPsiConfig config,
                     std::shared_ptr<yasl::link::Context> lctx)
    : config_(std::move(config)), lctx_(std::move(lctx)) {
  Init();
}

PsiResultReport BucketPsi::Run() {
  PsiResultReport report;

  // prepare fields vec
  selected_fields_.insert(selected_fields_.end(),
                          config_.input_params().select_fields().begin(),
                          config_.input_params().select_fields().end());

  // input dataset pre check
  SPDLOG_INFO("Begin sanity check for input file: {}, precheck_switch:{}",
              config_.input_params().path(), config_.input_params().precheck());
  std::shared_ptr<CsvChecker> checker;
  auto csv_check_f = std::async([&] {
    checker = std::make_shared<CsvChecker>(
        config_.input_params().path(), selected_fields_,
        std::filesystem::path(config_.output_params().path())
            .parent_path()
            .string(),
        !config_.input_params().precheck());
  });
  // keep alived
  SyncWait(lctx_, &csv_check_f);
  SPDLOG_INFO("End sanity check for input file: {}, size={}",
              config_.input_params().path(), checker->data_count());
  report.set_original_count(checker->data_count());

  // gather others hash digest
  std::vector<yasl::Buffer> digest_buf_list =
      yasl::link::AllGather(lctx_, checker->hash_digest(), "PSI:SYNC_DIGEST");

  // run psi
  std::vector<uint64_t> indices;
  bool digest_equal = HashListEqualTest(digest_buf_list);
  if (!digest_equal) {
    indices = RunPsi(checker->data_count());
  } else {
    SPDLOG_INFO("Skip doing psi, because dataset has been aligned!");
    indices.resize(checker->data_count());
    std::iota(indices.begin(), indices.end(), 0);
  }

  if (static_cast<size_t>(config_.receiver_rank()) != lctx_->Rank() &&
      config_.broadcast_result() == false) {
    report.set_intersection_count(-1);
    // no generate output file;
    return report;
  } else {
    report.set_intersection_count(indices.size());
  }

  // filter dataset
  SPDLOG_INFO("Begin post filtering, indices.size={}, should_sort={}",
              indices.size(), config_.output_params().need_sort());
  // sort indices
  std::sort(indices.begin(), indices.end());
  // use tmp file to avoid `shell Injection`
  auto timestamp_str = std::to_string(absl::ToUnixNanos(absl::Now()));
  auto tmp_sort_in_file =
      std::filesystem::path(config_.output_params().path())
          .parent_path()
          .append(fmt::format("tmp-sort-in-{}", timestamp_str));
  auto tmp_sort_out_file =
      std::filesystem::path(config_.output_params().path())
          .parent_path()
          .append(fmt::format("tmp-sort-out-{}", timestamp_str));
  // register remove of temp file.
  ON_SCOPE_EXIT([&] {
    std::error_code ec;
    std::filesystem::remove(tmp_sort_out_file, ec);
    if (ec.value() != 0) {
      SPDLOG_WARN("can not remove tmp file: {}, msg: {}",
                  tmp_sort_out_file.c_str(), ec.message());
    }
    std::filesystem::remove(tmp_sort_in_file, ec);
    if (ec.value() != 0) {
      SPDLOG_WARN("can not remove tmp file: {}, msg: {}",
                  tmp_sort_in_file.c_str(), ec.message());
    }
  });
  FilterFileByIndices(config_.input_params().path(), tmp_sort_in_file, indices,
                      kCsvHeaderLineCount);
  if (config_.output_params().need_sort() && !digest_equal) {
    MultiKeySort(tmp_sort_in_file, tmp_sort_out_file, selected_fields_);

    std::filesystem::rename(tmp_sort_out_file, config_.output_params().path());
  } else {
    std::filesystem::rename(tmp_sort_in_file, config_.output_params().path());
  }

  SPDLOG_INFO("End post filtering, in={}, out={}",
              config_.input_params().path(), config_.output_params().path());

  return report;
}

void BucketPsi::Init() {
  // TODO: deal input_params data_type

  if (config_.bucket_size() == 0) {
    config_.set_bucket_size(kBucketSize);
  }
  SPDLOG_INFO("bucket size set to {}", config_.bucket_size());

  MemoryPsiConfig config;
  config.set_psi_type(config_.psi_type());
  config.set_curve_type(config_.curve_type());
  config.set_receiver_rank(config_.receiver_rank());
  config.set_broadcast_result(config_.broadcast_result());
  mem_psi_ = std::make_unique<MemoryPsi>(config, lctx_);

  // create output folder.
  auto out_dir_path =
      std::filesystem::path(config_.output_params().path()).parent_path();
  std::error_code ec;
  std::filesystem::create_directory(out_dir_path, ec);
  YASL_ENFORCE(ec.value() == 0,
               "failed to create output dir={} for path={}, reason = {}",
               out_dir_path.string(), config_.output_params().path(),
               ec.message());
}

std::vector<uint64_t> BucketPsi::RunPsi(uint64_t self_items_count) {
  SPDLOG_INFO("Run psi protocol={}, self_items_count={}", config_.psi_type(),
              self_items_count);

  if (config_.psi_type() == PsiType::ECDH_PSI_2PC) {
    EcdhPsiOptions psi_options;
    psi_options.ecc_cryptor = CreateEccCryptor(CurveType::CURVE_25519);
    psi_options.link_ctx = lctx_;
    psi_options.target_rank = static_cast<size_t>(config_.receiver_rank());
    if (config_.broadcast_result()) {
      psi_options.target_rank = yasl::link::kAllRank;
    }

    auto batch_provider = std::make_shared<CsvBatchProvider>(
        config_.input_params().path(), selected_fields_);
    auto cipher_store = std::make_shared<DiskCipherStore>(
        std::filesystem::path(config_.output_params().path()).parent_path(),
        64);

    // Launch ECDH-PSI core.
    RunEcdhPsi(psi_options, batch_provider, cipher_store);

    return cipher_store->FinalizeAndComputeIndices();
  } else {
    return RunBucketPsi(self_items_count);
  }
}

std::vector<uint64_t> BucketPsi::RunBucketPsi(uint64_t self_items_count) {
  std::vector<uint64_t> ret;

  std::vector<size_t> items_size_list =
      AllGatherItemsSize(lctx_, self_items_count);

  std::vector<size_t> bucket_count_list(items_size_list.size());
  size_t max_bucket_count = 0;
  size_t min_item_size = self_items_count;

  for (size_t idx = 0; idx < items_size_list.size(); idx++) {
    bucket_count_list[idx] =
        (items_size_list[idx] + config_.bucket_size() - 1) /
        config_.bucket_size();
    max_bucket_count = std::max(max_bucket_count, bucket_count_list[idx]);
    min_item_size = std::min(min_item_size, items_size_list[idx]);

    SPDLOG_INFO("psi protocol={}, rank={} item_size={}", config_.psi_type(),
                idx, items_size_list[idx]);
  }

  // one party item_size is 0, no need to do intersection
  if (min_item_size == 0) {
    SPDLOG_INFO("psi protocol={}, min_item_size=0", config_.psi_type());
    return ret;
  }

  SPDLOG_INFO("psi protocol={}, bucket_count={}", config_.psi_type(),
              max_bucket_count);

  // hash bucket items
  auto bucket_store = CreateCacheFromCsv(
      config_.input_params().path(), selected_fields_,
      std::filesystem::path(config_.output_params().path()).parent_path(),
      max_bucket_count);
  for (size_t bucket_idx = 0; bucket_idx < bucket_store->BucketNum();
       bucket_idx++) {
    auto bucket_items_list = bucket_store->LoadBucketItems(bucket_idx);

    SPDLOG_INFO("run psi bucket_idx={}, bucket_item_size={} ", bucket_idx,
                bucket_items_list.size());

    std::vector<std::string> item_data_list;
    item_data_list.reserve(bucket_items_list.size());
    for (const auto& item : bucket_items_list) {
      item_data_list.push_back(item.base64_data);
    }

    auto result_list = mem_psi_->Run(item_data_list);

    SPDLOG_INFO("psi protocol={}, result_size={}", config_.psi_type(),
                result_list.size());

    // get result item indices
    GetResultIndices(item_data_list, bucket_items_list, result_list, &ret);
  }

  return ret;
}

void BucketPsi::GetResultIndices(
    const std::vector<std::string>& item_data_list,
    const std::vector<HashBucketCache::BucketItem>& item_list,
    std::vector<std::string>& result_list,
    std::vector<uint64_t>* indices) const {
  indices->reserve(indices->size() + result_list.size());
  if (result_list.empty()) {
    return;
  } else if (result_list.size() == item_list.size()) {
    for (const auto& item : item_list) {
      indices->push_back(item.index);
    }
    return;
  }

  std::sort(result_list.begin(), result_list.end());
  for (size_t i = 0; i < item_data_list.size(); ++i) {
    if (std::binary_search(result_list.begin(), result_list.end(),
                           item_data_list[i])) {
      indices->push_back(item_list[i].index);
    }
  }
}

}  // namespace spu::psi
