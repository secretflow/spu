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
#include "yacl/base/exception.h"
#include "yacl/crypto/utils/hash_util.h"
#include "yacl/utils/scope_guard.h"
#include "yacl/utils/serialize.h"

#include "spu/psi/core/ecdh_psi.h"
#include "spu/psi/cryptor/cryptor_selector.h"
#include "spu/psi/io/io.h"
#include "spu/psi/utils/batch_provider.h"
#include "spu/psi/utils/cipher_store.h"
#include "spu/psi/utils/csv_checker.h"
#include "spu/psi/utils/csv_header_analyzer.h"
#include "spu/psi/utils/serialize.h"
#include "spu/psi/utils/utils.h"

#include "interconnection/algos/psi.pb.h"

namespace spu::psi {

namespace {

constexpr size_t kCsvHeaderLineCount = 1;

constexpr size_t kBucketSize = 1 << 20;

bool HashListEqualTest(const std::vector<yacl::Buffer>& hash_list) {
  YACL_ENFORCE(!hash_list.empty(), "unsupported hash_list size={}",
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
                     std::shared_ptr<yacl::link::Context> lctx, bool ic_mode)
    : config_(std::move(config)), ic_mode_(ic_mode), lctx_(std::move(lctx)) {
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
  if (ic_mode_) {
    csv_check_f.get();
  } else {
    SyncWait(lctx_, &csv_check_f);
  }
  SPDLOG_INFO("End sanity check for input file: {}, size={}",
              config_.input_params().path(), checker->data_count());
  report.set_original_count(checker->data_count());

  // gather others hash digest
  bool digest_equal = false;
  if (!ic_mode_) {
    std::vector<yacl::Buffer> digest_buf_list =
        yacl::link::AllGather(lctx_, checker->hash_digest(), "PSI:SYNC_DIGEST");
    digest_equal = HashListEqualTest(digest_buf_list);
  }

  // run psi
  std::vector<uint64_t> indices;
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
  if (out_dir_path.empty()) {
    return;  // create file under CWD, no need to create parent dir
  }

  std::error_code ec;
  std::filesystem::create_directory(out_dir_path, ec);
  YACL_ENFORCE(ec.value() == 0,
               "failed to create output dir={} for path={}, reason = {}",
               out_dir_path.string(), config_.output_params().path(),
               ec.message());
}

org::interconnection::algos::psi::HandshakeResponse CheckSelectAlgo(
    const org::interconnection::algos::psi::HandshakeRequest& request) {
  org::interconnection::algos::psi::HandshakeResponse response;
  if (request.version() != 1) {
    response.mutable_header()->set_error_code(
        org::interconnection::UNSUPPORTED_VERSION);
    response.mutable_header()->set_error_msg(
        "Secretflow only support interconnection protocol version 1");
    return response;
  }

  auto psi_name = PsiType_Name(PsiType::ECDH_PSI_2PC);
  int algo_idx = 0;
  for (; algo_idx < request.supported_algos_size(); ++algo_idx) {
    if (request.supported_algos(algo_idx) == psi_name) {
      break;
    }
  }

  if (algo_idx >= request.supported_algos_size()) {
    response.mutable_header()->set_error_code(
        org::interconnection::UNSUPPORTED_ALGO);
    response.mutable_header()->set_error_msg(
        "Secretflow only support algo ECDH_PSI_2PC");
    return response;
  }

  if (algo_idx >= request.algo_params_size()) {
    response.mutable_header()->set_error_code(
        org::interconnection::INVALID_REQUEST);
    response.mutable_header()->set_error_msg("algo param not found");
    return response;
  }

  org::interconnection::algos::psi::EcdhPsiParamsProposal ec_params;
  if (!request.algo_params(algo_idx).UnpackTo(&ec_params)) {
    response.mutable_header()->set_error_code(
        org::interconnection::INVALID_REQUEST);
    response.mutable_header()->set_error_msg("algo param unpack fail");
    return response;
  }

  // ecdh-psi version must be 1
  bool version_check = false;
  for (const auto& supported_version : ec_params.supported_versions()) {
    if (supported_version == 1) {
      version_check = true;
      break;
    }
  }
  if (!version_check) {
    response.mutable_header()->set_error_code(
        org::interconnection::UNSUPPORTED_VERSION);
    response.mutable_header()->set_error_msg(
        "Secretflow only support ecdh-psi version 1");
    return response;
  }

  if (ec_params.curves_size() != ec_params.hash_methods_size()) {
    response.mutable_header()->set_error_code(
        org::interconnection::INVALID_REQUEST);
    response.mutable_header()->set_error_msg(
        "EcdhPsiParamsProposal curves and hash_methods length not equal");
    return response;
  }

  auto curve_name = CurveType_Name(CURVE_25519);
  std::string hash_name = "SHA_256";
  int curve_idx = 0;
  for (; curve_idx < ec_params.curves_size(); ++curve_idx) {
    if (ec_params.curves(curve_idx) == curve_name &&
        ec_params.hash_methods(curve_idx) == hash_name) {
      break;
    }
  }

  if (curve_idx >= ec_params.curves_size()) {
    response.mutable_header()->set_error_code(
        org::interconnection::UNSUPPORTED_PARAMS);
    response.mutable_header()->set_error_msg(
        "Secretflow only support algo ECDH_PSI_2PC(CURVE_25519, SHA_256)");
    return response;
  }

  response.mutable_header()->set_error_code(org::interconnection::OK);
  response.mutable_header()->clear_error_msg();
  response.set_algo(psi_name);

  org::interconnection::algos::psi::EcdhPsiParamsResult ec_params_res;
  ec_params_res.set_curve(curve_name);
  ec_params_res.set_hash_method(hash_name);
  YACL_ENFORCE(response.mutable_algo_params()->PackFrom(ec_params_res),
               "handshake: pack EcdhPsiParamsResult fail");
  return response;
}

void BucketPsi::Handshake(uint64_t self_items_count) {
  YACL_ENFORCE(config_.psi_type() == PsiType::ECDH_PSI_2PC,
               "IC mode only support ECDH_PSI_2PC");
  YACL_ENFORCE(lctx_->WorldSize() == 2, "ECDH_PSI_2PC only support 2PC");

  if (lctx_->Rank() == 0) {
    org::interconnection::algos::psi::HandshakeRequest handshake_request;
    handshake_request.set_version(1);  // The version is always 1 currently
    handshake_request.set_item_num(self_items_count);
    if (config_.broadcast_result()) {
      handshake_request.set_result_to_rank(-1);
    } else {
      handshake_request.set_result_to_rank(config_.receiver_rank());
    }

    // gen ecc params
    org::interconnection::algos::psi::EcdhPsiParamsProposal ec_params;
    ec_params.add_supported_versions(1);  // The version is always 1 currently
    ec_params.add_curves(CurveType_Name(CURVE_25519));
    ec_params.add_hash_methods("SHA_256");
    ec_params.add_curves(CurveType_Name(CURVE_SM2));
    ec_params.add_hash_methods("SHA_256");

    handshake_request.add_supported_algos(PsiType_Name(PsiType::ECDH_PSI_2PC));
    YACL_ENFORCE(handshake_request.add_algo_params()->PackFrom(ec_params),
                 "handshake: pack message fail");

    // send
    lctx_->Send(1, handshake_request.SerializeAsString(), "Handshake");
    // recv HandshakeResponse
    auto buf = lctx_->Recv(1, "Handshake_response");
    org::interconnection::algos::psi::HandshakeResponse response;
    YACL_ENFORCE(response.ParseFromArray(buf.data(), buf.size()),
                 "handshake: parse HandshakeResponse from array fail");
    YACL_ENFORCE(response.header().error_code() == org::interconnection::OK,
                 "{}", response.header().error_msg());

  } else {
    auto buf = lctx_->Recv(0, "Handshake");
    org::interconnection::algos::psi::HandshakeRequest handshake_request;
    YACL_ENFORCE(handshake_request.ParseFromArray(buf.data(), buf.size()),
                 "handshake: parse from array fail");

    auto response = CheckSelectAlgo(handshake_request);
    // check algo and params
    response.set_item_count(self_items_count);
    // check receiver rank
    int32_t my_result =
        config_.broadcast_result() ? -1 : config_.receiver_rank();
    if (handshake_request.result_to_rank() != my_result) {
      response.mutable_header()->set_error_code(
          org::interconnection::HANDSHAKE_REFUSED);
      response.mutable_header()->set_error_msg(
          fmt::format("result_to_rank mismatch, my:{}, peer:{}", my_result,
                      handshake_request.result_to_rank()));
    }

    lctx_->SendAsync(0, response.SerializeAsString(), "Handshake_response");
    YACL_ENFORCE(response.header().error_code() == org::interconnection::OK,
                 "{}", response.header().error_msg());
  }
}

std::vector<uint64_t> BucketPsi::RunPsi(uint64_t self_items_count) {
  SPDLOG_INFO("Run psi protocol={}, self_items_count={}", config_.psi_type(),
              self_items_count);

  if (ic_mode_) {
    Handshake(self_items_count);
  }

  if (config_.psi_type() == PsiType::ECDH_PSI_2PC) {
    EcdhPsiOptions psi_options;
    if (config_.curve_type() == CurveType::CURVE_INVALID_TYPE) {
      YACL_THROW("Unsupported curve type");
    }
    psi_options.ecc_cryptor = CreateEccCryptor(config_.curve_type());
    psi_options.link_ctx = lctx_;
    psi_options.target_rank = static_cast<size_t>(config_.receiver_rank());
    if (config_.broadcast_result()) {
      psi_options.target_rank = yacl::link::kAllRank;
    }
    psi_options.ic_mode = ic_mode_;

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
