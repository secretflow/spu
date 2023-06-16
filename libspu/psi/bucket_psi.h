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

#include <memory>
#include <string>
#include <vector>

#include "yacl/link/link.h"

#include "libspu/core/prelude.h"
#include "libspu/psi/memory_psi.h"
#include "libspu/psi/utils/csv_checker.h"
#include "libspu/psi/utils/hash_bucket_cache.h"

#include "libspu/psi/psi.pb.h"

namespace spu::psi {

class BucketPsi {
 public:
  // ic_mode: 互联互通模式，对方可以是非隐语应用
  // Interconnection mode, the other side can be non-secretflow application
  explicit BucketPsi(BucketPsiConfig config,
                     std::shared_ptr<yacl::link::Context> lctx,
                     bool ic_mode = false);
  ~BucketPsi() = default;

  PsiResultReport Run();

  // unbalanced get items_count when RunPSI
  // other psi use sanity check get items_count
  // TODO: sanity check affects performance maybe optional
  std::vector<uint64_t> RunPsi(uint64_t& self_items_count);

  std::unique_ptr<CsvChecker> CheckInput();

  void ProduceOutput(bool digest_equal, std::vector<uint64_t>& indices,
                     PsiResultReport& report);

 private:
  void Init();

  std::vector<uint64_t> RunBucketPsi(uint64_t self_items_count);

  // the item order of `item_data_list` and `item_list` needs to be the same
  static void GetResultIndices(
      const std::vector<std::string>& item_data_list,
      const std::vector<HashBucketCache::BucketItem>& item_list,
      std::vector<std::string>& result_list, std::vector<uint64_t>* indices);

  BucketPsiConfig config_;
  bool ic_mode_;

  std::shared_ptr<yacl::link::Context> lctx_;

  std::vector<std::string> selected_fields_;

  std::unique_ptr<MemoryPsi> mem_psi_;
};

}  // namespace spu::psi
