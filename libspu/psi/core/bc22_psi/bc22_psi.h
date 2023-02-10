// Copyright 2021 Ant Group Co., Ltd.
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

#include "absl/types/span.h"
#include "yacl/base/int128.h"

#include "libspu/psi/core/bc22_psi/generalized_cuckoo_hash.h"
#include "libspu/psi/core/communication.h"
#include "libspu/psi/utils/serialize.h"

namespace spu::psi {

// PSI from Pseudorandom Correlation Generators
// https://eprint.iacr.org/2022/334
// Journées C2 2022 – Hendaye April 10-15 2022 (inria.fr)

// VOLE
// Wolverine: Fast, Scalable, and Communication-Efficient
// Zero-Knowledge Proofs for Boolean and Arithmetic Circuits
// https://eprint.iacr.org/2020/925
// https://github.com/emp-toolkit/emp-zk

class Bc22PcgPsi {
 public:
  Bc22PcgPsi(std::shared_ptr<yacl::link::Context> link_ctx, PsiRoleType role);

  void RunPsi(absl::Span<const std::string> items);

  std::vector<std::string> GetIntersection() {
    if (role_ == PsiRoleType::Receiver) {
      return results_;
    } else {
      SPU_THROW("Bc22PcgPsi only Receiver get intersection");
    }
  }

 private:
  // exchange items number, compute compare bytes size
  void ExchangeItemsNumber(size_t self_item_num);

  // mBaRK-OPRF  sender/receiver
  std::string RunmBaRKOprfSender(absl::Span<const std::string> items,
                                 size_t compare_bytes_size);

  std::vector<std::string> RunmBaRKOprfReceiver(
      absl::Span<const std::string> items, size_t compare_bytes_size);

  // send/recv oprf
  void PcgPsiSendOprf(absl::Span<const std::string> items,
                      const std::string &oprfs, size_t compare_bytes_size);

  void PcgPsiRecvOprf(absl::Span<const std::string> items,
                      const std::vector<std::string> &oprf_encode_vec, size_t);

  // cuckoo_options
  CuckooIndex::Options cuckoo_options_;

  // Provides the link for the rank world.
  std::shared_ptr<yacl::link::Context> link_ctx_;

  // psi role sender/receiver
  PsiRoleType role_;

  // batch send/recv data size
  size_t batch_size_;

  // peer's item size
  size_t peer_items_num_ = 0;

  // intersection result
  std::vector<std::string> results_;
};

}  // namespace spu::psi
