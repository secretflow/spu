// Copyright 2023 zhangwfjh
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

#include <array>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "heu/library/algorithms/paillier_zahlen/paillier.h"
#include "yacl/link/link.h"

#include "libspu/core/prelude.h"

namespace spu::psi {

// Efficient Private Matching and Set Intersection
// http://www.pinkas.net/PAPERS/FNP04.pdf

using namespace heu::lib::algorithms::paillier_z;

class Party {
 public:
  struct Options {
    std::shared_ptr<yacl::link::Context> link_ctx;
    size_t leader_rank;
  };
  static constexpr size_t BinSize{5};
  using SecretPolynomial = std::vector<std::array<Ciphertext, BinSize>>;
  using Share = std::vector<size_t>;

  Party(const Options& options);
  virtual std::vector<std::string> Run(const std::vector<std::string>& inputs);

 private:
  void BroadcastPubKey();
  void SendEncryptedSet(const std::vector<size_t>& items) const;
  std::vector<SecretPolynomial> RecvEncryptedSet(size_t count) const;
  std::vector<Share> ZeroSharing(size_t count) const;
  std::vector<Share> SwapShares(const std::vector<Share>& shares) const;
  void SwapShares(const std::vector<Share>& shares,
                  const std::vector<size_t>& items,
                  const std::vector<SecretPolynomial>& hashings) const;
  Share AggregateShare(const std::vector<Share>& shares) const;
  std::vector<size_t> GetIntersection(const std::vector<size_t>& items,
                                      const Share& share) const;

  Options options_;
  std::vector<std::shared_ptr<Encryptor>> encryptors_;
  std::shared_ptr<Decryptor> decryptor_;

  // (ctx, world_size, my_rank, leader_rank)
  auto CollectContext() const {
    return std::make_tuple(options_.link_ctx, options_.link_ctx->WorldSize(),
                           options_.link_ctx->Rank(), options_.leader_rank);
  }
};

}  // namespace spu::psi
