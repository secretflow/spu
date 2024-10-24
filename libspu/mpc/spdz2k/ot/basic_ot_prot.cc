// Copyright 2023 Ant Group Co., Ltd.
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

#include "libspu/mpc/spdz2k/ot/basic_ot_prot.h"

namespace spu::mpc::spdz2k {

BasicOTProtocols::BasicOTProtocols(std::shared_ptr<Communicator> conn)
    : conn_(std::move(conn)) {
  SPU_ENFORCE(conn_ != nullptr);
  if (conn_->getRank() == 0) {
    ferret_sender_ = std::make_shared<FerretOT>(conn_, true);
    ferret_receiver_ = std::make_shared<FerretOT>(conn_, false);
  } else {
    ferret_receiver_ = std::make_shared<FerretOT>(conn_, false);
    ferret_sender_ = std::make_shared<FerretOT>(conn_, true);
  }
}

std::unique_ptr<BasicOTProtocols> BasicOTProtocols::Fork() {
  // TODO: we can take from cached ROTs from the caller
  auto conn = std::make_shared<Communicator>(conn_->lctx()->Spawn());
  return std::make_unique<BasicOTProtocols>(conn);
}

BasicOTProtocols::~BasicOTProtocols() { Flush(); }

void BasicOTProtocols::Flush() {
  if (ferret_sender_) {
    ferret_sender_->Flush();
  }
}

int BasicOTProtocols::Rank() const { return ferret_sender_->Rank(); }

}  // namespace spu::mpc::spdz2k
