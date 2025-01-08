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

#pragma once

#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/spdz2k/ot/ferret.h"

namespace spu::mpc::spdz2k {

class BasicOTProtocols {
 public:
  explicit BasicOTProtocols(std::shared_ptr<Communicator> conn);

  ~BasicOTProtocols();

  std::unique_ptr<BasicOTProtocols> Fork();

  int Rank() const;

  std::shared_ptr<FerretOT> GetSenderCOT() { return ferret_sender_; }

  std::shared_ptr<FerretOT> GetReceiverCOT() { return ferret_receiver_; }

  void Flush();

 private:
  std::shared_ptr<Communicator> conn_;
  std::shared_ptr<FerretOT> ferret_sender_;
  std::shared_ptr<FerretOT> ferret_receiver_;
};

}  // namespace spu::mpc::spdz2k
