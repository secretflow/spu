
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

#include "llvm/Support/CommandLine.h"

extern llvm::cl::opt<uint32_t> cli_rank;
extern llvm::cl::opt<std::string> cli_parties;
extern llvm::cl::opt<uint32_t> cli_party_num;
extern llvm::cl::opt<uint32_t> cli_numel;
extern llvm::cl::opt<uint32_t> cli_shiftbit;
extern llvm::cl::opt<std::string> cli_protocol;

namespace spu::mpc::bench {

const std::string kTwoPartyHosts = "127.0.0.1:9540,127.0.0.1:9541";
const std::string kThreePartyHosts =
    "127.0.0.1:9540,127.0.0.1:9541,127.0.0.1:9542";

}  // namespace spu::mpc::bench