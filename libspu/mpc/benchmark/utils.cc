
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

#include "utils.h"

llvm::cl::opt<uint32_t> cli_rank("rank", llvm::cl::init(0),
                                 llvm::cl::desc("self rank, starts with 0"));
llvm::cl::opt<std::string> cli_parties(
    "parties",
    llvm::cl::desc("server list, format: host1:port1[,host2:port2, ...]"));
llvm::cl::opt<uint32_t> cli_party_num("party_num", llvm::cl::init(0),
                                      llvm::cl::desc("server numbers"));
llvm::cl::opt<std::string> cli_protocol(
    "protocol",
    llvm::cl::desc("benchmark protocol, supported protocols: semi2k / aby3"));
llvm::cl::opt<uint32_t> cli_numel(
    "numel", llvm::cl::init(7), llvm::cl::desc("number of benchmark elements"));
llvm::cl::opt<uint32_t> cli_shiftbit("shiftbit", llvm::cl::init(2),
                                     llvm::cl::desc("benchmark shift bit"));
