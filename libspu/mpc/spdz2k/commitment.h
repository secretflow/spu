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

#include "yacl/crypto/base/hash/hash_interface.h"
#include "yacl/link/link.h"

// TODO: move commitment scheme to yacl
namespace spu::mpc {

std::string commit(size_t send_player, absl::string_view msg,
                   absl::string_view r, size_t hash_len = 32,
                   yacl::crypto::HashAlgorithm hash_type =
                       yacl::crypto::HashAlgorithm::BLAKE3);

bool commit_and_open(const std::shared_ptr<yacl::link::Context>& lctx,
                     const std::string& z_str,
                     std::vector<std::string>* z_strs);

}  // namespace spu::mpc
