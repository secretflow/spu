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

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "yacl/link/link.h"

namespace spu::psi {

//
// Compact and Malicious Private Set Intersection for Small Sets
// https://eprint.iacr.org/2021/1159.pdf
// opensource code
// https://github.com/osu-crypto/Mini-PSI
//
void MiniPsiSend(const std::shared_ptr<yacl::link::Context>& link_ctx,
                 const std::vector<std::string>& items);

std::vector<std::string> MiniPsiRecv(
    const std::shared_ptr<yacl::link::Context>& link_ctx,
    const std::vector<std::string>& items);

// use cuckoo hash to batch process
void MiniPsiSendBatch(const std::shared_ptr<yacl::link::Context>& link_ctx,
                      const std::vector<std::string>& items);

std::vector<std::string> MiniPsiRecvBatch(
    const std::shared_ptr<yacl::link::Context>& link_ctx,
    const std::vector<std::string>& items);
}  // namespace spu::psi