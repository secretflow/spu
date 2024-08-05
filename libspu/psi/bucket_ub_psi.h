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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "yacl/link/link.h"

#include "libspu/core/prelude.h"
#include "libspu/psi/core/ecdh_oprf_psi.h"

#include "libspu/psi/psi.pb.h"

namespace spu::psi {

std::pair<std::vector<uint64_t>, size_t> UbPsi(
    BucketPsiConfig config, std::shared_ptr<yacl::link::Context> lctx);

std::pair<std::vector<uint64_t>, size_t> UbPsiServerGenCache(
    BucketPsiConfig config, std::shared_ptr<yacl::link::Context> lctx,
    const spu::psi::EcdhOprfPsiOptions& psi_options);

std::pair<std::vector<uint64_t>, size_t> UbPsiClientTransferCache(
    BucketPsiConfig config, std::shared_ptr<yacl::link::Context> lctx,
    const spu::psi::EcdhOprfPsiOptions& psi_options,
    const std::string& tmp_dir);

std::pair<std::vector<uint64_t>, size_t> UbPsiServerTransferCache(
    BucketPsiConfig config, std::shared_ptr<yacl::link::Context> lctx,
    const spu::psi::EcdhOprfPsiOptions& psi_options,
    const std::string& tmp_dir);

std::pair<std::vector<uint64_t>, size_t> UbPsiClientShuffleOnline(
    BucketPsiConfig config, std::shared_ptr<yacl::link::Context> lctx,
    const spu::psi::EcdhOprfPsiOptions& psi_options,
    const std::string& tmp_dir);

std::pair<std::vector<uint64_t>, size_t> UbPsiServerShuffleOnline(
    BucketPsiConfig config, std::shared_ptr<yacl::link::Context> lctx,
    const spu::psi::EcdhOprfPsiOptions& psi_options,
    const std::string& tmp_dir);

std::pair<std::vector<uint64_t>, size_t> UbPsiClientOffline(
    BucketPsiConfig config, std::shared_ptr<yacl::link::Context> lctx,
    const spu::psi::EcdhOprfPsiOptions& psi_options,
    const std::string& tmp_dir);

std::pair<std::vector<uint64_t>, size_t> UbPsiServerOffline(
    BucketPsiConfig config, std::shared_ptr<yacl::link::Context> lctx,
    const spu::psi::EcdhOprfPsiOptions& psi_options,
    const std::string& tmp_dir);

std::pair<std::vector<uint64_t>, size_t> UbPsiClientOnline(
    BucketPsiConfig config, std::shared_ptr<yacl::link::Context> lctx,
    const spu::psi::EcdhOprfPsiOptions& psi_options,
    const std::string& tmp_dir);

std::pair<std::vector<uint64_t>, size_t> UbPsiServerOnline(
    BucketPsiConfig config, std::shared_ptr<yacl::link::Context> lctx,
    const spu::psi::EcdhOprfPsiOptions& psi_options,
    const std::string& tmp_dir);

}  // namespace spu::psi
