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

#include "yacl/link/link.h"

#include "libspu/psi/core/labeled_psi/sender_db.h"

#include "libspu/pir/pir.pb.h"

namespace spu::pir {

PirResultReport PirSetup(const PirSetupConfig &config);

PirResultReport PirServer(const std::shared_ptr<yacl::link::Context> &link_ctx,
                          const PirServerConfig &config);

PirResultReport PirClient(const std::shared_ptr<yacl::link::Context> &link_ctx,
                          const PirClientConfig &config);

PirResultReport PirMemoryServer(
    const std::shared_ptr<yacl::link::Context> &link_ctx,
    const PirSetupConfig &config);

// pir protocol based on labeled psi
PirResultReport LabeledPirSetup(const PirSetupConfig &config);

PirResultReport LabeledPirServer(
    const std::shared_ptr<yacl::link::Context> &link_ctx,
    const PirServerConfig &config);

PirResultReport LabeledPirMemoryServer(
    const std::shared_ptr<yacl::link::Context> &link_ctx,
    const PirServerConfig &config);

PirResultReport LabeledPirClient(
    const std::shared_ptr<yacl::link::Context> &link_ctx,
    const PirClientConfig &config);

}  // namespace spu::pir
