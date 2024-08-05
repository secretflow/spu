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

#include <memory>

#include "libspu/psi/core/ecdh_oprf/ecdh_oprf.h"

namespace spu::psi {

std::unique_ptr<IEcdhOprfServer> CreateEcdhOprfServer(
    yacl::ByteContainerView private_key, OprfType oprf_type,
    CurveType curve_type);

std::unique_ptr<IEcdhOprfServer> CreateEcdhOprfServer(OprfType oprf_type,
                                                      CurveType curve_type);

std::unique_ptr<IEcdhOprfClient> CreateEcdhOprfClient(OprfType oprf_type,
                                                      CurveType curve_type);

std::unique_ptr<IEcdhOprfClient> CreateEcdhOprfClient(
    yacl::ByteContainerView private_key, OprfType oprf_type,
    CurveType curve_type);

}  // namespace spu::psi
