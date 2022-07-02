// Copyright 2021 Ant Group Co., Ltd.
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

#include "spu/psi/cryptor/ecdh_oprf/basic_ecdh_oprf.h"
#include "spu/psi/cryptor/ecdh_oprf/ecdh_oprf.h"

namespace spu {

std::unique_ptr<IEcdhOprfServer> CreateEcdhOprfServer(
    yasl::ByteContainerView private_key, OprfType oprf_type,
    CurveType curve_type) {
  std::unique_ptr<IEcdhOprfServer> server;

  switch (oprf_type) {
    case OprfType::Basic: {
      switch (curve_type) {
        case CurveType::CurveFourQ: {
          SPDLOG_INFO("use fourq");
          server = std::make_unique<FourQBasicEcdhOprfServer>(private_key);
          break;
        }
        case CurveType::CurveSecp256k1:
        case CurveType::CurveSm2: {
          SPDLOG_INFO("use curve sm2/secp256k1");
          server =
              std::make_unique<BasicEcdhOprfServer>(private_key, curve_type);
          break;
        }
        default:
          YASL_THROW("unknown support Curve type: {}",
                     static_cast<int>(curve_type));
          break;
      }

      break;
    }
    default:
      YASL_THROW("unknown Oprf type: {}", static_cast<int>(oprf_type));
      break;
  }
  YASL_ENFORCE(server != nullptr, "EcdhOprfServer should not be nullptr");

  return server;
}

std::unique_ptr<IEcdhOprfServer> CreateEcdhOprfServer(OprfType oprf_type,
                                                      CurveType curve_type) {
  std::unique_ptr<IEcdhOprfServer> server;

  switch (oprf_type) {
    case OprfType::Basic: {
      switch (curve_type) {
        case CurveType::CurveFourQ: {
          SPDLOG_INFO("use fourq");
          server = std::make_unique<FourQBasicEcdhOprfServer>();
          break;
        }
        case CurveType::CurveSecp256k1:
        case CurveType::CurveSm2: {
          SPDLOG_INFO("use curve sm2/secp256k1");
          server = std::make_unique<BasicEcdhOprfServer>(curve_type);
          break;
        }
        default:
          YASL_THROW("unknown support Curve type: {}",
                     static_cast<int>(curve_type));
          break;
      }
      break;
    }
    default:
      YASL_THROW("unknown Oprf type: {}", static_cast<int>(oprf_type));
  }
  YASL_ENFORCE(server != nullptr, "EcdhOprfServer should not be nullptr");

  return server;
}

std::unique_ptr<IEcdhOprfClient> CreateEcdhOprfClient(OprfType oprf_type,
                                                      CurveType curve_type) {
  std::unique_ptr<IEcdhOprfClient> client;

  switch (oprf_type) {
    case OprfType::Basic: {
      switch (curve_type) {
        case CurveType::CurveFourQ: {
          client = std::make_unique<FourQBasicEcdhOprfClient>();
          break;
        }
        case CurveType::CurveSecp256k1:
        case CurveType::CurveSm2: {
          client = std::make_unique<BasicEcdhOprfClient>(curve_type);
          break;
        }
        default:
          YASL_THROW("unknown support Curve type: {}",
                     static_cast<int>(curve_type));
          break;
      }
      break;
    }
  }

  YASL_ENFORCE(client != nullptr, "EcdhOprfClient should not be nullptr");

  return client;
}
}  // namespace spu