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

#include "libspu/psi/core/ecdh_oprf/ecdh_oprf_selector.h"

#include "libspu/core/platform_utils.h"
#include "libspu/psi/core/ecdh_oprf/basic_ecdh_oprf.h"
#include "libspu/psi/core/ecdh_oprf/ecdh_oprf.h"

namespace spu::psi {

std::unique_ptr<IEcdhOprfServer> CreateEcdhOprfServer(
    yacl::ByteContainerView private_key, OprfType oprf_type,
    CurveType curve_type) {
  std::unique_ptr<IEcdhOprfServer> server;

  switch (oprf_type) {
    case OprfType::Basic: {
      switch (curve_type) {
        case CurveType::CURVE_FOURQ: {
          SPDLOG_INFO("use fourq");
#ifdef __x86_64__
          if (hasAVX2()) {
#endif
            server = std::make_unique<FourQBasicEcdhOprfServer>(private_key);
#ifdef __x86_64__
          }
#endif
          break;
        }
        case CurveType::CURVE_SECP256K1:
          [[fallthrough]];
        case CurveType::CURVE_SM2: {
          SPDLOG_INFO("use curve sm2/secp256k1");
          server =
              std::make_unique<BasicEcdhOprfServer>(private_key, curve_type);
          break;
        }
        default:
          SPU_THROW("unknown support Curve type: {}",
                    static_cast<int>(curve_type));
          break;
      }

      break;
    }
    default:
      SPU_THROW("unknown Oprf type: {}", static_cast<int>(oprf_type));
      break;
  }
  SPU_ENFORCE(server != nullptr, "EcdhOprfServer should not be nullptr");

  return server;
}

std::unique_ptr<IEcdhOprfServer> CreateEcdhOprfServer(OprfType oprf_type,
                                                      CurveType curve_type) {
  std::unique_ptr<IEcdhOprfServer> server;

  switch (oprf_type) {
    case OprfType::Basic: {
      switch (curve_type) {
        case CurveType::CURVE_FOURQ: {
          SPDLOG_INFO("use fourq");
#ifdef __x86_64__
          if (hasAVX2()) {
#endif
            server = std::make_unique<FourQBasicEcdhOprfServer>();
#ifdef __x86_64__
          }
#endif
          break;
        }
        case CurveType::CURVE_SECP256K1:
          [[fallthrough]];
        case CurveType::CURVE_SM2: {
          SPDLOG_INFO("use curve sm2/secp256k1");
          server = std::make_unique<BasicEcdhOprfServer>(curve_type);
          break;
        }
        default:
          SPU_THROW("unknown support Curve type: {}",
                    static_cast<int>(curve_type));
          break;
      }
      break;
    }
    default:
      SPU_THROW("unknown Oprf type: {}", static_cast<int>(oprf_type));
  }
  SPU_ENFORCE(server != nullptr, "EcdhOprfServer should not be nullptr");

  return server;
}

std::unique_ptr<IEcdhOprfClient> CreateEcdhOprfClient(OprfType oprf_type,
                                                      CurveType curve_type) {
  std::unique_ptr<IEcdhOprfClient> client;

  switch (oprf_type) {
    case OprfType::Basic: {
      switch (curve_type) {
        case CurveType::CURVE_FOURQ: {
#ifdef __x86_64__
          if (hasAVX2()) {
#endif
            client = std::make_unique<FourQBasicEcdhOprfClient>();
#ifdef __x86_64__
          }
#endif
          break;
        }
        case CurveType::CURVE_SECP256K1:
          [[fallthrough]];
        case CurveType::CURVE_SM2: {
          client = std::make_unique<BasicEcdhOprfClient>(curve_type);
          break;
        }
        default:
          SPU_THROW("unknown support Curve type: {}",
                    static_cast<int>(curve_type));
          break;
      }
      break;
    }
  }

  SPU_ENFORCE(client != nullptr, "EcdhOprfClient should not be nullptr");

  return client;
}

std::unique_ptr<IEcdhOprfClient> CreateEcdhOprfClient(
    yacl::ByteContainerView private_key, OprfType oprf_type,
    CurveType curve_type) {
  std::unique_ptr<IEcdhOprfClient> client;

  switch (oprf_type) {
    case OprfType::Basic: {
      switch (curve_type) {
        case CurveType::CURVE_FOURQ: {
#ifdef __x86_64__
          if (hasAVX2()) {
#endif
            client = std::make_unique<FourQBasicEcdhOprfClient>(private_key);
#ifdef __x86_64__
          }
#endif
          break;
        }
        case CurveType::CURVE_SECP256K1:
          [[fallthrough]];
        case CurveType::CURVE_SM2: {
          client =
              std::make_unique<BasicEcdhOprfClient>(curve_type, private_key);
          break;
        }
        default:
          SPU_THROW("unknown support Curve type: {}",
                    static_cast<int>(curve_type));
          break;
      }
      break;
    }
  }

  SPU_ENFORCE(client != nullptr, "EcdhOprfClient should not be nullptr");

  return client;
}

}  // namespace spu::psi
