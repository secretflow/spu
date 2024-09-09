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

#include "libspu/mpc/semi2k/beaver/beaver_impl/ttp_server/beaver_server.h"

#include <algorithm>
#include <vector>

#include "absl/strings/ascii.h"
#include "spdlog/spdlog.h"
#include "yacl/base/byte_container_view.h"
#include "yacl/base/exception.h"
#include "yacl/crypto/pke/asymmetric_sm2_crypto.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/mpc/common/prg_tensor.h"
#include "libspu/mpc/semi2k/beaver/beaver_impl/trusted_party/trusted_party.h"

#include "libspu/mpc/semi2k/beaver/beaver_impl/ttp_server/service.pb.h"

namespace brpc {

DECLARE_uint64(max_body_size);
DECLARE_int64(socket_max_unwritten_bytes);

}  // namespace brpc

namespace spu::mpc::semi2k::beaver::ttp_server {

namespace {

inline size_t CeilDiv(size_t a, size_t b) { return (a + b - 1) / b; }

class DecryptError : public yacl::Exception {
  using yacl::Exception::Exception;
};

template <class AdjustRequest>
std::tuple<std::vector<TrustedParty::Operand>,
           std::vector<std::vector<PrgSeed>>, size_t>
BuildOperand(
    const AdjustRequest& req, uint32_t field_size,
    const std::unique_ptr<yacl::crypto::AsymmetricDecryptor>& decryptor) {
  std::vector<TrustedParty::Operand> ops;
  std::vector<std::vector<PrgSeed>> seeds;
  size_t pad_length = 0;

  FieldType field;
  switch (field_size) {
    case 4: {
      field = FieldType::FM32;
      break;
    }
    case 8: {
      field = FieldType::FM64;
      break;
    }
    case 16: {
      field = FieldType::FM128;
      break;
    }
    default: {
      YACL_THROW("Unsupported field size {}", field_size);
    }
  }
  auto type = static_cast<FieldType>(field);

  std::vector<Shape> shapes;
  if constexpr (std::is_same_v<AdjustRequest, AdjustDotRequest>) {
    shapes.resize(req.prg_inputs().size());
    SPU_ENFORCE(shapes.size() == 3);
    shapes[0] = {static_cast<int64_t>(req.m()), static_cast<int64_t>(req.k())};
    shapes[1] = {static_cast<int64_t>(req.k()), static_cast<int64_t>(req.n())};
    shapes[2] = {static_cast<int64_t>(req.m()), static_cast<int64_t>(req.n())};
    for (int i = 0; i < 3; i++) {
      if (req.transpose_inputs()[i]) {
        std::reverse(shapes[i].begin(), shapes[i].end());
      }
    }
  } else {
    const auto& prg = req.prg_inputs()[0];
    auto buffer_len = prg.buffer_len();
    SPU_ENFORCE(buffer_len > 0);
    if constexpr (std::is_same_v<AdjustRequest, AdjustAndRequest>) {
      int64_t elsize = CeilDiv(buffer_len, field_size);
      size_t pad_buffer_len = elsize * field_size;
      pad_length = pad_buffer_len - buffer_len;
      shapes.resize(req.prg_inputs().size(), {elsize});
    } else {
      SPU_ENFORCE(buffer_len % field_size == 0);
      int64_t elsize = buffer_len / field_size;
      shapes.resize(req.prg_inputs().size(), {elsize});
    }
  }

  std::unordered_map<std::string, PrgSeed> decryptor_cache;
  auto try_decrypt = [&](const std::string& ciphertext) {
    auto it = decryptor_cache.find(ciphertext);
    if (it != decryptor_cache.end()) {
      return it->second;
    }

    std::vector<uint8_t> plaintext;
    try {
      plaintext = decryptor->Decrypt(ciphertext);
    } catch (const std::exception& e) {
      throw DecryptError(fmt::format("Decrypt Error {}", e.what()));
    }
    SPU_ENFORCE(plaintext.size() == sizeof(PrgSeed));
    PrgSeed seed;
    // FIXME: The client and server must use the same endianness order.
    std::memcpy(&seed, plaintext.data(), sizeof(PrgSeed));
    decryptor_cache.emplace(ciphertext, seed);
    return seed;
  };

  for (int64_t i = 0; i < req.prg_inputs().size(); i++) {
    const auto& prg = req.prg_inputs()[i];
    auto prg_count = prg.prg_count();
    auto buffer_len = prg.buffer_len();
    const auto& shape = shapes[i];
    SPU_ENFORCE(shape.numel() > 0);
    SPU_ENFORCE(static_cast<uint64_t>(shape.numel()) * field_size ==
                buffer_len + pad_length);
    std::vector<PrgSeed> seed;
    for (const auto& c : prg.encrypted_seeds()) {
      seed.push_back(try_decrypt(c));
    }
    seeds.emplace_back(std::move(seed));
    ops.push_back(
        TrustedParty::Operand{{shape, type, prg_count}, seeds.back()});
  }

  if constexpr (std::is_same_v<AdjustRequest, AdjustDotRequest>) {
    for (int i = 0; i < 3; i++) {
      ops[i].transpose = req.transpose_inputs()[i];
    }
  }

  return {std::move(ops), std::move(seeds), pad_length};
}

std::vector<yacl::Buffer> StripNdArray(std::vector<NdArrayRef>& nds,
                                       size_t pad_length) {
  std::vector<yacl::Buffer> ret;
  ret.reserve(nds.size());

  auto if_pad = [&](NdArrayRef& nd) {
    yacl::Buffer buf = std::move(*nd.buf());
    if (pad_length > 0) {
      buf.resize(buf.size() - pad_length);
    }
    return buf;
  };

  for (auto& nd : nds) {
    ret.push_back(if_pad(nd));
  }

  return ret;
}

template <class T>
struct dependent_false : std::false_type {};

template <class AdjustRequest>
std::vector<yacl::Buffer> AdjustImpl(
    const AdjustRequest& req,
    const std::unique_ptr<yacl::crypto::AsymmetricDecryptor>& decryptor) {
  std::vector<NdArrayRef> ret;
  size_t field_size;
  if constexpr (std::is_same_v<AdjustRequest, AdjustAndRequest>) {
    field_size = 128 / 8;
  } else {
    field_size = req.field_size();
  }
  auto [ops, seeds, pad_length] = BuildOperand(req, field_size, decryptor);

  if constexpr (std::is_same_v<AdjustRequest, AdjustMulRequest>) {
    auto adjust = TrustedParty::adjustMul(ops);
    ret.push_back(std::move(adjust));
  } else if constexpr (std::is_same_v<AdjustRequest, AdjustSquareRequest>) {
    auto adjust = TrustedParty::adjustSquare(ops);
    ret.push_back(std::move(adjust));
  } else if constexpr (std::is_same_v<AdjustRequest, AdjustDotRequest>) {
    auto adjust = TrustedParty::adjustDot(ops);
    ret.push_back(std::move(adjust));
  } else if constexpr (std::is_same_v<AdjustRequest, AdjustAndRequest>) {
    auto adjust = TrustedParty::adjustAnd(ops);
    ret.push_back(std::move(adjust));
  } else if constexpr (std::is_same_v<AdjustRequest, AdjustTruncRequest>) {
    auto adjust = TrustedParty::adjustTrunc(ops, req.bits());
    ret.push_back(std::move(adjust));
  } else if constexpr (std::is_same_v<AdjustRequest, AdjustTruncPrRequest>) {
    auto adjust = TrustedParty::adjustTruncPr(ops, req.bits());
    ret.push_back(std::move(std::get<0>(adjust)));
    ret.push_back(std::move(std::get<1>(adjust)));
  } else if constexpr (std::is_same_v<AdjustRequest, AdjustRandBitRequest>) {
    auto adjust = TrustedParty::adjustRandBit(ops);
    ret.push_back(std::move(adjust));
  } else if constexpr (std::is_same_v<AdjustRequest, AdjustEqzRequest>) {
    auto adjust = TrustedParty::adjustEqz(ops);
    ret.push_back(std::move(adjust));
  } else if constexpr (std::is_same_v<AdjustRequest, AdjustPermRequest>) {
    std::vector<int64_t> pv(req.perm_vec().begin(), req.perm_vec().end());
    auto adjust = TrustedParty::adjustPerm(ops, pv);
    ret.push_back(std::move(adjust));
  } else {
    static_assert(dependent_false<AdjustRequest>::value,
                  "not support AdjustRequest type");
  }

  return StripNdArray(ret, pad_length);
}

}  // namespace

class ServiceImpl final : public BeaverService {
 private:
  std::unique_ptr<yacl::crypto::AsymmetricDecryptor> decryptor_;

 public:
  ServiceImpl(const std::string& asym_crypto_schema,
              yacl::ByteContainerView server_private_key) {
    auto lower_schema = absl::AsciiStrToLower(asym_crypto_schema);
    if (lower_schema == "sm2") {
      decryptor_ =
          std::make_unique<yacl::crypto::Sm2Decryptor>(server_private_key);
    } else {
      SPU_THROW("not support asym_crypto_schema {}", asym_crypto_schema);
    }
  }

  template <class AdjustRequest>
  void Adjust(::google::protobuf::RpcController* controller,
              const AdjustRequest* req, AdjustResponse* rsp,
              ::google::protobuf::Closure* done) const {
    brpc::ClosureGuard done_guard(done);
    auto* cntl = static_cast<brpc::Controller*>(controller);
    std::string client_side(butil::endpoint2str(cntl->remote_side()).c_str());

    std::vector<yacl::Buffer> adjusts;
    try {
      adjusts = AdjustImpl(*req, decryptor_);
    } catch (const DecryptError& e) {
      auto err = fmt::format("Seed Decrypt error {}", e.what());
      SPDLOG_ERROR("{}, client {}", err, client_side);
      rsp->set_code(ErrorCode::SeedDecryptError);
      rsp->set_message(err);
      return;
    } catch (const std::exception& e) {
      auto err = fmt::format("adjust error {}", e.what());
      SPDLOG_ERROR("{}, client {}", err, client_side);
      rsp->set_code(ErrorCode::OpAdjustError);
      rsp->set_message(err);
      return;
    }

    rsp->set_code(ErrorCode::OK);
    for (auto& a : adjusts) {
      // FIXME: TTP adjuster server and client MUST have same endianness.
      rsp->add_adjust_outputs(a.data(), a.size());
      // how to move this buffer to pb ?
      a.reset();
    }
  }

  void AdjustMul(::google::protobuf::RpcController* controller,
                 const AdjustMulRequest* req, AdjustResponse* rsp,
                 ::google::protobuf::Closure* done) override {
    Adjust(controller, req, rsp, done);
  }

  void AdjustSquare(::google::protobuf::RpcController* controller,
                    const AdjustSquareRequest* req, AdjustResponse* rsp,
                    ::google::protobuf::Closure* done) override {
    Adjust(controller, req, rsp, done);
  }

  void AdjustDot(::google::protobuf::RpcController* controller,
                 const AdjustDotRequest* req, AdjustResponse* rsp,
                 ::google::protobuf::Closure* done) override {
    Adjust(controller, req, rsp, done);
  }

  void AdjustAnd(::google::protobuf::RpcController* controller,
                 const AdjustAndRequest* req, AdjustResponse* rsp,
                 ::google::protobuf::Closure* done) override {
    Adjust(controller, req, rsp, done);
  }

  void AdjustTrunc(::google::protobuf::RpcController* controller,
                   const AdjustTruncRequest* req, AdjustResponse* rsp,
                   ::google::protobuf::Closure* done) override {
    Adjust(controller, req, rsp, done);
  }

  void AdjustTruncPr(::google::protobuf::RpcController* controller,
                     const AdjustTruncPrRequest* req, AdjustResponse* rsp,
                     ::google::protobuf::Closure* done) override {
    Adjust(controller, req, rsp, done);
  }

  void AdjustRandBit(::google::protobuf::RpcController* controller,
                     const AdjustRandBitRequest* req, AdjustResponse* rsp,
                     ::google::protobuf::Closure* done) override {
    Adjust(controller, req, rsp, done);
  }

  void AdjustEqz(::google::protobuf::RpcController* controller,
                 const AdjustEqzRequest* req, AdjustResponse* rsp,
                 ::google::protobuf::Closure* done) override {
    Adjust(controller, req, rsp, done);
  }

  void AdjustPerm(::google::protobuf::RpcController* controller,
                  const AdjustPermRequest* req, AdjustResponse* rsp,
                  ::google::protobuf::Closure* done) override {
    Adjust(controller, req, rsp, done);
  }
};  // namespace spu::mpc::semi2k::beaver::ttp_server

std::unique_ptr<brpc::Server> RunServer(const ServerOptions& options) {
  brpc::FLAGS_max_body_size = std::numeric_limits<uint64_t>::max();
  brpc::FLAGS_socket_max_unwritten_bytes =
      std::numeric_limits<int64_t>::max() / 2;

  auto server = std::make_unique<brpc::Server>();
  auto svc = std::make_unique<ServiceImpl>(options.asym_crypto_schema,
                                           options.server_private_key);

  if (server->AddService(svc.release(), brpc::SERVER_OWNS_SERVICE) != 0) {
    SPDLOG_ERROR("Fail to add service");
    return nullptr;
  }

  // TODO: add TLS options for client/server two-way authentication
  brpc::ServerOptions brpc_options;
  brpc_options.has_builtin_services = true;
  if (server->Start(options.port, &brpc_options) != 0) {
    SPDLOG_ERROR("Fail to start Server");
    return nullptr;
  }
  return server;
}

int RunUntilAskedToQuit(const ServerOptions& options) {
  auto server = RunServer(options);

  SPU_ENFORCE(server);

  server->RunUntilAskedToQuit();

  return 0;
}

}  // namespace spu::mpc::semi2k::beaver::ttp_server
