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

#include <brpc/progressive_attachment.h>

#include <algorithm>
#include <cerrno>
#include <future>
#include <vector>

#include "absl/strings/ascii.h"
#include "spdlog/spdlog.h"
#include "yacl/base/byte_container_view.h"
#include "yacl/base/exception.h"
#include "yacl/crypto/pke/sm2_enc.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/mpc/common/prg_tensor.h"
#include "libspu/mpc/semi2k/beaver/beaver_impl/trusted_party/trusted_party.h"
#include "libspu/mpc/utils/permute.h"

#include "libspu/mpc/semi2k/beaver/beaver_impl/ttp_server/service.pb.h"

namespace brpc {

DECLARE_uint64(max_body_size);
DECLARE_int64(socket_max_unwritten_bytes);

}  // namespace brpc

namespace spu::mpc::semi2k::beaver::ttp_server {

namespace {

const int64_t kReplayChunkSize = 32L * 1024 * 1024;

inline size_t CeilDiv(size_t a, size_t b) { return (a + b - 1) / b; }

class DecryptError : public yacl::Exception {
  using yacl::Exception::Exception;
};

struct PermMeta {
  uint64_t prg_count;
  PrgSeed seed;
  int64_t size;
};

template <class AdjustRequest>
std::tuple<std::vector<TrustedParty::Operand>, PermMeta,
           std::vector<std::vector<PrgSeed>>, size_t>
BuildOperand(const AdjustRequest& req, uint32_t field_size,
             const std::unique_ptr<yacl::crypto::PkeDecryptor>& decryptor,
             ElementType eltype) {
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
        TrustedParty::Operand{{shape, type, prg_count, eltype}, seeds.back()});
  }

  if constexpr (std::is_same_v<AdjustRequest, AdjustDotRequest>) {
    for (int i = 0; i < 3; i++) {
      ops[i].transpose = req.transpose_inputs()[i];
    }
  }

  PermMeta perm;
  if constexpr (std::is_same_v<AdjustRequest, AdjustPermRequest>) {
    const PrgRandPermMeta& perm_meta = req.perm();
    perm.prg_count = perm_meta.prg_count();
    perm.size = perm_meta.size();
    perm.seed = try_decrypt(perm_meta.encrypted_seeds());
  }

  return {std::move(ops), std::move(perm), std::move(seeds), pad_length};
}

template <class T>
struct dependent_false : std::false_type {};

template <class AdjustRequest>
size_t GetBufferLength(const AdjustRequest& req) {
  if constexpr (std::is_same_v<AdjustRequest,
                               beaver::ttp_server::AdjustPermRequest>) {
    if (req.prg_inputs().size() > 0 && req.field_size() > 0) {
      return req.prg_inputs()[0].buffer_len() / req.field_size() *
             sizeof(int64_t);
    } else {
      SPDLOG_ERROR("Invalid request, prg_inputs size: {}, field_size: {}",
                   req.prg_inputs().size(), req.field_size());
    }
  }
  return 0;
}

void HandleStreamingError(
    butil::intrusive_ptr<brpc::ProgressiveAttachment>& pa) {
  int errsv = errno;
  YACL_THROW_IO_ERROR("streaming Write error, errno {}, strerror {}, client {}",
                      errsv, strerror(errsv),
                      butil::endpoint2str(pa->remote_side()).c_str());
}

void SendStreamData(const std::vector<NdArrayRef>& adjusts,
                    butil::intrusive_ptr<brpc::ProgressiveAttachment>& pa,
                    int64_t pad_length = 0) {
  SPU_ENFORCE(!adjusts.empty());

  // FIXME: TTP adjuster server and client MUST have same endianness.
  for (const auto& adjust : adjusts) {
    const auto& buf = adjust.buf();
    const auto* data = buf->data<uint8_t>();
    const int64_t need_seed = buf->size() - pad_length;

    int64_t pos = 0;
    while (pos < need_seed) {
      const int64_t send_size = std::min(need_seed - pos, kReplayChunkSize);
      std::array<uint8_t, 1 + sizeof(int64_t)> flags;
      flags[0] = 0;
      std::memcpy(&flags[1], &send_size, sizeof(int64_t));
      if (pa->Write(flags.data(), flags.size()) != 0) {
        HandleStreamingError(pa);
      }
      if (pa->Write(data + pos, send_size) != 0) {
        HandleStreamingError(pa);
      }
      pos += send_size;
    }
  }
}

void SendError(butil::intrusive_ptr<brpc::ProgressiveAttachment>& pa,
               ErrorCode code, const std::string& err) {
  std::array<uint8_t, 1 + sizeof(int64_t)> flags;
  int64_t err_size = err.size();
  flags[0] = code;
  // FIXME: TTP adjuster server and client MUST have same endianness.
  std::memcpy(&flags[1], &err_size, sizeof(int64_t));

  try {
    if (pa->Write(flags.data(), flags.size()) != 0) {
      HandleStreamingError(pa);
    }
    if (pa->Write(err.data(), err.size()) != 0) {
      HandleStreamingError(pa);
    }
  } catch (const std::exception& e) {
    // streaming write error, we can do nothing but logging
    SPDLOG_ERROR(
        "error happend during send error to client, error try to send {}, "
        "error happend {}",
        err, e.what());
    return;
  }
}

template <class AdjustRequest>
std::vector<NdArrayRef> AdjustImpl(const AdjustRequest& req,
                                   absl::Span<TrustedParty::Operand> ops,
                                   const PermMeta& perm) {
  std::vector<NdArrayRef> ret;
  if constexpr (std::is_same_v<AdjustRequest, AdjustMulRequest>) {
    auto adjust = TrustedParty::adjustMul(ops);
    ret.push_back(std::move(adjust));
  } else if constexpr (std::is_same_v<AdjustRequest, AdjustMulPrivRequest>) {
    auto adjust = TrustedParty::adjustMulPriv(ops);
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
    uint64_t prg_count = perm.prg_count;
    auto pv = genRandomPerm(perm.size, perm.seed, &prg_count);
    auto adjust = TrustedParty::adjustPerm(ops, pv);
    ret.push_back(std::move(adjust));
  } else {
    static_assert(dependent_false<AdjustRequest>::value,
                  "not support AdjustRequest type");
  }

  return ret;
}

template <class AdjustRequest>
void AdjustAndSend(
    brpc::Controller* cntl, const AdjustRequest* req,
    ::google::protobuf::Closure* done,
    const std::unique_ptr<yacl::crypto::PkeDecryptor>& decryptor) {
  std::string client_side(butil::endpoint2str(cntl->remote_side()).c_str());
  auto pa = cntl->CreateProgressiveAttachment();

  std::tuple<std::vector<TrustedParty::Operand>, PermMeta,
             std::vector<std::vector<PrgSeed>>, size_t>
      adjust_params;

  // AdjustAndSend using streaming send, needs call done before starting
  // calculation, done will free req, but calculation needs to use req
  // so we make a copy here.
  const auto request = *req;
  {
    brpc::ClosureGuard done_guard(done);
    try {
      size_t field_size;
      if constexpr (std::is_same_v<AdjustRequest, AdjustAndRequest>) {
        field_size = 128 / 8;
      } else {
        field_size = request.field_size();
      }
      ElementType eltype = ElementType::kRing;
      // enable eltype for selected requests here
      // later all requests may support gfmp
      if constexpr (std::is_same_v<AdjustRequest, AdjustMulRequest> ||
                    std::is_same_v<AdjustRequest, AdjustMulPrivRequest>) {
        if (request.element_type() == ElType::GFMP) {
          eltype = ElementType::kGfmp;
        }
      }
      adjust_params = BuildOperand(request, field_size, decryptor, eltype);
    } catch (const DecryptError& e) {
      auto err = fmt::format("Seed Decrypt error {}", e.what());
      SPDLOG_ERROR("{}, client {}", err, client_side);
      SendError(pa, ErrorCode::SeedDecryptError, err);
      return;
    } catch (const std::exception& e) {
      auto err = fmt::format("adjust error {}", e.what());
      SPDLOG_ERROR("{}, client {}", err, client_side);
      SendError(pa, ErrorCode::OpAdjustError, err);
      return;
    }
  }

  try {
    auto& [ops, perm, seeds, pad_length] = adjust_params;
    if constexpr (std::is_same_v<AdjustRequest, AdjustDotRequest> ||
                  std::is_same_v<AdjustRequest, AdjustPermRequest>) {
      auto adjusts = AdjustImpl(request, absl::MakeSpan(ops), perm);
      SendStreamData(adjusts, pa);
    } else {
      SPU_ENFORCE_EQ(beaver::ttp_server::kReplayChunkSize % 128, 0U);
      SPU_ENFORCE(!ops.empty());
      for (size_t idx = 1; idx < ops.size(); idx++) {
        SPU_ENFORCE(ops[0].desc.shape == ops[idx].desc.shape);
      }
      int64_t left_elements = ops[0].desc.shape.at(0);
      int64_t chunk_elements =
          beaver::ttp_server::kReplayChunkSize / SizeOf(ops[0].desc.field);
      while (left_elements > 0) {
        int64_t cur_elements = std::min(left_elements, chunk_elements);
        left_elements -= cur_elements;
        for (auto& op : ops) {
          op.desc.shape[0] = cur_elements;
        }
        auto adjusts = AdjustImpl(request, absl::MakeSpan(ops), perm);
        if (left_elements > 0) {
          SendStreamData(adjusts, pa);
        } else {
          SendStreamData(adjusts, pa, pad_length);
        }
      }
    }
  } catch (const yacl::IoError& e) {
    // streaming write error, we can do nothing but logging
    SPDLOG_ERROR(e.what());
    return;
  } catch (const std::exception& e) {
    // some other error happened, try send to client.
    auto err = fmt::format("adjust error {}", e.what());
    SPDLOG_ERROR("{}, client {}", err, client_side);
    SendError(pa, ErrorCode::OpAdjustError, err);
    return;
  }
}

}  // namespace

class ServiceImpl final : public BeaverService {
 private:
  std::unique_ptr<yacl::crypto::PkeDecryptor> decryptor_;

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
    auto* cntl = static_cast<brpc::Controller*>(controller);
    AdjustAndSend(cntl, req, done, decryptor_);
  }

  void AdjustMul(::google::protobuf::RpcController* controller,
                 const AdjustMulRequest* req, AdjustResponse* rsp,
                 ::google::protobuf::Closure* done) override {
    Adjust(controller, req, rsp, done);
  }

  void AdjustMulPriv(::google::protobuf::RpcController* controller,
                     const AdjustMulPrivRequest* req, AdjustResponse* rsp,
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

  // workaround fix
  brpc::ServerOptions brpc_options = server->options();

  if (options.brpc_ssl_options) {
    *brpc_options.mutable_ssl_options() = options.brpc_ssl_options.value();
  }

  brpc_options.has_builtin_services = false;
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
