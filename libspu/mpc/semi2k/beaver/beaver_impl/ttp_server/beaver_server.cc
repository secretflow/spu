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
#include "libspu/mpc/semi2k/beaver/beaver_impl/ttp_server/beaver_stream.h"

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

class StreamReader : public brpc::StreamInputHandler {
 public:
  enum class Status : int8_t {
    kNotFinished,
    kNormalFinished,
    kAbnormalFinished,
    kStreamFailed,
  };

  explicit StreamReader(size_t total_buf_len) {
    total_buf_len_ = total_buf_len;
    future_finished_ = promise_finished_.get_future();
    future_closed_ = promise_closed_.get_future();
  }

  int on_received_messages(brpc::StreamId id, butil::IOBuf* const messages[],
                           size_t size) override {
    SPDLOG_DEBUG("on_received_messages, stream id: {}", id);
    for (size_t i = 0; i < size; ++i) {
      if (status_ != Status::kNotFinished) {
        SPDLOG_WARN("unexpected messages received");
        return -1;
      }
      const auto& message = messages[i];
      SPDLOG_DEBUG("receive buf size: {}", message->size());
      buf_.append(message->movable());
      if (buf_.length() == total_buf_len_) {
        status_ = Status::kNormalFinished;
        promise_finished_.set_value(status_);
      } else if (buf_.length() > total_buf_len_) {
        SPDLOG_ERROR("buf length ({}) greater than expected buf size ({})",
                     buf_.length(), total_buf_len_);
        status_ = Status::kAbnormalFinished;
        promise_finished_.set_value(status_);
      }
    }
    return 0;
  }

  void on_idle_timeout(brpc::StreamId id) override {
    SPDLOG_INFO("Stream {} idle timeout", id);
  }

  void on_closed(brpc::StreamId id) override {
    SPDLOG_DEBUG("Stream {} closed", id);
    promise_closed_.set_value();
  }

  void on_failed(brpc::StreamId id, int error_code,
                 const std::string& error_text) override {
    SPDLOG_ERROR("Stream {} failed, error_code: {}, error_text: {}", id,
                 error_code, error_text);
    status_ = Status::kStreamFailed;
    promise_finished_.set_value(status_);
  }

  const auto& GetBufRef() const {
    SPU_ENFORCE(status_ == Status::kNormalFinished);
    return buf_;
  }

  Status WaitFinished() { return future_finished_.get(); };

  void WaitClosed() { future_closed_.wait(); }

 private:
  butil::IOBuf buf_;
  size_t total_buf_len_;
  Status status_ = Status::kNotFinished;
  std::promise<Status> promise_finished_;
  std::promise<void> promise_closed_;
  std::future<Status> future_finished_;
  std::future<void> future_closed_;
};

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

void SendStreamData(brpc::StreamId stream_id,
                    absl::Span<const yacl::Buffer> buf_vec) {
  SPU_ENFORCE(!buf_vec.empty());
  for (size_t idx = 1; idx < buf_vec.size(); ++idx) {
    SPU_ENFORCE_EQ(buf_vec[0].size(), buf_vec[idx].size());
  }

  size_t chunk_size = kDownStreamChunkSize / buf_vec.size();
  // FIXME: TTP adjuster server and client MUST have same endianness.
  size_t left_buf_size = buf_vec[0].size();
  int64_t chunk_idx = 0;
  while (left_buf_size > 0) {
    butil::IOBuf io_buf;
    BeaverDownStreamMeta meta;
    io_buf.append(&meta, sizeof(meta));

    size_t cur_chunk_size = std::min(left_buf_size, chunk_size);
    for (const auto& buf : buf_vec) {
      int ret = io_buf.append(buf.data<char>() + (chunk_idx * chunk_size),
                              cur_chunk_size);
      SPU_ENFORCE_EQ(ret, 0, "Append data to IO buffer failed");
    }

    // StreamWrite result cannot be EAGAIN, given that we have not set
    // max_buf_size
    SPU_ENFORCE_EQ(brpc::StreamWrite(stream_id, io_buf), 0);

    left_buf_size -= cur_chunk_size;
    ++chunk_idx;
  }
}

template <class AdjustRequest>
std::vector<NdArrayRef> AdjustImpl(const AdjustRequest& req,
                                   absl::Span<TrustedParty::Operand> ops,
                                   StreamReader& stream_reader) {
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
    auto status = stream_reader.WaitFinished();
    SPU_ENFORCE(status == StreamReader::Status::kNormalFinished,
                "Stream reader finished abnormally, status: {}",
                static_cast<int32_t>(status));
    const auto& buf = stream_reader.GetBufRef();
    SPU_ENFORCE(buf.length() % sizeof(int64_t) == 0);
    std::vector<int64_t> pv(buf.length() / sizeof(int64_t));
    buf.copy_to(pv.data());
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
    const AdjustRequest& req, brpc::StreamId stream_id,
    StreamReader& stream_reader,
    const std::unique_ptr<yacl::crypto::PkeDecryptor>& decryptor) {
  size_t field_size;
  if constexpr (std::is_same_v<AdjustRequest, AdjustAndRequest>) {
    field_size = 128 / 8;
  } else {
    field_size = req.field_size();
  }
  ElementType eltype = ElementType::kRing;
  // enable eltype for selected requests here
  // later all requests may support gfmp
  if constexpr (std::is_same_v<AdjustRequest, AdjustMulRequest> ||
                std::is_same_v<AdjustRequest, AdjustMulPrivRequest>) {
    if (req.element_type() == ElType::GFMP) {
      eltype = ElementType::kGfmp;
    }
  }
  auto [ops, seeds, pad_length] =
      BuildOperand(req, field_size, decryptor, eltype);

  if constexpr (std::is_same_v<AdjustRequest, AdjustDotRequest> ||
                std::is_same_v<AdjustRequest, AdjustPermRequest>) {
    auto adjusts = AdjustImpl(req, absl::MakeSpan(ops), stream_reader);
    auto buf_vec = StripNdArray(adjusts, pad_length);
    SendStreamData(stream_id, buf_vec);
    return;
  }

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
    auto adjusts = AdjustImpl(req, absl::MakeSpan(ops), stream_reader);
    if (left_elements > 0) {
      auto buf_vec = StripNdArray(adjusts, 0);
      SendStreamData(stream_id, buf_vec);
    } else {
      auto buf_vec = StripNdArray(adjusts, pad_length);
      SendStreamData(stream_id, buf_vec);
    }
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
    std::string client_side(butil::endpoint2str(cntl->remote_side()).c_str());
    brpc::StreamId stream_id = brpc::INVALID_STREAM_ID;
    auto request = *req;
    StreamReader reader(GetBufferLength(*req));

    // To address the scenario where clients transmit data after an RPC
    // response, give precedence to setting up absl::MakeCleanup before invoking
    // brpc::ClosureGuard to ensure proper resource management
    auto cleanup = absl::MakeCleanup([&]() {
      auto cleanup = absl::MakeCleanup([&]() {
        if (stream_id != brpc::INVALID_STREAM_ID) {
          // To avoid encountering a core dump, it is essential to close the
          // process stream prior to the destruction of the StreamReader object
          reader.WaitClosed();
        }
      });
      try {
        AdjustAndSend(request, stream_id, reader, decryptor_);
      } catch (const DecryptError& e) {
        auto err = fmt::format("Seed Decrypt error {}", e.what());
        SPDLOG_ERROR("{}, client {}", err,
                     client_side);  // TODO: catch the function name
        BeaverDownStreamMeta meta;
        meta.err_code = ErrorCode::SeedDecryptError;
        butil::IOBuf buf;
        SPU_ENFORCE_EQ(buf.append(&meta, sizeof(meta)), 0);
        SPU_ENFORCE_EQ(buf.append(err.c_str()), 0);
        brpc::StreamWrite(stream_id, buf);
        return;
      } catch (const std::exception& e) {
        auto err = fmt::format("adjust error {}", e.what());
        SPDLOG_ERROR("{}, client {}", err, client_side);
        BeaverDownStreamMeta meta;
        meta.err_code = ErrorCode::OpAdjustError;
        butil::IOBuf buf;
        SPU_ENFORCE_EQ(buf.append(&meta, sizeof(meta)), 0);
        SPU_ENFORCE_EQ(buf.append(err.c_str()), 0);
        brpc::StreamWrite(stream_id, buf);
        return;
      }
    });

    brpc::ClosureGuard done_guard(done);
    brpc::StreamOptions stream_options;
    stream_options.max_buf_size = 0;  // there is no flow control for downstream
    stream_options.handler = &reader;
    if (brpc::StreamAccept(&stream_id, *cntl, &stream_options) != 0) {
      SPDLOG_ERROR("Failed to accept stream");
      rsp->set_code(ErrorCode::StreamAcceptError);
      return;
    }
    rsp->set_code(ErrorCode::OK);
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
