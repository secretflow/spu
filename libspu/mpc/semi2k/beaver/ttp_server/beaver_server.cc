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

#include "libspu/mpc/semi2k/beaver/ttp_server/beaver_server.h"

#include <shared_mutex>
#include <vector>

#include "absl/types/span.h"
#include "spdlog/spdlog.h"
#include "yacl/utils/serialize.h"

#include "libspu/core/array_ref.h"
#include "libspu/mpc/common/prg_tensor.h"
#include "libspu/mpc/semi2k/beaver/trusted_party.h"

#include "libspu/mpc/semi2k/beaver/ttp_server/service.pb.h"

namespace brpc {

DECLARE_uint64(max_body_size);
DECLARE_int64(socket_max_unwritten_bytes);

}  // namespace brpc

namespace spu::mpc::semi2k::beaver::ttp_server {

namespace {
int32_t kServerSupportedVersion = 1;

template <class AdjustRequest>
std::vector<PrgArrayDesc> BuildDescs(const AdjustRequest& req) {
  std::vector<PrgArrayDesc> ret;
  auto field = req.field();

  SPU_ENFORCE(FieldType_IsValid(field));
  auto type = static_cast<FieldType>(field);

  for (const auto& p : req.prg_inputs()) {
    auto prg_count = static_cast<uint64_t>(p.prg_count());
    auto size = static_cast<size_t>(p.size());
    SPU_ENFORCE(size % SizeOf(type) == 0);
    size_t numel = size / SizeOf(type);
    ret.push_back(PrgArrayDesc{numel, type, prg_count});
  }

  return ret;
}

template <class T>
struct dependent_false : std::false_type {};

template <class AdjustRequest>
std::vector<ArrayRef> AdjustImpl(const AdjustRequest& req,
                                 absl::Span<const PrgSeed> seeds) {
  std::vector<ArrayRef> ret;
  auto descs = BuildDescs(req);
  if constexpr (std::is_same_v<AdjustRequest, AdjustMulRequest>) {
    auto adjust = TrustedParty::adjustMul(descs, seeds);
    ret.push_back(std::move(adjust));
  } else if constexpr (std::is_same_v<AdjustRequest, AdjusDotRequest>) {
    auto adjust =
        TrustedParty::adjustDot(descs, seeds, req.m(), req.n(), req.k());
    ret.push_back(std::move(adjust));
  } else if constexpr (std::is_same_v<AdjustRequest, AdjustAndRequest>) {
    auto adjust = TrustedParty::adjustAnd(descs, seeds);
    ret.push_back(std::move(adjust));
  } else if constexpr (std::is_same_v<AdjustRequest, AdjustTruncRequest>) {
    auto adjust = TrustedParty::adjustTrunc(descs, seeds, req.bits());
    ret.push_back(std::move(adjust));
  } else if constexpr (std::is_same_v<AdjustRequest, AdjustTruncPrRequest>) {
    auto adjust = TrustedParty::adjustTruncPr(descs, seeds, req.bits());
    ret.push_back(std::move(std::get<0>(adjust)));
    ret.push_back(std::move(std::get<1>(adjust)));
  } else if constexpr (std::is_same_v<AdjustRequest, AdjustRandBitRequest>) {
    auto adjust = TrustedParty::adjustRandBit(descs, seeds);
    ret.push_back(std::move(adjust));
  } else {
    static_assert(dependent_false<AdjustRequest>::value,
                  "not support AdjustRequest type");
  }

  return ret;
}

}  // namespace

class ServiceImpl final : public BeaverService {
 private:
  struct Session {
    Session(int32_t world_size, int32_t adjust_rank)
        : world_size(world_size),
          adjust_rank(adjust_rank),
          seeds(world_size, 0),
          rank_ready(world_size, 0),
          session_ready(false) {}

    int32_t world_size;
    int32_t adjust_rank;
    std::vector<PrgSeed> seeds;
    std::vector<int8_t> rank_ready;
    bool session_ready;
  };
  mutable std::shared_mutex mutex_;
  std::map<std::string, std::shared_ptr<Session>> sessions_;

 private:
  std::shared_ptr<Session> GetSession(const std::string& session_id,
                                      AdjustResponse* rsp) const {
    std::shared_lock lock(mutex_);

    const auto& itr = sessions_.find(session_id);
    if (itr == sessions_.end()) {
      rsp->set_code(ErrorCode::SessionError);
      rsp->set_message(fmt::format("session={} not found", session_id));
      return nullptr;
    }

    if (!itr->second->session_ready) {
      rsp->set_code(ErrorCode::SessionError);
      rsp->set_message(fmt::format("session={} not ready", session_id));
      return nullptr;
    }

    return itr->second;
  }

 public:
  ServiceImpl() = default;

  void CreateSession(::google::protobuf::RpcController* controller,
                     const CreateSessionRequest* req,
                     CreateSessionResponse* rsp,
                     ::google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    auto* cntl = static_cast<brpc::Controller*>(controller);
    std::string client_side(butil::endpoint2str(cntl->remote_side()).c_str());

    const auto& session_id = req->session_id();
    auto adjust_rank = req->adjust_rank();
    auto world_size = req->world_size();
    auto rank = req->rank();

    PrgSeed prg_seed;
    try {
      prg_seed = yacl::DeserializeUint128(req->prg_seed());
    } catch (const std::exception& e) {
      SPDLOG_ERROR(
          "Deserialize PrgSeed err={}, session_id {}, rank {}, client {}",
          e.what(), session_id, rank, client_side);
      rsp->set_code(ErrorCode::SessionError);
      rsp->set_message(fmt::format("Deserialize PrgSeed error={}", e.what()));
      return;
    }

    if (kServerSupportedVersion < req->required_version()) {
      SPDLOG_ERROR("unsupported version {}, session_id {}, rank {}, client {}",
                   req->required_version(), session_id, rank, client_side);
      rsp->set_code(ErrorCode::SessionError);
      rsp->set_message(
          fmt::format("unsupported version {}", req->required_version()));
      return;
    }

    {
      std::unique_lock lock(mutex_);

      std::shared_ptr<Session> ss;
      if (sessions_.find(session_id) == sessions_.end()) {
        // first arrival party
        ss = std::make_shared<Session>(world_size, adjust_rank);
        sessions_[session_id] = ss;
        SPDLOG_INFO("new session: {}, rank {}, client {}", session_id, rank,
                    client_side);
      } else {
        // other parties add to this session.
        ss = sessions_[session_id];
        SPDLOG_INFO("add to session: {}, rank={}, client {}", session_id, rank,
                    client_side);
      }

      if (ss->adjust_rank != adjust_rank) {
        auto err = fmt::format(
            "adjust_rank mismatch, session_id {}, rank={}, client={}, request "
            "adjust_rank={}, session adjust_rank={}",
            session_id, rank, client_side, ss->adjust_rank);
        SPDLOG_ERROR(err);
        rsp->set_code(ErrorCode::SessionError);
        rsp->set_message(err);
        return;
      }

      if (ss->world_size != world_size) {
        auto err = fmt::format(
            "world_size mismatch, session_id {}, rank={}, client={}, request "
            "world_size={}, session world_size={}",
            session_id, rank, client_side, world_size, ss->world_size);
        SPDLOG_ERROR(err);
        rsp->set_code(ErrorCode::SessionError);
        rsp->set_message(err);
        return;
      }

      ss->seeds[rank] = prg_seed;
      ss->rank_ready[rank] = 1;
      if (std::all_of(ss->rank_ready.cbegin(), ss->rank_ready.cend(),
                      [](int8_t r) { return r == 1; })) {
        ss->session_ready = true;
      }
    }
    rsp->set_code(ErrorCode::OK);
  }

  void DeleteSession(::google::protobuf::RpcController* controller,
                     const DeleteSessionRequest* req,
                     DeleteSessionResponse* rsp,
                     ::google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    auto* cntl = static_cast<brpc::Controller*>(controller);
    std::string client_side(butil::endpoint2str(cntl->remote_side()).c_str());

    SPDLOG_INFO("del session: {}, client {}", req->session_id(), client_side);

    std::unique_lock lock(mutex_);
    sessions_.erase(req->session_id());
    rsp->set_code(ErrorCode::OK);
  }

  template <class AdjustRequest>
  void Adjust(::google::protobuf::RpcController* controller,
              const AdjustRequest* req, AdjustResponse* rsp,
              ::google::protobuf::Closure* done) const {
    brpc::ClosureGuard done_guard(done);
    auto* cntl = static_cast<brpc::Controller*>(controller);
    std::string client_side(butil::endpoint2str(cntl->remote_side()).c_str());

    const auto& session_id = req->session_id();
    auto ss = GetSession(session_id, rsp);
    if (!ss) {
      SPDLOG_ERROR("GetSession err {}, client {}", rsp->message(), client_side);
      return;
    }

    // TODO: add rank & mac check, make sure this rpc is called by
    // ss->adjust_rank

    std::vector<ArrayRef> adjusts;
    try {
      adjusts = AdjustImpl(*req, ss->seeds);
    } catch (const std::exception& e) {
      auto err = fmt::format("adjust err {}", e.what());
      SPDLOG_ERROR("{}, session {}, client {}", err, session_id, client_side);
      rsp->set_code(ErrorCode::OpAdjustError);
      rsp->set_message(err);
      return;
    }

    rsp->set_code(ErrorCode::OK);
    for (auto& a : adjusts) {
      // FIXME: TTP adjuster server and client MUST have same endianness.
      rsp->add_adjust_outputs(a.data(), a.numel() * a.elsize());
      // how to move this buffer to pb ?
      a.buf()->reset();
    }
  }

  void AdjustMul(::google::protobuf::RpcController* controller,
                 const AdjustMulRequest* req, AdjustResponse* rsp,
                 ::google::protobuf::Closure* done) override {
    Adjust(controller, req, rsp, done);
  }

  void AdjustDot(::google::protobuf::RpcController* controller,
                 const AdjusDotRequest* req, AdjustResponse* rsp,
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
};

std::unique_ptr<brpc::Server> RunServer(int32_t port) {
  brpc::FLAGS_max_body_size = std::numeric_limits<uint64_t>::max();
  brpc::FLAGS_socket_max_unwritten_bytes =
      std::numeric_limits<int64_t>::max() / 2;

  auto server = std::make_unique<brpc::Server>();
  auto svc = std::make_unique<ServiceImpl>();

  if (server->AddService(svc.release(), brpc::SERVER_OWNS_SERVICE) != 0) {
    SPDLOG_ERROR("Fail to add service");
    return nullptr;
  }

  // TODO: add TLS options for client/server two-way authentication
  brpc::ServerOptions options;
  options.has_builtin_services = true;
  if (server->Start(port, &options) != 0) {
    SPDLOG_ERROR("Fail to start Server");
    return nullptr;
  }
  return server;
}

int RunUntilAskedToQuit(int32_t port) {
  auto server = RunServer(port);

  SPU_ENFORCE(server);

  server->RunUntilAskedToQuit();

  return 0;
}

}  // namespace spu::mpc::semi2k::beaver::ttp_server
