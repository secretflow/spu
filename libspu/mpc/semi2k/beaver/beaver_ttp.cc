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

#include "libspu/mpc/semi2k/beaver/beaver_ttp.h"

#include <random>
#include <thread>
#include <utility>

#include "yacl/crypto/utils/rand.h"
#include "yacl/link/link.h"
#include "yacl/utils/serialize.h"

#include "libspu/mpc/common/prg_tensor.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace brpc {

DECLARE_uint64(max_body_size);
DECLARE_int64(socket_max_unwritten_bytes);

}  // namespace brpc

namespace spu::mpc::semi2k {

namespace {

int32_t kRequiredVersion = 1;

template <class AdjustRequest>
AdjustRequest BuildAdjustRequest(std::string_view session,
                                 absl::Span<const PrgArrayDesc> descs) {
  AdjustRequest ret;

  SPU_ENFORCE(!descs.empty());

  ret.set_session_id(session.data(), session.size());
  for (const auto& desc : descs) {
    auto* input = ret.add_prg_inputs();
    input->set_prg_count(desc.prg_counter);
    input->set_size(desc.numel * SizeOf(desc.field));
  }
  ret.set_field(static_cast<int64_t>(descs[0].field));
  return ret;
}

template <class T>
struct dependent_false : std::false_type {};

template <class AdjustRequest>
std::vector<ArrayRef> RpcCall(brpc::Channel& channel, AdjustRequest req,
                              FieldType ret_field) {
  brpc::Controller cntl;
  beaver::ttp_server::BeaverService::Stub stub(&channel);
  beaver::ttp_server::AdjustResponse rsp;

  if constexpr (std::is_same_v<AdjustRequest,
                               beaver::ttp_server::AdjustMulRequest>) {
    stub.AdjustMul(&cntl, &req, &rsp, nullptr);
  } else if constexpr (std::is_same_v<AdjustRequest,
                                      beaver::ttp_server::AdjusDotRequest>) {
    stub.AdjustDot(&cntl, &req, &rsp, nullptr);
  } else if constexpr (std::is_same_v<AdjustRequest,
                                      beaver::ttp_server::AdjustAndRequest>) {
    stub.AdjustAnd(&cntl, &req, &rsp, nullptr);
  } else if constexpr (std::is_same_v<AdjustRequest,
                                      beaver::ttp_server::AdjustTruncRequest>) {
    stub.AdjustTrunc(&cntl, &req, &rsp, nullptr);
  } else if constexpr (std::is_same_v<
                           AdjustRequest,
                           beaver::ttp_server::AdjustTruncPrRequest>) {
    stub.AdjustTruncPr(&cntl, &req, &rsp, nullptr);
  } else if constexpr (std::is_same_v<
                           AdjustRequest,
                           beaver::ttp_server::AdjustRandBitRequest>) {
    stub.AdjustRandBit(&cntl, &req, &rsp, nullptr);
  } else {
    static_assert(dependent_false<AdjustRequest>::value,
                  "not support AdjustRequest type");
  }

  SPU_ENFORCE(!cntl.Failed(), "Adjust RpcCall failed, code={} error={}",
              cntl.ErrorCode(), cntl.ErrorText());
  SPU_ENFORCE(rsp.code() == beaver::ttp_server::ErrorCode::OK,
              "Adjust server failed code={}, error={}", rsp.code(),
              rsp.message());

  std::vector<ArrayRef> ret;
  for (const auto& output : rsp.adjust_outputs()) {
    SPU_ENFORCE(output.size() % SizeOf(ret_field) == 0);
    size_t size = output.size() / SizeOf(ret_field);
    // FIXME: change beaver interface: change return type to buffer.
    ArrayRef array(makeType<RingTy>(ret_field), size);
    // FIXME: TTP adjuster server and client MUST have same endianness.
    std::memcpy(array.data(), output.data(), output.size());
    ret.push_back(std::move(array));
  }

  return ret;
}

}  // namespace

BeaverTtp::~BeaverTtp() {
  if (lctx_->Rank() == 0) {
    // all parties should arrive here, only rank0 destroys the remote session.
    beaver::ttp_server::DeleteSessionRequest req;
    req.set_session_id(options_.session_id);

    brpc::Controller cntl;
    beaver::ttp_server::BeaverService::Stub stub(&channel_);
    beaver::ttp_server::DeleteSessionResponse rsp;
    stub.DeleteSession(&cntl, &req, &rsp, NULL);

    if (cntl.Failed()) {
      // we can do nothing more.
      SPDLOG_ERROR("delete session rpc failed, code={} error={}",
                   cntl.ErrorCode(), cntl.ErrorText());
    }
    if (rsp.code() != beaver::ttp_server::ErrorCode::OK) {
      // we can do nothing more.
      SPDLOG_ERROR("delete session server failed code={}, error={}", rsp.code(),
                   rsp.message());
    }
  }
}

BeaverTtp::BeaverTtp(std::shared_ptr<yacl::link::Context> lctx, Options ops)
    : lctx_(std::move(std::move(lctx))),
      seed_(yacl::crypto::RandSeed(true)),
      counter_(0),
      options_(std::move(ops)),
      child_counter_(0) {
  brpc::FLAGS_max_body_size = std::numeric_limits<uint64_t>::max();
  brpc::FLAGS_socket_max_unwritten_bytes =
      std::numeric_limits<int64_t>::max() / 2;
  // init remote connection.
  SPU_ENFORCE(lctx_);
  SPU_ENFORCE_GT(lctx_->WorldSize(), options_.adjust_rank);
  {
    brpc::ChannelOptions brc_options;
    brc_options.protocol = options_.brpc_channel_protocol;
    brc_options.connection_type = options_.brpc_channel_connection_type;
    brc_options.timeout_ms = options_.brpc_timeout_ms;
    brc_options.max_retry = options_.brpc_max_retry;
    // TODO TLS

    if (channel_.Init(options_.server_host.c_str(),
                      options_.brpc_load_balancer_name.c_str(),
                      &brc_options) != 0) {
      SPU_THROW("Fail to initialize channel for BeaverTtp, server_host {}",
                options_.server_host);
    }
  }

  {
    beaver::ttp_server::CreateSessionRequest req;
    {
      req.set_session_id(options_.session_id);
      req.set_adjust_rank(options_.adjust_rank);
      req.set_world_size(lctx_->WorldSize());
      req.set_rank(lctx_->Rank());
      auto seed_buf = yacl::SerializeUint128(seed_);
      req.set_prg_seed(seed_buf.data(), seed_buf.size());
      req.set_required_version(kRequiredVersion);
    }

    brpc::Controller cntl;
    beaver::ttp_server::BeaverService::Stub stub(&channel_);
    beaver::ttp_server::CreateSessionResponse rsp;
    stub.CreateSession(&cntl, &req, &rsp, nullptr);

    SPU_ENFORCE(!cntl.Failed(), "create session rpc failed, code={} error={}",
                cntl.ErrorCode(), cntl.ErrorText());
    SPU_ENFORCE(rsp.code() == beaver::ttp_server::ErrorCode::OK,
                "create session server failed code={}, error={}", rsp.code(),
                rsp.message());
  }

  yacl::link::Barrier(lctx_, "BeaverTtp Init");
}

BeaverTtp::Triple BeaverTtp::Mul(FieldType field, size_t size) {
  std::vector<PrgArrayDesc> descs(3);

  auto a = prgCreateArray(field, size, seed_, &counter_, descs.data());
  auto b = prgCreateArray(field, size, seed_, &counter_, &descs[1]);
  auto c = prgCreateArray(field, size, seed_, &counter_, &descs[2]);

  if (lctx_->Rank() == options_.adjust_rank) {
    auto req = BuildAdjustRequest<beaver::ttp_server::AdjustMulRequest>(
        options_.session_id, descs);
    auto adjusts = RpcCall(channel_, req, field);
    SPU_ENFORCE_EQ(adjusts.size(), 1U);
    ring_add_(c, adjusts[0]);
  }

  return {a, b, c};
}

BeaverTtp::Triple BeaverTtp::Dot(FieldType field, size_t m, size_t n,
                                 size_t k) {
  std::vector<PrgArrayDesc> descs(3);

  auto a = prgCreateArray(field, m * k, seed_, &counter_, descs.data());
  auto b = prgCreateArray(field, k * n, seed_, &counter_, &descs[1]);
  auto c = prgCreateArray(field, m * n, seed_, &counter_, &descs[2]);

  if (lctx_->Rank() == options_.adjust_rank) {
    auto req = BuildAdjustRequest<beaver::ttp_server::AdjusDotRequest>(
        options_.session_id, descs);
    req.set_m(m);
    req.set_n(n);
    req.set_k(k);
    auto adjusts = RpcCall(channel_, req, field);
    SPU_ENFORCE_EQ(adjusts.size(), 1U);
    ring_add_(c, adjusts[0]);
  }

  return {a, b, c};
}

BeaverTtp::Triple BeaverTtp::And(FieldType field, size_t size) {
  std::vector<PrgArrayDesc> descs(3);

  auto a = prgCreateArray(field, size, seed_, &counter_, descs.data());
  auto b = prgCreateArray(field, size, seed_, &counter_, &descs[1]);
  auto c = prgCreateArray(field, size, seed_, &counter_, &descs[2]);

  if (lctx_->Rank() == options_.adjust_rank) {
    auto req = BuildAdjustRequest<beaver::ttp_server::AdjustAndRequest>(
        options_.session_id, descs);
    auto adjusts = RpcCall(channel_, req, field);
    SPU_ENFORCE_EQ(adjusts.size(), 1U);
    ring_xor_(c, adjusts[0]);
  }

  return {a, b, c};
}

BeaverTtp::Pair BeaverTtp::Trunc(FieldType field, size_t size, size_t bits) {
  std::vector<PrgArrayDesc> descs(2);

  auto a = prgCreateArray(field, size, seed_, &counter_, descs.data());
  auto b = prgCreateArray(field, size, seed_, &counter_, &descs[1]);

  if (lctx_->Rank() == options_.adjust_rank) {
    auto req = BuildAdjustRequest<beaver::ttp_server::AdjustTruncRequest>(
        options_.session_id, descs);
    req.set_bits(bits);
    auto adjusts = RpcCall(channel_, req, field);
    SPU_ENFORCE_EQ(adjusts.size(), 1U);
    ring_add_(b, adjusts[0]);
  }

  return {a, b};
}

BeaverTtp::Triple BeaverTtp::TruncPr(FieldType field, size_t size,
                                     size_t bits) {
  std::vector<PrgArrayDesc> descs(3);

  auto r = prgCreateArray(field, size, seed_, &counter_, descs.data());
  auto rc = prgCreateArray(field, size, seed_, &counter_, &descs[1]);
  auto rb = prgCreateArray(field, size, seed_, &counter_, &descs[2]);

  if (lctx_->Rank() == options_.adjust_rank) {
    auto req = BuildAdjustRequest<beaver::ttp_server::AdjustTruncPrRequest>(
        options_.session_id, descs);
    req.set_bits(bits);
    auto adjusts = RpcCall(channel_, req, field);
    SPU_ENFORCE_EQ(adjusts.size(), 2U);
    ring_add_(rc, adjusts[0]);
    ring_add_(rb, adjusts[1]);
  }

  return {r, rc, rb};
}

ArrayRef BeaverTtp::RandBit(FieldType field, size_t size) {
  std::vector<PrgArrayDesc> descs(1);
  auto a = prgCreateArray(field, size, seed_, &counter_, &descs[0]);

  if (lctx_->Rank() == options_.adjust_rank) {
    auto req = BuildAdjustRequest<beaver::ttp_server::AdjustRandBitRequest>(
        options_.session_id, descs);
    auto adjusts = RpcCall(channel_, req, field);
    SPU_ENFORCE_EQ(adjusts.size(), 1U);
    ring_add_(a, adjusts[0]);
  }

  return a;
}

std::unique_ptr<Beaver> BeaverTtp::Spawn() {
  auto new_options = options_;
  new_options.session_id =
      fmt::format("{}_{}", options_.session_id, child_counter_++);
  return std::make_unique<BeaverTtp>(lctx_->Spawn(), std::move(new_options));
}

}  // namespace spu::mpc::semi2k
