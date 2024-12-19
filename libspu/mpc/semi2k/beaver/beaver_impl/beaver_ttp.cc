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

#include "libspu/mpc/semi2k/beaver/beaver_impl/beaver_ttp.h"

#include <future>
#include <utility>
#include <vector>

#include "yacl/crypto/pke/sm2_enc.h"
#include "yacl/crypto/rand/rand.h"
#include "yacl/link/algorithm/allgather.h"

#include "libspu/mpc/common/prg_tensor.h"
#include "libspu/mpc/semi2k/beaver/beaver_impl/ttp_server/beaver_stream.h"
#include "libspu/mpc/utils/gfmp_ops.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace brpc {

DECLARE_uint64(max_body_size);
DECLARE_int64(socket_max_unwritten_bytes);

}  // namespace brpc

namespace spu::mpc::semi2k {

namespace {

inline size_t CeilDiv(size_t a, size_t b) { return (a + b - 1) / b; }

void FillReplayDesc(Beaver::ReplayDesc* desc, FieldType field, int64_t size,
                    const std::vector<Beaver::PrgSeedBuff>& encrypted_seeds,
                    PrgCounter counter, PrgSeed self_seed,
                    ElementType eltype = ElementType::kRing) {
  if (desc == nullptr || desc->status != Beaver::Init) {
    return;
  }
  desc->size = size;
  desc->field = field;
  desc->prg_counter = counter;
  desc->encrypted_seeds = encrypted_seeds;
  desc->seed = self_seed;
  desc->eltype = eltype;
}

template <class AdjustRequest>
AdjustRequest BuildAdjustRequest(
    absl::Span<const PrgArrayDesc> descs,
    absl::Span<const absl::Span<const yacl::Buffer>> descs_seed) {
  AdjustRequest ret;

  SPU_ENFORCE(!descs.empty());

  uint32_t field_size;
  ElementType eltype = ElementType::kRing;

  for (size_t i = 0; i < descs.size(); i++) {
    const auto& desc = descs[i];
    auto* input = ret.add_prg_inputs();
    input->set_prg_count(desc.prg_counter);
    field_size = SizeOf(desc.field);
    eltype = desc.eltype;

    input->set_buffer_len(desc.shape.numel() * SizeOf(desc.field));

    absl::Span<const yacl::Buffer> seeds;
    if (descs_seed.size() == descs.size()) {
      seeds = descs_seed[i];
    } else {
      SPU_ENFORCE(descs_seed.size() == 1);
      seeds = descs_seed[0];
    }
    for (const auto& s : seeds) {
      input->add_encrypted_seeds(s.data(), s.size());
    }
  }
  if constexpr (!std::is_same_v<AdjustRequest,
                                beaver::ttp_server::AdjustAndRequest>) {
    ret.set_field_size(field_size);
  }
  if constexpr (std::is_same_v<AdjustRequest,
                               beaver::ttp_server::AdjustMulRequest> ||
                std::is_same_v<AdjustRequest,
                               beaver::ttp_server::AdjustMulPrivRequest>) {
    if (eltype == ElementType::kGfmp)
      ret.set_element_type(beaver::ttp_server::ElType::GFMP);
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

  StreamReader(int32_t num_buf, size_t buf_len) {
    SPU_ENFORCE(num_buf > 0);
    SPU_ENFORCE(buf_len > 0);
    buf_vec_.resize(num_buf);
    buf_len_ = buf_len;
    future_finished_ = promise_finished_.get_future();
    future_closed_ = promise_closed_.get_future();
  }

  int on_received_messages(brpc::StreamId id, butil::IOBuf* const messages[],
                           size_t size) override {
    SPDLOG_DEBUG("on_received_messages, stream id: {}", id);
    for (size_t i = 0; i < size; ++i) {
      if (status_ != Status::kNotFinished) {
        SPDLOG_ERROR("unexpected messages received");
        return -1;
      }

      SPDLOG_DEBUG("receive buf size: {}", messages[i]->size());
      const auto& message = messages[i];
      beaver::ttp_server::BeaverDownStreamMeta meta;
      message->copy_to(&meta, sizeof(meta));
      message->pop_front(sizeof(meta));
      if (meta.err_code != 0) {
        SPDLOG_ERROR("response error from server, err_code: {}, err_text: {}",
                     meta.err_code, message->to_string());
        status_ = Status::kAbnormalFinished;
        promise_finished_.set_value(status_);
        return -2;
      }

      SPU_ENFORCE(message->length() % buf_vec_.size() == 0);
      size_t msg_len = message->length() / buf_vec_.size();
      for (size_t buf_idx = 0; buf_idx < buf_vec_.size(); ++buf_idx) {
        message->append_to(&buf_vec_[buf_idx], msg_len, buf_idx * msg_len);
      }

      SPU_ENFORCE(buf_vec_[0].length() <= buf_len_,
                  "unexpected bytes received");
      if (buf_vec_[0].length() == buf_len_) {
        status_ = Status::kNormalFinished;
        promise_finished_.set_value(status_);
      }
    }
    return 0;
  }

  void on_idle_timeout(brpc::StreamId id) override {
    SPDLOG_WARN("Stream {} idle timeout", id);
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

  const auto& GetBufVecRef() const {
    SPU_ENFORCE(status_ == Status::kNormalFinished);
    return buf_vec_;
  }

  Status WaitFinished() { return future_finished_.get(); };

  void WaitClosed() { future_closed_.wait(); }

 private:
  std::vector<butil::IOBuf> buf_vec_;
  size_t buf_len_;
  Status status_ = Status::kNotFinished;
  std::promise<Status> promise_finished_;
  std::promise<void> promise_closed_;
  std::future<Status> future_finished_;
  std::future<void> future_closed_;
};

// Obtain a tuple containing num_buf and buf_len
template <class AdjustRequest>
std::tuple<int32_t, int64_t> GetBufferLength(const AdjustRequest& req) {
  if constexpr (std::is_same_v<AdjustRequest,
                               beaver::ttp_server::AdjustDotRequest>) {
    SPU_ENFORCE_EQ(req.prg_inputs().size(), 3);
    return {1, req.prg_inputs()[2].buffer_len()};
  } else if constexpr (std::is_same_v<
                           AdjustRequest,
                           beaver::ttp_server::AdjustTruncPrRequest>) {
    SPU_ENFORCE_GE(req.prg_inputs().size(), 1);
    return {2, req.prg_inputs()[0].buffer_len()};
  } else {
    SPU_ENFORCE_GE(req.prg_inputs().size(), 1);
    return {1, req.prg_inputs()[0].buffer_len()};
  }
}

template <class AdjustRequest>
std::vector<NdArrayRef> RpcCall(
    brpc::Channel& channel, AdjustRequest req, FieldType ret_field,
    const std::vector<butil::IOBuf>* upstream_messages = nullptr) {
  brpc::Controller cntl;
  beaver::ttp_server::BeaverService::Stub stub(&channel);
  beaver::ttp_server::AdjustResponse rsp;

  auto [num_buf, buf_len] = GetBufferLength(req);
  StreamReader reader(num_buf, buf_len);
  brpc::StreamOptions stream_options;
  stream_options.max_buf_size = 2 * beaver::ttp_server::kUpStreamChunkSize;
  stream_options.handler = &reader;
  brpc::StreamId stream_id;
  SPU_ENFORCE_EQ(brpc::StreamCreate(&stream_id, cntl, &stream_options), 0,
                 "Failed to create stream");
  auto cleanup = absl::MakeCleanup([&stream_id, &reader]() {
    SPU_ENFORCE(brpc::StreamClose(stream_id) == 0);
    reader.WaitClosed();
  });

  if constexpr (std::is_same_v<AdjustRequest,
                               beaver::ttp_server::AdjustMulRequest>) {
    stub.AdjustMul(&cntl, &req, &rsp, nullptr);
  } else if constexpr (std::is_same_v<
                           AdjustRequest,
                           beaver::ttp_server::AdjustMulPrivRequest>) {
    stub.AdjustMulPriv(&cntl, &req, &rsp, nullptr);
  } else if constexpr (std::is_same_v<
                           AdjustRequest,
                           beaver::ttp_server::AdjustSquareRequest>) {
    stub.AdjustSquare(&cntl, &req, &rsp, nullptr);
  } else if constexpr (std::is_same_v<AdjustRequest,
                                      beaver::ttp_server::AdjustDotRequest>) {
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
  } else if constexpr (std::is_same_v<AdjustRequest,
                                      beaver::ttp_server::AdjustEqzRequest>) {
    stub.AdjustEqz(&cntl, &req, &rsp, nullptr);
  } else if constexpr (std::is_same_v<AdjustRequest,
                                      beaver::ttp_server::AdjustPermRequest>) {
    stub.AdjustPerm(&cntl, &req, &rsp, nullptr);
  } else {
    static_assert(dependent_false<AdjustRequest>::value,
                  "not support AdjustRequest type");
  }

  SPU_ENFORCE(!cntl.Failed(), "Adjust RpcCall failed, code={} error={}",
              cntl.ErrorCode(), cntl.ErrorText());
  SPU_ENFORCE(rsp.code() == beaver::ttp_server::ErrorCode::OK,
              "Adjust server failed code={}, error={}",
              ErrorCode_Name(rsp.code()), rsp.message());

  if (upstream_messages != nullptr) {
    for (const auto& message : *upstream_messages) {
      int ret = brpc::StreamWrite(stream_id, message);
      if (ret == EAGAIN) {
        SPU_ENFORCE_EQ(brpc::StreamWait(stream_id, nullptr), 0);
        ret = brpc::StreamWrite(stream_id, message);
      }
      SPU_ENFORCE_EQ(ret, 0, "Write stream failed");
      SPDLOG_DEBUG("write buf size {} to stream id {}", message.length(),
                   stream_id);
    }
  }

  auto status = reader.WaitFinished();
  SPU_ENFORCE(status == StreamReader::Status::kNormalFinished,
              "Stream reader finished abnormally, status: {}",
              static_cast<int32_t>(status));
  std::vector<NdArrayRef> ret;
  for (const auto& buf : reader.GetBufVecRef()) {
    SPU_ENFORCE(buf.length() % SizeOf(ret_field) == 0);
    int64_t size = buf.length() / SizeOf(ret_field);
    // FIXME: change beaver interface: change return type to buffer.
    NdArrayRef array(makeType<RingTy>(ret_field), {size});
    // FIXME: TTP adjuster server and client MUST have same endianness.
    buf.copy_to(array.data());
    ret.push_back(std::move(array));
  }

  return ret;
}

}  // namespace

BeaverTtp::BeaverTtp(std::shared_ptr<yacl::link::Context> lctx, Options ops)
    : lctx_(std::move(lctx)),
      seed_(yacl::crypto::SecureRandSeed()),
      counter_(0),
      options_(std::move(ops)) {
  brpc::FLAGS_max_body_size = std::numeric_limits<uint64_t>::max();
  brpc::FLAGS_socket_max_unwritten_bytes =
      std::numeric_limits<int64_t>::max() / 2;
  // init remote connection.
  SPU_ENFORCE(lctx_);
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

  yacl::Buffer encrypted_seed;
  {
    std::unique_ptr<yacl::crypto::PkeEncryptor> encryptor;
    auto lower_schema = absl::AsciiStrToLower(options_.asym_crypto_schema);
    if (lower_schema == "sm2") {
      encryptor = std::make_unique<yacl::crypto::Sm2Encryptor>(
          options_.server_public_key);
    } else {
      SPU_THROW("not support asym_crypto_schema {}",
                options_.asym_crypto_schema);
    }
    auto encrypted = encryptor->Encrypt(
        {reinterpret_cast<const void*>(&seed_), sizeof(PrgSeed)});
    encrypted_seed = yacl::Buffer(encrypted);
  }

  encrypted_seeds_ = yacl::link::AllGather(lctx_, encrypted_seed,
                                           "BEAVER_TTP:SYNC_ENCRYPTED_SEEDS");
}

// TODO: kGfmp supports more operations
BeaverTtp::Triple BeaverTtp::Mul(FieldType field, int64_t size,
                                 ReplayDesc* x_desc, ReplayDesc* y_desc,
                                 ElementType eltype) {
  std::vector<PrgArrayDesc> descs(3);
  std::vector<absl::Span<const PrgSeedBuff>> descs_seed(3, encrypted_seeds_);
  Shape shape({size, 1});

  auto if_replay = [&](const ReplayDesc* replay_desc, size_t idx) {
    if (replay_desc == nullptr || replay_desc->status != Beaver::Replay) {
      return prgCreateArray(field, shape, seed_, &counter_, &descs[idx],
                            eltype);
    } else {
      SPU_ENFORCE(replay_desc->field == field);
      SPU_ENFORCE(replay_desc->size == size);
      SPU_ENFORCE(replay_desc->encrypted_seeds.size() == lctx_->WorldSize());
      if (lctx_->Rank() == options_.adjust_rank) {
        descs_seed[idx] = replay_desc->encrypted_seeds;
        descs[idx].field = field;
        descs[idx].eltype = eltype;
        descs[idx].shape = shape;
        descs[idx].prg_counter = replay_desc->prg_counter;
      }
      PrgCounter tmp_counter = replay_desc->prg_counter;
      return prgCreateArray(field, shape, replay_desc->seed, &tmp_counter,
                            &descs[idx], eltype);
    }
  };

  FillReplayDesc(x_desc, field, size, encrypted_seeds_, counter_, seed_,
                 eltype);
  auto a = if_replay(x_desc, 0);
  FillReplayDesc(y_desc, field, size, encrypted_seeds_, counter_, seed_,
                 eltype);
  auto b = if_replay(y_desc, 1);
  auto c = prgCreateArray(field, shape, seed_, &counter_, &descs[2], eltype);

  if (lctx_->Rank() == options_.adjust_rank) {
    auto req = BuildAdjustRequest<beaver::ttp_server::AdjustMulRequest>(
        descs, descs_seed);
    auto adjusts = RpcCall(channel_, req, field);
    SPU_ENFORCE_EQ(adjusts.size(), 1U);
    if (eltype == ElementType::kGfmp) {
      auto T = c.eltype();
      gfmp_add_mod_(c, adjusts[0].reshape(shape).as(T));
    } else {
      ring_add_(c, adjusts[0].reshape(shape));
    }
  }

  Triple ret;
  std::get<0>(ret) = std::move(*a.buf());
  std::get<1>(ret) = std::move(*b.buf());
  std::get<2>(ret) = std::move(*c.buf());

  return ret;
}

BeaverTtp::Pair BeaverTtp::MulPriv(FieldType field, int64_t size,
                                   ElementType eltype) {
  std::vector<PrgArrayDesc> descs(2);
  std::vector<absl::Span<const PrgSeedBuff>> descs_seed(2, encrypted_seeds_);
  Shape shape({size, 1});
  auto a_or_b =
      prgCreateArray(field, shape, seed_, &counter_, &descs[0], eltype);
  auto c = prgCreateArray(field, shape, seed_, &counter_, &descs[1], eltype);
  if (lctx_->Rank() == options_.adjust_rank) {
    auto req = BuildAdjustRequest<beaver::ttp_server::AdjustMulPrivRequest>(
        descs, descs_seed);
    auto adjusts = RpcCall(channel_, req, field);
    SPU_ENFORCE_EQ(adjusts.size(), 1U);
    if (eltype == ElementType::kGfmp) {
      auto T = c.eltype();
      gfmp_add_mod_(c, adjusts[0].reshape(shape).as(T));
    } else {
      ring_add_(c, adjusts[0].reshape(shape));
    }
  }

  Pair ret;
  std::get<0>(ret) = std::move(*a_or_b.buf());
  std::get<1>(ret) = std::move(*c.buf());

  return ret;
}

BeaverTtp::Pair BeaverTtp::Square(FieldType field, int64_t size,
                                  ReplayDesc* x_desc) {
  std::vector<PrgArrayDesc> descs(2);
  std::vector<absl::Span<const PrgSeedBuff>> descs_seed(2, encrypted_seeds_);
  Shape shape({size, 1});

  auto if_replay = [&](const ReplayDesc* replay_desc, size_t idx) {
    if (replay_desc == nullptr || replay_desc->status != Beaver::Replay) {
      return prgCreateArray(field, shape, seed_, &counter_, &descs[idx]);
    } else {
      SPU_ENFORCE(replay_desc->field == field);
      SPU_ENFORCE(replay_desc->size == size);
      SPU_ENFORCE(replay_desc->encrypted_seeds.size() == lctx_->WorldSize());
      if (lctx_->Rank() == options_.adjust_rank) {
        descs_seed[idx] = replay_desc->encrypted_seeds;
        descs[idx].field = field;
        descs[idx].shape = shape;
        descs[idx].prg_counter = replay_desc->prg_counter;
      }
      PrgCounter tmp_counter = replay_desc->prg_counter;
      return prgCreateArray(field, shape, replay_desc->seed, &tmp_counter,
                            &descs[idx]);
    }
  };

  FillReplayDesc(x_desc, field, size, encrypted_seeds_, counter_, seed_);
  auto a = if_replay(x_desc, 0);
  auto b = prgCreateArray(field, shape, seed_, &counter_, &descs[1]);

  if (lctx_->Rank() == options_.adjust_rank) {
    auto req = BuildAdjustRequest<beaver::ttp_server::AdjustSquareRequest>(
        descs, descs_seed);
    auto adjusts = RpcCall(channel_, req, field);
    SPU_ENFORCE_EQ(adjusts.size(), 1U);
    ring_add_(b, adjusts[0].reshape(shape));
  }

  Pair ret;
  std::get<0>(ret) = std::move(*a.buf());
  std::get<1>(ret) = std::move(*b.buf());

  return ret;
}

BeaverTtp::Triple BeaverTtp::Dot(FieldType field, int64_t m, int64_t n,
                                 int64_t k, ReplayDesc* x_desc,
                                 ReplayDesc* y_desc) {
  std::vector<PrgArrayDesc> descs(3);
  std::vector<absl::Span<const PrgSeedBuff>> descs_seed(3, encrypted_seeds_);
  std::vector<Shape> shapes(3);
  std::vector<bool> transpose_inputs(3);
  shapes[0] = {m, k};
  shapes[1] = {k, n};
  shapes[2] = {m, n};

  auto if_replay = [&](const ReplayDesc* replay_desc, size_t idx) {
    if (replay_desc == nullptr) {
      return prgCreateArray(field, shapes[idx], seed_, &counter_, &descs[idx]);
    } else {
      SPU_ENFORCE(replay_desc->field == field);
      SPU_ENFORCE(replay_desc->encrypted_seeds.size() == lctx_->WorldSize());
      if (replay_desc->status == Beaver::TransposeReplay) {
        std::reverse(shapes[idx].begin(), shapes[idx].end());
      }
      if (lctx_->Rank() == options_.adjust_rank) {
        descs_seed[idx] = replay_desc->encrypted_seeds;
        descs[idx].field = field;
        descs[idx].shape = shapes[idx];
        descs[idx].prg_counter = replay_desc->prg_counter;
        transpose_inputs[idx] = replay_desc->status == Beaver::TransposeReplay;
      }
      PrgCounter tmp_counter = replay_desc->prg_counter;
      auto ret = prgCreateArray(field, shapes[idx], replay_desc->seed,
                                &tmp_counter, nullptr);
      if (replay_desc->status == Beaver::TransposeReplay) {
        ret = ret.transpose().clone();
      }
      return ret;
    }
  };

  FillReplayDesc(x_desc, field, m * k, encrypted_seeds_, counter_, seed_);
  auto a = if_replay(x_desc, 0);
  FillReplayDesc(y_desc, field, k * n, encrypted_seeds_, counter_, seed_);
  auto b = if_replay(y_desc, 1);
  auto c = prgCreateArray(field, {m, n}, seed_, &counter_, &descs[2]);

  if (lctx_->Rank() == options_.adjust_rank) {
    auto req = BuildAdjustRequest<beaver::ttp_server::AdjustDotRequest>(
        descs, descs_seed);
    req.set_m(m);
    req.set_n(n);
    req.set_k(k);
    for (bool t : transpose_inputs) {
      req.add_transpose_inputs(t);
    }
    auto adjusts = RpcCall(channel_, req, field);
    SPU_ENFORCE_EQ(adjusts.size(), 1U);
    ring_add_(c, adjusts[0].reshape(c.shape()));
  }

  Triple ret;
  std::get<0>(ret) = std::move(*a.buf());
  std::get<1>(ret) = std::move(*b.buf());
  std::get<2>(ret) = std::move(*c.buf());

  return ret;
}

BeaverTtp::Triple BeaverTtp::And(int64_t size) {
  std::vector<PrgArrayDesc> descs(3);
  // inside beaver, use max field for efficiency
  auto field = FieldType::FM128;
  int64_t elsize = CeilDiv(size, SizeOf(field));
  Shape shape({elsize, 1});
  std::vector<absl::Span<const PrgSeedBuff>> descs_seed(1, encrypted_seeds_);

  auto a = prgCreateArray(field, shape, seed_, &counter_, descs.data());
  auto b = prgCreateArray(field, shape, seed_, &counter_, &descs[1]);
  auto c = prgCreateArray(field, shape, seed_, &counter_, &descs[2]);

  if (lctx_->Rank() == options_.adjust_rank) {
    auto req = BuildAdjustRequest<beaver::ttp_server::AdjustAndRequest>(
        descs, descs_seed);
    auto adjusts = RpcCall(channel_, req, field);
    SPU_ENFORCE_EQ(adjusts.size(), 1U);
    ring_xor_(c, adjusts[0].reshape(c.shape()));
  }

  Triple ret;
  std::get<0>(ret) = std::move(*a.buf());
  std::get<1>(ret) = std::move(*b.buf());
  std::get<2>(ret) = std::move(*c.buf());
  std::get<0>(ret).resize(size);
  std::get<1>(ret).resize(size);
  std::get<2>(ret).resize(size);

  return ret;
}

BeaverTtp::Pair BeaverTtp::Trunc(FieldType field, int64_t size, size_t bits) {
  std::vector<PrgArrayDesc> descs(2);
  std::vector<absl::Span<const PrgSeedBuff>> descs_seed(1, encrypted_seeds_);
  Shape shape({size, 1});

  auto a = prgCreateArray(field, shape, seed_, &counter_, descs.data());
  auto b = prgCreateArray(field, shape, seed_, &counter_, &descs[1]);

  if (lctx_->Rank() == options_.adjust_rank) {
    auto req = BuildAdjustRequest<beaver::ttp_server::AdjustTruncRequest>(
        descs, descs_seed);
    req.set_bits(bits);
    auto adjusts = RpcCall(channel_, req, field);
    SPU_ENFORCE_EQ(adjusts.size(), 1U);
    ring_add_(b, adjusts[0].reshape(b.shape()));
  }

  Pair ret;
  ret.first = std::move(*a.buf());
  ret.second = std::move(*b.buf());
  return ret;
}

BeaverTtp::Triple BeaverTtp::TruncPr(FieldType field, int64_t size,
                                     size_t bits) {
  std::vector<PrgArrayDesc> descs(3);
  std::vector<absl::Span<const PrgSeedBuff>> descs_seed(1, encrypted_seeds_);
  Shape shape({size, 1});

  auto r = prgCreateArray(field, shape, seed_, &counter_, descs.data());
  auto rc = prgCreateArray(field, shape, seed_, &counter_, &descs[1]);
  auto rb = prgCreateArray(field, shape, seed_, &counter_, &descs[2]);

  if (lctx_->Rank() == options_.adjust_rank) {
    auto req = BuildAdjustRequest<beaver::ttp_server::AdjustTruncPrRequest>(
        descs, descs_seed);
    req.set_bits(bits);
    auto adjusts = RpcCall(channel_, req, field);
    SPU_ENFORCE_EQ(adjusts.size(), 2U);
    ring_add_(rc, adjusts[0].reshape(rc.shape()));
    ring_add_(rb, adjusts[1].reshape(rb.shape()));
  }

  Triple ret;
  std::get<0>(ret) = std::move(*r.buf());
  std::get<1>(ret) = std::move(*rc.buf());
  std::get<2>(ret) = std::move(*rb.buf());

  return ret;
}

BeaverTtp::Array BeaverTtp::RandBit(FieldType field, int64_t size) {
  std::vector<PrgArrayDesc> descs(1);
  std::vector<absl::Span<const PrgSeedBuff>> descs_seed(1, encrypted_seeds_);
  Shape shape({size, 1});

  auto a = prgCreateArray(field, shape, seed_, &counter_, descs.data());

  if (lctx_->Rank() == options_.adjust_rank) {
    auto req = BuildAdjustRequest<beaver::ttp_server::AdjustRandBitRequest>(
        descs, descs_seed);
    auto adjusts = RpcCall(channel_, req, field);
    SPU_ENFORCE_EQ(adjusts.size(), 1U);
    ring_add_(a, adjusts[0].reshape(a.shape()));
  }

  return std::move(*a.buf());
}

BeaverTtp::Pair BeaverTtp::PermPair(FieldType field, int64_t size,
                                    size_t perm_rank,
                                    absl::Span<const int64_t> perm_vec) {
  std::vector<PrgArrayDesc> descs(2);
  std::vector<absl::Span<const PrgSeedBuff>> descs_seed(1, encrypted_seeds_);
  Shape shape({size, 1});

  auto a = prgCreateArray(field, shape, seed_, &counter_, descs.data());
  auto b = prgCreateArray(field, shape, seed_, &counter_, &descs[1]);

  if (lctx_->Rank() == perm_rank) {
    auto req = BuildAdjustRequest<beaver::ttp_server::AdjustPermRequest>(
        descs, descs_seed);
    std::vector<butil::IOBuf> stream_data;
    size_t left_buf_size = perm_vec.size() * sizeof(int64_t);
    size_t chunk_idx = 0;
    while (left_buf_size > 0) {
      using beaver::ttp_server::kUpStreamChunkSize;
      size_t cur_chunk_size = std::min(left_buf_size, kUpStreamChunkSize);
      stream_data.emplace_back();
      stream_data.back().append(reinterpret_cast<const char*>(perm_vec.data()) +
                                    (chunk_idx * kUpStreamChunkSize),
                                cur_chunk_size);
      ++chunk_idx;
      left_buf_size -= cur_chunk_size;
    }
    auto adjusts = RpcCall(channel_, req, field, &stream_data);
    SPU_ENFORCE_EQ(adjusts.size(), 1U);
    ring_add_(b, adjusts[0].reshape(b.shape()));
  }

  Pair ret;
  ret.first = std::move(*a.buf());
  ret.second = std::move(*b.buf());
  return ret;
}

std::unique_ptr<Beaver> BeaverTtp::Spawn() {
  auto new_options = options_;
  return std::make_unique<BeaverTtp>(lctx_->Spawn(), std::move(new_options));
}

BeaverTtp::Pair BeaverTtp::Eqz(FieldType field, int64_t size) {
  std::vector<PrgArrayDesc> descs(2);
  std::vector<absl::Span<const PrgSeedBuff>> descs_seed(1, encrypted_seeds_);
  Shape shape({size, 1});

  auto a = prgCreateArray(field, shape, seed_, &counter_, descs.data());
  auto b = prgCreateArray(field, shape, seed_, &counter_, &descs[1]);

  if (lctx_->Rank() == options_.adjust_rank) {
    auto req = BuildAdjustRequest<beaver::ttp_server::AdjustEqzRequest>(
        descs, descs_seed);
    auto adjusts = RpcCall(channel_, req, field);
    SPU_ENFORCE_EQ(adjusts.size(), 1U);
    ring_xor_(b, adjusts[0].reshape(shape));
  }

  Pair ret;
  ret.first = std::move(*a.buf());
  ret.second = std::move(*b.buf());
  return ret;
}
}  // namespace spu::mpc::semi2k
