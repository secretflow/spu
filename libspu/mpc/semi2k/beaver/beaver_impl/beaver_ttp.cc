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

#include <condition_variable>
#include <future>
#include <mutex>
#include <utility>
#include <vector>

#include "brpc/progressive_reader.h"
#include "yacl/crypto/pke/sm2_enc.h"
#include "yacl/crypto/rand/rand.h"
#include "yacl/link/algorithm/allgather.h"
#include "yacl/link/algorithm/broadcast.h"
#include "yacl/utils/serialize.h"

#include "libspu/mpc/common/prg_tensor.h"
#include "libspu/mpc/utils/gfmp_ops.h"
#include "libspu/mpc/utils/permute.h"
#include "libspu/mpc/utils/ring_ops.h"

#include "libspu/mpc/semi2k/beaver/beaver_impl/ttp_server/service.pb.h"

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

  uint32_t field_size = 0;
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

class ProgressiveReader : public brpc::ProgressiveReader {
 public:
  ProgressiveReader(int32_t num_buf, int64_t buf_len)
      : buffer_remain_size_(buf_len * num_buf),

        receive_buffers_(num_buf) {
    for (auto& b : receive_buffers_) {
      b.resize(buf_len);
    }
  }

  butil::Status OnReadOnePart(const void* data, size_t length) final {
    size_t consumed = 0;
    try {
      while (consumed < length) {
        const auto* consume_data =
            reinterpret_cast<const uint8_t*>(data) + consumed;
        size_t remain_length = length - consumed;

        if (current_state_ == ReadFlags) {
          consumed += copy_to_flags(consume_data, remain_length);
        } else if (current_state_ == ReadChunk) {
          consumed += copy_to_buffer(consume_data, remain_length);
        } else if (current_state_ == ReadError) {
          consumed += copy_to_error(consume_data, remain_length);
        } else if (current_state_ == End) {
          return butil::Status(
              -1, "response size mismatch, receive data after end");
        }
      }
      if (current_state_ == End && !server_error_msg_.empty()) {
        return butil::Status(
            -1,
            fmt::format("server side error code {}, msg {}",
                        beaver::ttp_server::ErrorCode_Name(server_error_code_),
                        server_error_msg_));
      }
    } catch (const std::exception& e) {
      return butil::Status(-1, fmt::format("unexpected error {}", e.what()));
    }

    return butil::Status::OK();
  }

  void OnEndOfMessage(const butil::Status& status) final {
    {
      std::lock_guard lk(lock_);
      if (current_state_ == End) {
        // received all data.
        read_status_ = status;
      } else if (status.ok()) {
        // rpc streaming finished, but we expected more data
        read_status_ =
            butil::Status(-1, "response size mismatch, need more data");
      } else {
        // some error happend in network or OnReadOnePart
        read_status_ = status;
      }
    }
    cond_.notify_all();
  }

  void Wait() {
    {
      std::unique_lock lk(lock_);
      cond_.wait(lk, [this] { return read_status_.has_value(); });
    }
    SPU_ENFORCE(read_status_->ok(), "Beaver Streaming data err: {}",
                read_status_->error_str());
  }

  std::vector<yacl::Buffer> PopBuffer() {
    {
      std::lock_guard lk(lock_);
      SPU_ENFORCE(current_state_ == End, "pls wait streaming finished");
    }
    return std::move(receive_buffers_);
  }

 private:
  size_t copy_to_flags(const void* data, size_t length) {
    size_t cp_size = std::min(static_cast<size_t>(flags_.size() - flags_pos_), length);
    std::memcpy(flags_.data() + flags_pos_, data, cp_size);
    flags_pos_ += cp_size;
    if (flags_pos_ == flags_.size()) {
      flags_pos_ = 0;
      int64_t chunk_size = 0;
      std::memcpy(&chunk_size, &flags_[1], sizeof(int64_t));
      chunk_remain_size_ = chunk_size;
      if (flags_[0] == 0) {
        current_state_ = ReadChunk;
      } else if (beaver::ttp_server::ErrorCode_IsValid(flags_[0])) {
        server_error_code_ =
            static_cast<beaver::ttp_server::ErrorCode>(flags_[0]);
        current_state_ = ReadError;
      } else {
        SPU_THROW("unexpected flags[0] {}", flags_[0]);
      }
    }

    return cp_size;
  }

  size_t copy_to_buffer(const void* data, size_t length) {
    length = std::min(length, chunk_remain_size_);
    chunk_remain_size_ -= length;
    if (chunk_remain_size_ == 0) {
      current_state_ = ReadFlags;
    }

    if (length > buffer_remain_size_) {
      SPU_THROW("response size mismatch, too many data for buffer");
    }

    buffer_remain_size_ -= length;
    if (buffer_remain_size_ == 0) {
      current_state_ = End;
    }

    size_t data_pos = 0;
    while (data_pos < length) {
      if (current_buffer_idx_ >= receive_buffers_.size()) {
        SPU_THROW("response size mismatch, outof index");
      }
      auto& buffer = receive_buffers_[current_buffer_idx_];
      auto cp_size = std::min(length, static_cast<size_t>(buffer.size() - current_buffer_pos_));
      std::memcpy(buffer.data<uint8_t>() + current_buffer_pos_,
                  reinterpret_cast<const uint8_t*>(data) + data_pos, cp_size);
      current_buffer_pos_ += cp_size;
      if (current_buffer_pos_ == static_cast<size_t>(buffer.size())) {
        current_buffer_pos_ = 0;
        current_buffer_idx_ += 1;
      }
      data_pos += cp_size;
    }

    return length;
  }

  size_t copy_to_error(const void* data, size_t length) {
    length = std::min(length, chunk_remain_size_);
    chunk_remain_size_ -= length;
    if (chunk_remain_size_ == 0) {
      current_state_ = End;
    }

    server_error_msg_.append(reinterpret_cast<const char*>(data), length);
    return length;
  }

 private:
  enum State : uint8_t {
    ReadFlags = 0,
    ReadChunk = 1,
    ReadError = 2,
    End = 3,
  };
  State current_state_{ReadFlags};
  size_t flags_pos_{};
  std::array<uint8_t, 1 + sizeof(int64_t)> flags_;
  size_t chunk_remain_size_{};
  std::string server_error_msg_;
  beaver::ttp_server::ErrorCode server_error_code_;

  size_t buffer_remain_size_;
  size_t current_buffer_idx_{};
  size_t current_buffer_pos_{};
  std::vector<yacl::Buffer> receive_buffers_;

  std::mutex lock_;
  std::condition_variable cond_;
  std::optional<butil::Status> read_status_;
};

template <class AdjustRequest>
std::vector<NdArrayRef> RpcCall(brpc::Channel& channel,
                                const AdjustRequest& req, FieldType ret_field) {
  beaver::ttp_server::BeaverService::Stub stub(&channel);
  beaver::ttp_server::AdjustResponse rsp;
  brpc::Controller cntl;
  cntl.response_will_be_read_progressively();

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

  auto [num_buf, buf_len] = GetBufferLength(req);
  ProgressiveReader reader(num_buf, buf_len);
  cntl.ReadProgressiveAttachmentBy(&reader);
  reader.Wait();
  auto buffers = reader.PopBuffer();

  std::vector<NdArrayRef> ret;
  for (auto& buf : buffers) {
    SPU_ENFORCE(buf.size() % SizeOf(ret_field) == 0);
    int64_t size = buf.size() / SizeOf(ret_field);
    // FIXME: change beaver interface: change return type to buffer.
    // FIXME: TTP adjuster server and client MUST have same endianness.
    NdArrayRef array(std::make_shared<yacl::Buffer>(std::move(buf)),
                     makeType<RingTy>(ret_field), {size});
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
    SPU_ENFORCE(options_.brpc_channel_protocol == "http" ||
                    options_.brpc_channel_protocol == "h2",
                "beaver only support http 1.1 or http 2");
    brc_options.protocol = options_.brpc_channel_protocol;
    brc_options.timeout_ms = options_.brpc_timeout_ms;
    brc_options.max_retry = options_.brpc_max_retry;

    if (options_.brpc_ssl_options) {
      *brc_options.mutable_ssl_options() = options_.brpc_ssl_options.value();
    }

    if (channel_.Init(options_.server_host.c_str(), &brc_options) != 0) {
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

BeaverTtp::PremTriple BeaverTtp::PermPair(FieldType field, int64_t size,
                                          size_t perm_rank) {
  constexpr char kTag[] = "BEAVER_TFP:PERM";
  std::vector<PrgArrayDesc> descs(2);
  std::vector<absl::Span<const PrgSeedBuff>> descs_seed(1, encrypted_seeds_);
  Shape shape({size, 1});

  auto a = prgCreateArray(field, shape, seed_, &counter_, descs.data());
  auto b = prgCreateArray(field, shape, seed_, &counter_, &descs[1]);

  if (lctx_->Rank() == options_.adjust_rank) {
    auto req = BuildAdjustRequest<beaver::ttp_server::AdjustPermRequest>(
        descs, descs_seed);
    auto* perm_meta = req.mutable_perm();
    perm_meta->set_prg_count(counter_);
    perm_meta->set_size(size);
    auto& perm_seed = encrypted_seeds_[perm_rank];
    perm_meta->set_encrypted_seeds(perm_seed.data(), perm_seed.size());
    auto adjusts = RpcCall(channel_, req, field);
    SPU_ENFORCE_EQ(adjusts.size(), 1U);
    ring_add_(b, adjusts[0].reshape(b.shape()));
  }

  Index pi;
  if (lctx_->Rank() == perm_rank) {
    pi = genRandomPerm(size, seed_, &counter_);
  }

  auto new_counter_buf = yacl::link::Broadcast(
      lctx_, yacl::SerializeVars<PrgCounter>(counter_), perm_rank, kTag);

  counter_ = yacl::DeserializeVars<PrgCounter>(new_counter_buf);

  PremTriple ret;
  std::get<0>(ret) = std::move(*a.buf());
  std::get<1>(ret) = std::move(*b.buf());
  std::get<2>(ret) = std::move(pi);

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
