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

#include "libspu/mpc/cheetah/ot/yacl/ferret.h"

#include <utility>

#include "spdlog/spdlog.h"
#include "yacl/base/buffer.h"
#include "yacl/crypto/tools/crhash.h"
#include "yacl/crypto/tools/rp.h"
#include "yacl/link/link.h"

#include "libspu/mpc/cheetah/ot/ot_util.h"
#include "libspu/mpc/cheetah/ot/yacl/mitccrh_exp.h"
#include "libspu/mpc/cheetah/ot/yacl/yacl_ote_adapter.h"
#include "libspu/mpc/cheetah/ot/yacl/yacl_util.h"

namespace spu::mpc::cheetah {

constexpr size_t kOTBatchSize = 8;  // emp-ot/cot.h

class BufferedIO {
 public:
  std::shared_ptr<Communicator> conn_;

  constexpr static uint64_t SEND_BUFFER_SIZE = 1024 * 1024;

  uint32_t send_op_;
  uint32_t recv_op_;

  std::vector<uint8_t> send_buffer_;
  uint64_t send_buffer_used_;

  std::vector<uint8_t> recv_buffer_;
  uint64_t recv_buffer_used_;

  explicit BufferedIO(std::shared_ptr<Communicator> conn)
      : conn_(std::move(conn)),
        send_op_(0),
        recv_op_(0),
        send_buffer_used_(0),
        recv_buffer_used_(0) {
    send_buffer_.resize(SEND_BUFFER_SIZE);
  }

  ~BufferedIO() {
    try {
      flush();
    } catch (const std::exception& e) {
      SPDLOG_ERROR("Error in flush: {}", e.what());
    }
  }

  void flush() {
    if (send_buffer_used_ == 0) {
      return;
    }

    conn_->sendAsync(
        conn_->nextRank(),
        absl::Span<const uint8_t>{send_buffer_.data(), send_buffer_used_},
        fmt::format("BufferedIO send:{}", send_op_++));

    std::memset(send_buffer_.data(), 0, SEND_BUFFER_SIZE);
    send_buffer_used_ = 0;
  }

  void fill_recv() {
    recv_buffer_ = conn_->recv<uint8_t>(
        conn_->nextRank(), fmt::format("BufferedIO recv:{}", recv_op_++));
    recv_buffer_used_ = 0;
  }

  void send_data(const void* data, int len) {
    size_t send_buffer_left = SEND_BUFFER_SIZE - send_buffer_used_;
    if (send_buffer_left <= static_cast<size_t>(len)) {
      std::memcpy(send_buffer_.data() + send_buffer_used_, data,
                  send_buffer_left);
      send_buffer_used_ += send_buffer_left;
      flush();

      send_data(static_cast<const char*>(data) + send_buffer_left,
                len - send_buffer_left);
    } else {
      std::memcpy(send_buffer_.data() + send_buffer_used_, data, len);
      send_buffer_used_ += len;
    }
  }

  void recv_data(void* data, int len) {
    if (send_buffer_used_ > 0) {
      flush();
    }

    size_t recv_buffer_left = recv_buffer_.size() - recv_buffer_used_;
    if (recv_buffer_left >= static_cast<size_t>(len)) {
      std::memcpy(data, recv_buffer_.data() + recv_buffer_used_, len);
      recv_buffer_used_ += len;
    } else {
      if (recv_buffer_.size() != 0) {
        std::memcpy(data, recv_buffer_.data() + recv_buffer_used_,
                    recv_buffer_left);
      }
      fill_recv();

      recv_data(static_cast<char*>(data) + recv_buffer_left,
                len - recv_buffer_left);
    }
  }
};

struct YaclFerretOt::Impl {
 private:
  const bool is_sender_;

  std::shared_ptr<BufferedIO> io_{nullptr};
  std::shared_ptr<YaclOTeAdapter> ferret_{nullptr};

  MITCCRHExp<8> mitccrh_exp_{};

  inline void SendRCOT(absl::Span<uint128_t> output) {
    SPU_ENFORCE(is_sender_);
    ferret_->send_rcot(output);
  }

  inline void RecvRCOT(absl::Span<uint8_t> choices,
                       absl::Span<uint128_t> output) {
    SPU_ENFORCE(!is_sender_);
    ferret_->recv_rcot(output, choices);
  }

  inline void SendCOT(uint128_t* output, size_t n) {
    SPU_ENFORCE(is_sender_);
    ferret_->send_cot(absl::MakeSpan(output, n));
  }

  inline void RecvCOT(absl::Span<const uint8_t> choices,
                      absl::Span<uint128_t> output) {
    SPU_ENFORCE(!is_sender_);
    ferret_->recv_cot(output, choices);
  }

 public:
  Impl(std::shared_ptr<Communicator> conn, bool is_sender, bool use_soft_spoken)
      : is_sender_(is_sender) {
    SPU_ENFORCE(conn != nullptr);

    io_ = std::make_shared<BufferedIO>(conn);
    if (use_soft_spoken) {
      ferret_ = std::make_shared<YaclSsOTeAdapter>(conn->lctx(), is_sender);
    } else {
      ferret_ = std::make_shared<YaclFerretOTeAdapter>(conn->lctx(), is_sender);
    }
    ferret_->OneTimeSetup();
  }

  ~Impl() = default;

  int Rank() const { return io_->conn_->getRank(); }

  void Flush() {
    if (io_) {
      io_->flush();
    }
  }

  void SendRandCorrelatedMsgChosenChoice(uint128_t* output, size_t n) {
    SendCOT(output, n);
  }

  void RecvRandCorrelatedMsgChosenChoice(absl::Span<const uint8_t> choices,
                                         absl::Span<uint128_t> output) {
    SPU_ENFORCE_EQ(choices.size(), output.size());
    SPU_ENFORCE(!output.empty());
    RecvCOT(choices, output);
  }

  void SendRandMsgChosenChoice(uint128_t* msg0, uint128_t* msg1, size_t n) {
    SendRandCorrelatedMsgChosenChoice(msg0, n);
    auto msg_span0 = absl::MakeSpan(msg0, n);
    auto msg_span1 = absl::MakeSpan(msg1, n);
    auto delta = ferret_->GetDelta();

    // Use CrHash to Break the correlated randomness
    for (uint64_t i = 0; i < n; ++i) {
      msg_span1[i] = msg_span0[i] ^ delta;
    }
    yc::ParaCrHashInplace_128(msg_span0);
    yc::ParaCrHashInplace_128(msg_span1);
  }

  void RecvRandMsgChosenChoice(absl::Span<const uint8_t> choices,
                               absl::Span<uint128_t> output) {
    size_t n = output.size();
    SPU_ENFORCE(n > 0);
    SPU_ENFORCE_EQ(choices.size(), n);
    RecvRandCorrelatedMsgChosenChoice(choices, output);

    // Use CrHash to Break the correlated randomness
    yc::ParaCrHashInplace_128(output);
  }

  template <typename T>
  void SendCorrelatedMsgChosenChoice_Collapse(absl::Span<const T> corr,
                                              absl::Span<T> output,
                                              int bit_width, int num_level) {
    size_t n = corr.size();
    SPU_ENFORCE_EQ(n, output.size());
    if (bit_width == 0) {
      bit_width = 8 * sizeof(T);
    }
    SPU_ENFORCE(bit_width > 0 && bit_width <= (int)(8 * sizeof(T)),
                "bit_width={} out-of-range T={} bits", bit_width,
                sizeof(T) * 8);
    SPU_ENFORCE(
        num_level > 0 && (n % num_level) == 0 && (bit_width >= num_level),
        "invalid num_level {}", num_level);

    yacl::Buffer buf(n * sizeof(uint128_t));
    auto rcm_output = MakeSpan_Uint128(buf);

    SendRandCorrelatedMsgChosenChoice(rcm_output.data(), n);

    std::array<uint128_t, 2 * kOTBatchSize> pad;
    std::vector<T> corr_output(kOTBatchSize);

    const size_t eltsize = 8 * sizeof(T);
    const size_t collapse_size = n / num_level;
    std::vector<T> packed_corr_output(kOTBatchSize);
    for (size_t i = 0; i < n; i += kOTBatchSize) {
      size_t this_batch = std::min(kOTBatchSize, n - i);
      // NOTE(lwj) this batch might cross two collapse, but the bit_width should
      // be fine since we decrease the bit_width along each collapse batch.
      size_t this_batch_bw = bit_width - i / collapse_size;
      bool packable = eltsize > this_batch_bw;

      for (size_t j = 0; j < this_batch; ++j) {
        pad[2 * j] = rcm_output[i + j];
        pad[2 * j + 1] = rcm_output[i + j] ^ ferret_->GetDelta();
      }

      yc::ParaCrHashInplace_128(absl::MakeSpan(pad));

      for (size_t j = 0; j < this_batch; ++j) {
        output[i + j] = (T)(pad[2 * j]);
        corr_output[j] = (T)(pad[2 * j + 1]);
        corr_output[j] += corr[i + j] + output[i + j];
      }

      if (packable) {
        size_t used =
            ZipArray<T>({corr_output.data(), this_batch}, this_batch_bw,
                        absl::MakeSpan(packed_corr_output));
        SPU_ENFORCE(used == CeilDiv(this_batch * this_batch_bw, eltsize));
        io_->send_data(packed_corr_output.data(), used * sizeof(T));
      } else {
        io_->send_data(corr_output.data(), sizeof(T) * this_batch);
      }
    }
    io_->flush();
  }

  template <typename T>
  void RecvCorrelatedMsgChosenChoice_Collapse(absl::Span<const uint8_t> choices,
                                              absl::Span<T> output,
                                              int bit_width, int num_level) {
    size_t n = choices.size();
    SPU_ENFORCE_EQ(n, output.size());
    if (bit_width == 0) {
      bit_width = 8 * sizeof(T);
    }
    SPU_ENFORCE(bit_width > 0 && bit_width <= (int)(8 * sizeof(T)),
                "bit_width={} out-of-range T={} bits", bit_width,
                sizeof(T) * 8);
    SPU_ENFORCE(
        num_level > 0 && (n % num_level) == 0 && (bit_width >= num_level),
        "invalid num_level {}", num_level);

    yacl::Buffer buf(n * sizeof(uint128_t));
    auto rcm_output = MakeSpan_Uint128(buf);

    RecvRandCorrelatedMsgChosenChoice(choices, rcm_output);

    std::array<uint128_t, kOTBatchSize> pad;
    std::vector<T> corr_output(kOTBatchSize);

    const size_t eltsize = 8 * sizeof(T);
    const size_t collapse_size = n / num_level;
    std::vector<T> packed_corr_output(kOTBatchSize);
    for (size_t i = 0; i < n; i += kOTBatchSize) {
      size_t this_batch = std::min(kOTBatchSize, n - i);
      size_t this_batch_bw = bit_width - i / collapse_size;
      bool packable = eltsize > this_batch_bw;

      std::memcpy(pad.data(), rcm_output.data() + i,
                  this_batch * sizeof(uint128_t));
      // Use CrHash
      yc::ParaCrHashInplace_128(absl::MakeSpan(pad));

      if (packable) {
        size_t used = CeilDiv(this_batch * this_batch_bw, eltsize);
        io_->recv_data(packed_corr_output.data(), sizeof(T) * used);
        UnzipArray<T>({packed_corr_output.data(), used}, this_batch_bw,
                      {corr_output.data(), this_batch});
      } else {
        io_->recv_data(corr_output.data(), sizeof(T) * this_batch);
      }

      for (size_t j = 0; j < this_batch; ++j) {
        output[i + j] = (T)(pad[j]);
        if (choices[i + j]) {
          output[i + j] = corr_output[j] - output[i + j];
        }
      }
    }
  }

  template <typename T>
  void SendCorrelatedMsgChosenChoice(absl::Span<const T> corr,
                                     absl::Span<T> output, int bit_width) {
    size_t n = corr.size();
    SPU_ENFORCE_EQ(n, output.size());
    if (bit_width == 0) {
      bit_width = 8 * sizeof(T);
    }
    SPU_ENFORCE(bit_width > 0 && bit_width <= (int)(8 * sizeof(T)),
                "bit_width={} out-of-range T={} bits", bit_width,
                sizeof(T) * 8);

    yacl::Buffer buf(n * sizeof(uint128_t));
    auto rcm_output = MakeSpan_Uint128(buf);

    SendRandCorrelatedMsgChosenChoice(rcm_output.data(), n);

    std::array<uint128_t, 2 * kOTBatchSize> pad;
    std::vector<T> corr_output(kOTBatchSize);

    size_t eltsize = 8 * sizeof(T);
    bool packable = eltsize > (size_t)bit_width;
    size_t packed_size = CeilDiv(kOTBatchSize * bit_width, eltsize);

    std::vector<T> packed_corr_output;
    if (packable) {
      packed_corr_output.resize(packed_size);
    }

    for (size_t i = 0; i < n; i += kOTBatchSize) {
      size_t this_batch = std::min(kOTBatchSize, n - i);
      for (size_t j = 0; j < this_batch; ++j) {
        pad[2 * j] = rcm_output[i + j];
        pad[2 * j + 1] = rcm_output[i + j] ^ ferret_->GetDelta();
      }

      yc::ParaCrHashInplace_128(absl::MakeSpan(pad));

      for (size_t j = 0; j < this_batch; ++j) {
        output[i + j] = (T)(pad[2 * j]);
        corr_output[j] = (T)(pad[2 * j + 1]);
        corr_output[j] += corr[i + j] + output[i + j];
      }

      if (packable) {
        size_t used = ZipArray<T>({corr_output.data(), this_batch}, bit_width,
                                  absl::MakeSpan(packed_corr_output));
        SPU_ENFORCE(used == CeilDiv(this_batch * bit_width, eltsize));
        io_->send_data(packed_corr_output.data(), used * sizeof(T));
      } else {
        io_->send_data(corr_output.data(), sizeof(T) * this_batch);
      }
    }
    io_->flush();
  }

  template <typename T>
  void RecvCorrelatedMsgChosenChoice(absl::Span<const uint8_t> choices,
                                     absl::Span<T> output, int bit_width) {
    size_t n = choices.size();
    SPU_ENFORCE_EQ(n, output.size());
    if (bit_width == 0) {
      bit_width = 8 * sizeof(T);
    }
    SPU_ENFORCE(bit_width > 0 && bit_width <= (int)(8 * sizeof(T)),
                "bit_width={} out-of-range T={} bits", bit_width,
                sizeof(T) * 8);

    yacl::Buffer buf(n * sizeof(uint128_t));
    auto rcm_output = MakeSpan_Uint128(buf);

    RecvRandCorrelatedMsgChosenChoice(choices, rcm_output);

    std::array<uint128_t, kOTBatchSize> pad;
    std::vector<T> corr_output(kOTBatchSize);

    size_t eltsize = 8 * sizeof(T);
    bool packable = eltsize > (size_t)bit_width;
    size_t packed_size = CeilDiv(kOTBatchSize * bit_width, eltsize);

    std::vector<T> packed_corr_output;
    if (packable) {
      packed_corr_output.resize(packed_size);
    }

    for (size_t i = 0; i < n; i += kOTBatchSize) {
      size_t this_batch = std::min(kOTBatchSize, n - i);

      std::memcpy(pad.data(), rcm_output.data() + i,
                  this_batch * sizeof(uint128_t));
      // Use CrHash
      yc::ParaCrHashInplace_128(absl::MakeSpan(pad));

      if (packable) {
        size_t used = CeilDiv(this_batch * bit_width, eltsize);
        io_->recv_data(packed_corr_output.data(), sizeof(T) * used);
        UnzipArray<T>({packed_corr_output.data(), used}, bit_width,
                      {corr_output.data(), this_batch});
      } else {
        io_->recv_data(corr_output.data(), sizeof(T) * this_batch);
      }

      for (size_t j = 0; j < this_batch; ++j) {
        output[i + j] = (T)(pad[j]);
        if (choices[i + j]) {
          output[i + j] = corr_output[j] - output[i + j];
        }
      }
    }
  }

  void RecvRandMsgRandChoice(absl::Span<uint8_t> choices,
                             absl::Span<uint128_t> output) {
    size_t n = choices.size();
    SPU_ENFORCE(n > 0);
    SPU_ENFORCE_EQ(n, output.size());
    RecvRCOT(choices, output);

    yc::ParaCrHashInplace_128(output);
  }

  void SendRandMsgRandChoice(absl::Span<uint128_t> output0,
                             absl::Span<uint128_t> output1) {
    size_t n = output0.size();
    SPU_ENFORCE(n > 0);
    SPU_ENFORCE_EQ(n, output1.size());
    SendRCOT(output0);
    auto delta = ferret_->GetDelta();

    // Use CrHash to Break the correlated randomness
    for (uint64_t i = 0; i < n; ++i) {
      output1[i] = output0[i] ^ delta;
    }
    yc::ParaCrHashInplace_128(output0);
    yc::ParaCrHashInplace_128(output1);
  }

  void SendChosenMsgChosenChoice(const uint128_t* msg0, const uint128_t* msg1,
                                 size_t n) {
    SPU_ENFORCE(msg0 != nullptr && msg1 != nullptr);
    SPU_ENFORCE(n > 0);

    yacl::Buffer buf(n * sizeof(uint128_t));
    auto rcm_data = MakeSpan_Uint128(buf);

    SendRandCorrelatedMsgChosenChoice(rcm_data.data(), n);

    uint128_t delta = ferret_->GetDelta();
    std::array<uint128_t, 2 * kOTBatchSize> pad;
    for (size_t i = 0; i < n; i += kOTBatchSize) {
      size_t this_batch = std::min(kOTBatchSize, n - i);

      for (size_t j = 0; j < this_batch; ++j) {
        pad[2 * j] = rcm_data[i + j];
        pad[2 * j + 1] = rcm_data[i + j] ^ delta;
      }

      yc::ParaCrHashInplace_128(absl::MakeSpan(pad));
      for (size_t j = 0; j < this_batch; ++j) {
        pad[2 * j] ^= msg0[i + j];
        pad[2 * j + 1] ^= msg1[i + j];
      }

      io_->send_data(pad.data(), 2 * sizeof(uint128_t) * this_batch);
    }
    io_->flush();
  }

  void RecvChosenMsgChosenChoice(absl::Span<const uint8_t> choices,
                                 absl::Span<uint128_t> output) {
    size_t n = choices.size();
    SPU_ENFORCE(n > 0);
    SPU_ENFORCE_EQ(n, output.size());
    RecvRandCorrelatedMsgChosenChoice(choices, output);

    std::array<uint128_t, kOTBatchSize> pad;
    std::array<uint128_t, 2 * kOTBatchSize> recv;
    for (size_t i = 0; i < n; i += kOTBatchSize) {
      size_t this_batch = std::min(kOTBatchSize, n - i);
      size_t nbytes = this_batch * sizeof(uint128_t);
      std::memcpy(pad.data(), output.data() + i, nbytes);
      yc::ParaCrHashInplace_128(absl::MakeSpan(pad));
      io_->recv_data(recv.data(), 2 * nbytes);
      for (size_t j = 0; j < this_batch; ++j) {
        output[i + j] = recv[2 * j + (choices[i + j] & 1)] ^ pad[j];
      }
    }
  }

  template <typename T>
  void SendRandMsgRandChoice(absl::Span<T> output0, absl::Span<T> output1,
                             size_t bit_width = 0) {
    size_t n = output0.size();
    SPU_ENFORCE(n > 0);
    SPU_ENFORCE_EQ(n, output1.size());
    const T mask = makeBitsMask<T>(bit_width);

    yacl::Buffer buf(2 * n * sizeof(uint128_t));
    auto rm_data = MakeSpan_Uint128(buf);

    auto rm_data0 = rm_data.subspan(0, n);
    auto rm_data1 = rm_data.subspan(n, n);
    SendRandMsgRandChoice(rm_data0, rm_data1);

    std::transform(rm_data0.cbegin(), rm_data0.cend(), output0.data(),
                   [mask](const uint128_t& x) { return (T)x & mask; });
    std::transform(rm_data1.cbegin(), rm_data1.cend(), output1.data(),
                   [mask](const uint128_t& x) { return (T)x & mask; });
  }

  template <typename T>
  void RecvRandMsgRandChoice(absl::Span<uint8_t> choices, absl::Span<T> output,
                             size_t bit_width = 0) {
    size_t n = choices.size();
    SPU_ENFORCE(n > 0);
    SPU_ENFORCE_EQ(n, output.size());
    const T mask = makeBitsMask<T>(bit_width);

    yacl::Buffer buf(n * sizeof(uint128_t));
    auto rm_data = MakeSpan_Uint128(buf);

    RecvRandMsgRandChoice(choices, rm_data);

    std::transform(rm_data.cbegin(), rm_data.cend(), output.data(),
                   [mask](const uint128_t& x) { return ((T)x) & mask; });
  }

  template <typename T>
  void SendChosenMsgChosenChoice(absl::Span<const T> msg_array, size_t N,
                                 size_t bit_width) {
    SPU_ENFORCE(N >= 2 && N <= 256,
                fmt::format("N should 2 <= N <= 256, but got N={}", N));
    SPU_ENFORCE(bit_width > 0 && bit_width <= 8 * sizeof(T));
    const size_t Nn = msg_array.size();
    SPU_ENFORCE(Nn > 0 && 0 == (Nn % N));
    const size_t n = Nn / N;
    size_t logN = absl::bit_width(N) - 1;

    // REF: "Oblivious transfer and polynomial evaluation"
    // Construct one-of-N-OT from logN instances of one-of-2-OT
    // Send: (s_{0, j}, s_{1, j}) for 0 <= j < logN
    // Recv:  c_j \in {0, 1}

    yacl::Buffer buf_data0(n * logN * sizeof(uint128_t));
    yacl::Buffer buf_data1(n * logN * sizeof(uint128_t));
    auto rm_data0 = MakeSpan_Uint128(buf_data0);
    auto rm_data1 = MakeSpan_Uint128(buf_data1);

    SendRandMsgChosenChoice(rm_data0.data(), rm_data1.data(), n * logN);

    yacl::Buffer buf_in0((N - 1) * sizeof(uint128_t));
    yacl::Buffer buf_in1((N - 1) * sizeof(uint128_t));
    auto hash_in0 = MakeSpan_Uint128(buf_in0);
    auto hash_in1 = MakeSpan_Uint128(buf_in1);

    {
      size_t idx = 0;
      for (size_t x = 0; x < logN; ++x) {
        for (size_t y = 0; y < (1UL << x); ++y) {
          hash_in0[idx] = yacl::MakeUint128(y, 0);
          hash_in1[idx] = yacl::MakeUint128((1 << x) | y, 0);
          ++idx;
        }
      }
    }

    yacl::Buffer buf_out0((N - 1) * sizeof(uint128_t));
    yacl::Buffer buf_out1((N - 1) * sizeof(uint128_t));
    yacl::Buffer buf_pad(kOTBatchSize * N * sizeof(uint128_t));

    auto hash_out0 = MakeSpan_Uint128(buf_out0);
    auto hash_out1 = MakeSpan_Uint128(buf_out1);
    auto pad = MakeSpan_Uint128(buf_pad);

    const T msg_mask = makeBitsMask<T>(bit_width);
    size_t eltsize = 8 * sizeof(T);
    bool packable = eltsize > (size_t)bit_width;

    std::vector<T> to_send(kOTBatchSize * N);
    size_t packed_size = CeilDiv(kOTBatchSize * N * bit_width, eltsize);
    std::vector<T> packed_to_send;
    if (packable) {
      // NOTE: pack bit chunks into single T element if possible
      packed_to_send.resize(packed_size);
    }

    for (size_t i = 0; i < n; i += kOTBatchSize) {
      size_t this_batch = std::min(kOTBatchSize, n - i);
      std::memset(pad.data(), 0, pad.size() * sizeof(uint128_t));

      for (size_t j = 0; j < this_batch; ++j) {
        mitccrh_exp_.renew_ks(
            reinterpret_cast<uint128_t*>(&rm_data0[(i + j) * logN]), logN);
        mitccrh_exp_.hash_exp(hash_out0.data(), hash_in0.data(), logN);

        mitccrh_exp_.renew_ks(
            reinterpret_cast<uint128_t*>(&rm_data1[(i + j) * logN]), logN);
        mitccrh_exp_.hash_exp(hash_out1.data(), hash_in1.data(), logN);

        for (size_t k = 0; k < N; ++k) {
          size_t idx = 0;
          for (size_t s = 0; s < logN; ++s) {
            uint32_t prefer = k & ((1 << s) - 1);
            SPU_ENFORCE(idx + prefer + 1 < N);
            if (0 == (k & (1 << s))) {
              pad[j * N + k] ^= hash_out0[idx + prefer];
            } else {
              pad[j * N + k] ^= hash_out1[idx + prefer];
            }
            idx += (1 << s);
          }
        }
      }  // loop-j

      // The i-th messages start msg_array[i * N] to msg_array[i * N + N -
      // 1]
      for (size_t j = 0; j < this_batch; ++j) {
        const auto* this_msg = msg_array.data() + (i + j) * N;
        for (size_t k = 0; k < N; ++k) {
          to_send[j * N + k] = ((T)(pad[j * N + k]) ^ this_msg[k]) & msg_mask;
          // to_send[j * N + k] &= msg_mask;
        }
      }

      if (packable) {
        size_t used = ZipArray<T>({to_send.data(), N * this_batch}, bit_width,
                                  absl::MakeSpan(packed_to_send));
        SPU_ENFORCE(used == CeilDiv(N * this_batch * bit_width, eltsize));
        io_->send_data(packed_to_send.data(), used * sizeof(T));
      } else {
        io_->send_data(to_send.data(), N * this_batch * sizeof(T));
      }
    }
    io_->flush();
  }

  template <typename T>
  void RecvChosenMsgChosenChoice(absl::Span<const uint8_t> choices, size_t N,
                                 absl::Span<T> output, size_t bit_width) {
    SPU_ENFORCE(N >= 2 && N <= 256,
                fmt::format("N should 2 <= N <= 256, but got N={}", N));
    SPU_ENFORCE(bit_width > 0 && bit_width <= 8 * sizeof(T));

    const size_t n = choices.size();
    SPU_ENFORCE_EQ(output.size(), n);
    const size_t logN = absl::bit_width(N) - 1;

    SPU_ENFORCE(std::all_of(choices.data(), choices.data() + n,
                            [N](uint8_t b) { return b < N; }),
                "choice out-of-bound N");
    std::vector<uint8_t> bool_choices(n * logN, 0);
    // decomposite into binary form
    for (size_t i = 0; i < n; ++i) {
      uint8_t c = choices[i];
      for (size_t j = 0; j < logN; ++j, c >>= 1) {
        bool_choices[i * logN + j] = c & 1;
      }
    }

    // rm_data[logN * i + k] = 1-of-2 OT on the k-th bits of the i-th
    // message
    yacl::Buffer buf(n * logN * sizeof(uint128_t));
    auto rm_data = MakeSpan_Uint128(buf);

    RecvRandMsgChosenChoice(absl::MakeSpan(bool_choices), rm_data);

    yacl::Buffer buf_in(logN * sizeof(uint128_t));
    yacl::Buffer buf_out(logN * sizeof(uint128_t));
    yacl::Buffer buf_pad(kOTBatchSize * sizeof(uint128_t));

    auto hash_in = MakeSpan_Uint128(buf_in);
    auto hash_out = MakeSpan_Uint128(buf_out);
    auto pad = MakeSpan_Uint128(buf_pad);

    const T msg_mask = makeBitsMask<T>(bit_width);
    size_t eltsize = 8 * sizeof(T);
    bool packable = eltsize > (size_t)bit_width;
    size_t packed_size = CeilDiv(kOTBatchSize * N * bit_width, eltsize);

    std::vector<T> recv(kOTBatchSize * N);
    std::vector<T> packed_recv;
    if (packable) {
      packed_recv.resize(packed_size);
    }

    for (size_t i = 0; i < n; i += kOTBatchSize) {
      size_t this_batch = std::min(kOTBatchSize, n - i);
      size_t used = CeilDiv(N * this_batch * bit_width, eltsize);
      if (packable) {
        io_->recv_data(packed_recv.data(), used * sizeof(T));
        UnzipArray<T>({packed_recv.data(), used}, bit_width,
                      {recv.data(), N * this_batch});
      } else {
        io_->recv_data(recv.data(), N * this_batch * sizeof(T));
      }

      std::memset(pad.data(), 0, kOTBatchSize * sizeof(uint128_t));
      for (size_t j = 0; j < this_batch; ++j) {
        for (size_t s = 0; s < logN; s++) {
          auto h = choices[i + j] & makeBitsMask<uint8_t>(1 + s);
          hash_in[s] = yacl::MakeUint128(h, 0);
        }
        mitccrh_exp_.renew_ks(
            reinterpret_cast<uint128_t*>(&rm_data[(i + j) * logN]), logN);
        mitccrh_exp_.hash_single(hash_out.data(), hash_in.data(), logN);

        pad[j] = std::accumulate(hash_out.begin(), hash_out.end(), pad[j],
                                 std::bit_xor<uint128_t>());
      }

      for (size_t j = 0; j < this_batch; ++j) {
        output[i + j] = ((T)(pad[j]) ^ recv[N * j + choices[i + j]]) & msg_mask;
      }
    }
  }

  template <typename T>
  void SendRMCC(absl::Span<T> output0, absl::Span<T> output1,
                size_t bit_width) {
    const size_t n = output0.size();
    SPU_ENFORCE(n > 0);
    SPU_ENFORCE_EQ(n, output1.size());

    yacl::Buffer buf(2 * n * sizeof(uint128_t));
    auto rm_data = MakeSpan_Uint128(buf);

    auto* rm_data0 = rm_data.data();
    auto* rm_data1 = rm_data0 + n;
    SendRandMsgChosenChoice(rm_data0, rm_data1, n);

    // Type conversion
    const T msg_mask = makeBitsMask<T>(bit_width);
    std::transform(
        rm_data0, rm_data0 + n, output0.data(),
        [msg_mask](const uint128_t& val) { return ((T)val) & msg_mask; });
    std::transform(
        rm_data1, rm_data1 + n, output1.data(),
        [msg_mask](const uint128_t& val) { return ((T)val) & msg_mask; });
  }

  // Modified by @wenfan
  template <typename T>
  void RecvRMCC(absl::Span<const uint8_t> choices, absl::Span<T> output,
                size_t bit_width) {
    const size_t n = choices.size();
    SPU_ENFORCE(n > 0);
    SPU_ENFORCE_EQ(n, output.size());

    yacl::Buffer buf(n * sizeof(uint128_t));
    auto rm_data = MakeSpan_Uint128(buf);

    RecvRandMsgChosenChoice(choices, rm_data);

    // Type conversion
    const T msg_mask = makeBitsMask<T>(bit_width);
    std::transform(
        rm_data.begin(), rm_data.end(), output.begin(),
        [msg_mask](const uint128_t& val) { return ((T)val) & msg_mask; });
  }

  // Inplace
  void RecvRandMsgRandChoice(absl::Span<uint8_t> choices,
                             absl::Span<uint128_t> output, size_t bit_width) {
    size_t n = choices.size();
    SPU_ENFORCE(n > 0);
    SPU_ENFORCE_EQ(n, output.size());
    const uint128_t mask = makeBitsMask<uint128_t>(bit_width);

    RecvRandMsgRandChoice(choices, output);

    for (size_t i = 0; i < n; ++i) {
      output[i] &= mask;
    }
  }

  // Inplace
  void SendRandMsgRandChoice(absl::Span<uint128_t> output0,
                             absl::Span<uint128_t> output1, size_t bit_width) {
    size_t n = output0.size();
    SPU_ENFORCE(n > 0);
    SPU_ENFORCE_EQ(n, output1.size());
    const uint128_t mask = makeBitsMask<uint128_t>(bit_width);

    SendRandMsgRandChoice(output0, output1);
    for (size_t i = 0; i < n; ++i) {
      output0[i] &= mask;
    }
    for (size_t i = 0; i < n; ++i) {
      output1[i] &= mask;
    }
  }

  // Inplace
  void SendRMCC(absl::Span<uint128_t> output0, absl::Span<uint128_t> output1,
                size_t bit_width) {
    const size_t n = output0.size();
    SPU_ENFORCE(n > 0);
    SPU_ENFORCE_EQ(n, output1.size());

    SendRandMsgChosenChoice(output0.data(), output1.data(), n);

    const uint128_t msg_mask = makeBitsMask<uint128_t>(bit_width);

    for (size_t i = 0; i < n; ++i) {
      output0[i] &= msg_mask;
    }
    for (size_t i = 0; i < n; ++i) {
      output1[i] &= msg_mask;
    }
  }

  // Inplace
  void RecvRMCC(absl::Span<const uint8_t> choices, absl::Span<uint128_t> output,
                size_t bit_width) {
    const size_t n = choices.size();
    SPU_ENFORCE(n > 0);
    SPU_ENFORCE_EQ(n, output.size());

    RecvRandMsgChosenChoice(choices, output);
    const uint128_t msg_mask = makeBitsMask<uint128_t>(bit_width);

    for (size_t i = 0; i < n; ++i) {
      output[i] &= msg_mask;
    }
  }
};

YaclFerretOt::YaclFerretOt(std::shared_ptr<Communicator> conn, bool is_sender,
                           bool use_soft_spoken) {
  impl_ = std::make_shared<Impl>(conn, is_sender, use_soft_spoken);
}

int YaclFerretOt::Rank() const { return impl_->Rank(); }

void YaclFerretOt::Flush() { impl_->Flush(); }

YaclFerretOt::~YaclFerretOt() { impl_->Flush(); }

template <typename T>
size_t CheckBitWidth(size_t bw) {
  size_t m = 8 * sizeof(T);
  SPU_ENFORCE(bw <= m);
  if (bw == 0) {
    bw = m;
  }
  return bw;
}

#define DEF_SEND_RECV(T)                                                       \
  void YaclFerretOt::SendCAMCC(absl::Span<const T> corr, absl::Span<T> output, \
                               int bw) {                                       \
    impl_->SendCorrelatedMsgChosenChoice<T>(corr, output, bw);                 \
  }                                                                            \
  void YaclFerretOt::RecvCAMCC(absl::Span<const uint8_t> choices,              \
                               absl::Span<T> output, int bw) {                 \
    impl_->RecvCorrelatedMsgChosenChoice<T>(choices, output, bw);              \
  }                                                                            \
  void YaclFerretOt::SendCAMCC_Collapse(                                       \
      absl::Span<const T> corr, absl::Span<T> output, int bw, int num_level) { \
    impl_->SendCorrelatedMsgChosenChoice_Collapse<T>(corr, output, bw,         \
                                                     num_level);               \
  }                                                                            \
  void YaclFerretOt::RecvCAMCC_Collapse(absl::Span<const uint8_t> choices,     \
                                        absl::Span<T> output, int bw,          \
                                        int num_level) {                       \
    impl_->RecvCorrelatedMsgChosenChoice_Collapse<T>(choices, output, bw,      \
                                                     num_level);               \
  }                                                                            \
  void YaclFerretOt::SendRMRC(absl::Span<T> output0, absl::Span<T> output1,    \
                              size_t bit_width) {                              \
    bit_width = CheckBitWidth<T>(bit_width);                                   \
    impl_->SendRandMsgRandChoice<T>(output0, output1, bit_width);              \
  }                                                                            \
  void YaclFerretOt::RecvRMRC(absl::Span<uint8_t> choices,                     \
                              absl::Span<T> output, size_t bit_width) {        \
    bit_width = CheckBitWidth<T>(bit_width);                                   \
    impl_->RecvRandMsgRandChoice<T>(choices, output, bit_width);               \
  }                                                                            \
  void YaclFerretOt::SendCMCC(absl::Span<const T> msg_array, size_t N,         \
                              size_t bit_width) {                              \
    bit_width = CheckBitWidth<T>(bit_width);                                   \
    impl_->SendChosenMsgChosenChoice<T>(msg_array, N, bit_width);              \
  }                                                                            \
  void YaclFerretOt::RecvCMCC(absl::Span<const uint8_t> choices, size_t N,     \
                              absl::Span<T> output, size_t bit_width) {        \
    bit_width = CheckBitWidth<T>(bit_width);                                   \
    impl_->RecvChosenMsgChosenChoice<T>(choices, N, output, bit_width);        \
  }                                                                            \
  void YaclFerretOt::SendRMCC(absl::Span<T> output0, absl::Span<T> output1,    \
                              size_t bit_width) {                              \
    bit_width = CheckBitWidth<T>(bit_width);                                   \
    impl_->SendRMCC<T>(output0, output1, bit_width);                           \
  }                                                                            \
  void YaclFerretOt::RecvRMCC(absl::Span<const uint8_t> choices,               \
                              absl::Span<T> output, size_t bit_width) {        \
    bit_width = CheckBitWidth<T>(bit_width);                                   \
    impl_->RecvRMCC<T>(choices, output, bit_width);                            \
  }

DEF_SEND_RECV(uint8_t)
DEF_SEND_RECV(uint32_t)
DEF_SEND_RECV(uint64_t)
DEF_SEND_RECV(uint128_t)

#undef DEF_SEND_RECV
}  // namespace spu::mpc::cheetah
