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

#include "libspu/mpc/cheetah/ot/ferret.h"

#include <utility>

#include "emp-ot/cot.h"
#include "emp-ot/ferret/ferret_cot.h"
#include "emp-tool/io/io_channel.h"
#include "spdlog/spdlog.h"
#include "yacl/base/buffer.h"
#include "yacl/link/link.h"

#include "libspu/mpc/cheetah/ot/mitccrh_exp.h"
#include "libspu/mpc/cheetah/ot/util.h"

#define PRE_OT_DATA_REG_SEND_FILE_ALICE "pre_ferret_data_reg_send_alice"
#define PRE_OT_DATA_REG_SEND_FILE_BOB "pre_ferret_data_reg_send_bob"
#define PRE_OT_DATA_REG_RECV_FILE_ALICE "pre_ferret_data_reg_recv_alice"
#define PRE_OT_DATA_REG_RECV_FILE_BOB "pre_ferret_data_reg_recv_bob"

namespace spu::mpc::cheetah {
constexpr size_t kOTBatchSize = emp::ot_bsize;  // emp-ot/cot.h
using OtBaseTyp = emp::block;

// A concrete class for emp::IOChannel
// Because emp::FerretCOT needs a uniform API of emp::IOChannel
class CheetahIO : public emp::IOChannel<CheetahIO> {
 public:
  std::shared_ptr<Communicator> conn_;

  constexpr static uint64_t SEND_BUFFER_SIZE = 1024 * 1024;

  uint32_t send_op_;
  uint32_t recv_op_;

  std::vector<uint8_t> send_buffer_;
  uint64_t send_buffer_used_;

  std::vector<uint8_t> recv_buffer_;
  uint64_t recv_buffer_used_;

  explicit CheetahIO(std::shared_ptr<Communicator> conn)
      : conn_(std::move(conn)),
        send_op_(0),
        recv_op_(0),
        send_buffer_used_(0),
        recv_buffer_used_(0) {
    send_buffer_.resize(SEND_BUFFER_SIZE);
  }

  ~CheetahIO() {
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
        fmt::format("CheetahIO send:{}", send_op_++));

    std::memset(send_buffer_.data(), 0, SEND_BUFFER_SIZE);
    send_buffer_used_ = 0;
  }

  void fill_recv() {
    recv_buffer_ = conn_->recv<uint8_t>(
        conn_->nextRank(), fmt::format("CheetahIO recv:{}", recv_op_++));
    recv_buffer_used_ = 0;
  }

  void send_data_internal(const void* data, int len) {
    size_t send_buffer_left = SEND_BUFFER_SIZE - send_buffer_used_;
    if (send_buffer_left <= static_cast<size_t>(len)) {
      std::memcpy(send_buffer_.data() + send_buffer_used_, data,
                  send_buffer_left);
      send_buffer_used_ += send_buffer_left;
      flush();

      send_data_internal(static_cast<const char*>(data) + send_buffer_left,
                         len - send_buffer_left);
    } else {
      std::memcpy(send_buffer_.data() + send_buffer_used_, data, len);
      send_buffer_used_ += len;
    }
  }

  void recv_data_internal(void* data, int len) {
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

      recv_data_internal(static_cast<char*>(data) + recv_buffer_left,
                         len - recv_buffer_left);
    }
  }

  template <typename T>
  void send_data_partial(const T* data, int len, int bitlength) {
    if (bitlength == sizeof(T) * 8) {
      send_data_internal(static_cast<const void*>(data), len * sizeof(T));
      return;
    }

    int compact_len = (bitlength + 7) / 8;
    std::vector<uint8_t> bytes(len);
    for (int i = 0; i < compact_len; i++) {
      for (int j = 0; j < len; j++) {
        bytes[j] = uint8_t(data[j] >> (i * 8));
      }
      send_data_internal(bytes.data(), len);
    }
  }

  template <typename T>
  void recv_data_partial(T* data, int len, int bitlength) {
    if (bitlength == sizeof(T) * 8) {
      recv_data_internal(static_cast<void*>(data), len * sizeof(T));
      return;
    }
    std::memset(data, 0, len * sizeof(T));

    int compact_len = (bitlength + 7) / 8;
    std::vector<uint8_t> bytes(len);
    for (int i = 0; i < compact_len; i++) {
      recv_data_internal(bytes.data(), len);
      for (int j = 0; j < len; j++) {
        data[j] |= T(bytes[j]) << (i * 8);
      }
    }
    T mask = (T(1) << bitlength) - 1;
    for (int i = 0; i < len; i++) {
      data[i] &= mask;
    }
  }
};

struct FerretOT::Impl {
 private:
  const bool is_sender_;

  std::shared_ptr<CheetahIO> io_{nullptr};
  std::array<CheetahIO*, 1> io_holder_;
  std::shared_ptr<emp::FerretCOT<CheetahIO>> ferret_{nullptr};

  MITCCRHExp<8> mitccrh_exp_{};

  void RandCOT(absl::Span<OtBaseTyp> output) {
    ferret_->rcot(output.data(), output.size());
  }

  void SendCOT(OtBaseTyp* output, size_t n) {
    SPU_ENFORCE(is_sender_);
    ferret_->send_cot(output, n);
  }

  void RecvCOT(absl::Span<const uint8_t> choices,
               absl::Span<OtBaseTyp> output) {
    SPU_ENFORCE(not is_sender_);
    ferret_->recv_cot(output.data(),
                      reinterpret_cast<const bool*>(choices.data()),
                      output.size());
  }

 public:
  Impl(std::shared_ptr<Communicator> conn, bool is_sender)
      : is_sender_(is_sender) {
    SPU_ENFORCE(conn != nullptr);
    constexpr int thread = 1;
    constexpr bool malicious = false;
    constexpr bool run_setup = true;
    int role = is_sender ? emp::ALICE : emp::BOB;
    io_ = std::make_shared<CheetahIO>(conn);
    io_holder_[0] = io_.get();
    std::string save_file;
    if (conn->getRank() == 0) {
      if (is_sender) {
        save_file = PRE_OT_DATA_REG_SEND_FILE_ALICE;
      } else {
        save_file = PRE_OT_DATA_REG_RECV_FILE_ALICE;
      }
    } else {
      if (is_sender) {
        save_file = PRE_OT_DATA_REG_SEND_FILE_BOB;
      } else {
        save_file = PRE_OT_DATA_REG_RECV_FILE_BOB;
      }
    }

    ferret_ = std::make_shared<emp::FerretCOT<CheetahIO>>(
        role, thread, io_holder_.data(), malicious, run_setup);

    OtBaseTyp seed;
    if (is_sender_) {
      ferret_->prg.random_block(&seed, 1);
      io_->send_block(&seed, 1);
      io_->flush();
      ferret_->mitccrh.setS(seed);
    } else {
      io_->recv_block(&seed, 1);
      ferret_->mitccrh.setS(seed);
    }
  }

  ~Impl() = default;

  int Rank() const { return io_->conn_->getRank(); }

  void Flush() {
    if (io_) {
      io_->flush();
    }
  }

  void SendRandCorrelatedMsgChosenChoice(OtBaseTyp* output, size_t n) {
    SPU_ENFORCE(n > 0 && output != nullptr);
    SendCOT(output, n);
  }

  void RecvRandCorrelatedMsgChosenChoice(absl::Span<const uint8_t> choices,
                                         absl::Span<OtBaseTyp> output) {
    SPU_ENFORCE_EQ(choices.size(), output.size());
    SPU_ENFORCE(!output.empty());
    RecvCOT(choices, output);
  }

  void SendRandMsgChosenChoice(OtBaseTyp* msg0, OtBaseTyp* msg1, size_t n) {
    SendRandCorrelatedMsgChosenChoice(msg0, n);

    std::array<OtBaseTyp, 2 * kOTBatchSize> pad;

    for (size_t i = 0; i < n; i += kOTBatchSize) {
      size_t this_batch = std::min(kOTBatchSize, n - i);
      for (size_t j = 0; j < this_batch; ++j) {
        pad[2 * j] = msg0[i + j];
        pad[2 * j + 1] = msg0[i + j] ^ ferret_->Delta;
      }

      ferret_->mitccrh.template hash<kOTBatchSize, 2>(pad.data());

      for (size_t j = 0; j < this_batch; ++j) {
        msg0[i + j] = pad[2 * j];
        msg1[i + j] = pad[2 * j + 1];
      }
    }
  }

  void RecvRandMsgChosenChoice(absl::Span<const uint8_t> choices,
                               absl::Span<OtBaseTyp> output) {
    size_t n = output.size();
    SPU_ENFORCE(n > 0);
    SPU_ENFORCE_EQ(choices.size(), n);
    RecvRandCorrelatedMsgChosenChoice(choices, output);

    std::array<OtBaseTyp, kOTBatchSize> pad;
    for (size_t i = 0; i < n; i += kOTBatchSize) {
      size_t this_batch = std::min(kOTBatchSize, n - i);
      size_t nbytes = this_batch * sizeof(OtBaseTyp);

      std::memcpy(pad.data(), output.data() + i, nbytes);
      ferret_->mitccrh.template hash<kOTBatchSize, 1>(pad.data());
      std::memcpy(output.data() + i, pad.data(), nbytes);
    }
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

    std::vector<OtBaseTyp> rcm_output(n);
    RecvRandCorrelatedMsgChosenChoice(choices, absl::MakeSpan(rcm_output));

    size_t pack_load = 8 * sizeof(T) / bit_width;
    std::array<OtBaseTyp, kOTBatchSize> pad;
    std::vector<T> corr_output(kOTBatchSize);
    std::vector<T> packed_corr_output;
    if (pack_load > 1) {
      packed_corr_output.resize(CeilDiv(corr_output.size(), pack_load));
    }

    for (size_t i = 0; i < n; i += kOTBatchSize) {
      size_t this_batch = std::min(kOTBatchSize, n - i);

      std::memcpy(pad.data(), rcm_output.data() + i,
                  this_batch * sizeof(OtBaseTyp));
      ferret_->mitccrh.template hash<kOTBatchSize, 1>(pad.data());

      if (pack_load > 1) {
        size_t used = CeilDiv(this_batch, pack_load);
        io_->recv_data(packed_corr_output.data(), sizeof(T) * used);
        UnzipArray<T>({packed_corr_output.data(), used}, bit_width,
                      {corr_output.data(), this_batch});
      } else {
        io_->recv_data(corr_output.data(), sizeof(T) * this_batch);
      }

      for (size_t j = 0; j < this_batch; ++j) {
        output[i + j] = ConvFromBlock<T>(pad[j]);
        if (choices[i + j]) {
          output[i + j] = corr_output[j] - output[i + j];
        }
      }
    }
  }

  void RecvRandMsgRandChoice(absl::Span<uint8_t> choices,
                             absl::Span<OtBaseTyp> output) {
    size_t n = choices.size();
    SPU_ENFORCE(n > 0);
    SPU_ENFORCE_EQ(n, output.size());
    RandCOT(output);

    for (size_t i = 0; i < n; ++i) {
      choices[i] = emp::getLSB(output[i]);
    }

    std::array<OtBaseTyp, kOTBatchSize> pad;
    for (size_t i = 0; i < n; i += kOTBatchSize) {
      size_t sze = std::min(kOTBatchSize, n - i) * sizeof(OtBaseTyp);
      std::memcpy(pad.data(), output.data() + i, sze);
      ferret_->mitccrh.template hash<kOTBatchSize, 1>(pad.data());
      std::memcpy(output.data() + i, pad.data(), sze);
    }
  }

  void SendRandMsgRandChoice(absl::Span<OtBaseTyp> output0,
                             absl::Span<OtBaseTyp> output1) {
    size_t n = output0.size();
    SPU_ENFORCE(n > 0);
    SPU_ENFORCE_EQ(n, output1.size());
    RandCOT(output0);

    std::array<OtBaseTyp, 2 * kOTBatchSize> pad;
    for (size_t i = 0; i < n; i += kOTBatchSize) {
      size_t this_batch = std::min(kOTBatchSize, n - i);

      for (size_t j = 0; j < this_batch; ++j) {
        pad[2 * j] = output0[i + j];
        pad[2 * j + 1] = output0[i + j] ^ ferret_->Delta;
      }
      ferret_->mitccrh.template hash<kOTBatchSize, 2>(pad.data());
      for (size_t j = 0; j < this_batch; ++j) {
        output0[i + j] = pad[2 * j];
        output1[i + j] = pad[2 * j + 1];
      }
    }
  }

  void SendChosenMsgChosenChoice(const OtBaseTyp* msg0, const OtBaseTyp* msg1,
                                 size_t n) {
    SPU_ENFORCE(msg0 != nullptr && msg1 != nullptr);
    SPU_ENFORCE(n > 0);
    std::unique_ptr<OtBaseTyp[]> rcm_data(new OtBaseTyp[n]);
    SendRandCorrelatedMsgChosenChoice(rcm_data.get(), n);

    std::array<OtBaseTyp, 2 * kOTBatchSize> pad;
    for (size_t i = 0; i < n; i += kOTBatchSize) {
      size_t this_batch = std::min(kOTBatchSize, n - i);

      for (size_t j = 0; j < this_batch; ++j) {
        pad[2 * j] = rcm_data[i + j];
        pad[2 * j + 1] = rcm_data[i + j] ^ ferret_->Delta;
      }

      ferret_->mitccrh.template hash<kOTBatchSize, 2>(pad.data());
      for (size_t j = 0; j < this_batch; ++j) {
        pad[2 * j] ^= msg0[i + j];
        pad[2 * j + 1] ^= msg1[i + j];
      }

      io_->send_data(pad.data(), 2 * sizeof(OtBaseTyp) * this_batch);
    }
  }

  void RecvChosenMsgChosenChoice(absl::Span<const uint8_t> choices,
                                 absl::Span<OtBaseTyp> output) {
    size_t n = choices.size();
    SPU_ENFORCE(n > 0);
    SPU_ENFORCE_EQ(n, output.size());
    RecvRandCorrelatedMsgChosenChoice(choices, output);

    std::array<OtBaseTyp, kOTBatchSize> pad;
    std::array<OtBaseTyp, 2 * kOTBatchSize> recv;
    for (size_t i = 0; i < n; i += kOTBatchSize) {
      size_t this_batch = std::min(kOTBatchSize, n - i);
      size_t nbytes = this_batch * sizeof(OtBaseTyp);
      std::memcpy(pad.data(), output.data() + i, nbytes);
      ferret_->mitccrh.template hash<kOTBatchSize, 1>(pad.data());
      io_->recv_data(recv.data(), 2 * nbytes);
      for (size_t j = 0; j < this_batch; ++j) {
        output[i + j] = recv[2 * j + (choices[i + j] & 1)] ^ pad[j];
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
    std::unique_ptr<OtBaseTyp[]> rcm_output(new OtBaseTyp[n]);

    SendRandCorrelatedMsgChosenChoice(rcm_output.get(), n);

    size_t pack_load = 8 * sizeof(T) / bit_width;
    std::array<OtBaseTyp, 2 * kOTBatchSize> pad;
    std::vector<T> corr_output(kOTBatchSize);
    std::vector<T> packed_corr_output;
    if (pack_load > 1) {
      packed_corr_output.resize(CeilDiv(corr_output.size(), pack_load));
    }

    for (size_t i = 0; i < n; i += kOTBatchSize) {
      size_t this_batch = std::min(kOTBatchSize, n - i);
      for (size_t j = 0; j < this_batch; ++j) {
        pad[2 * j] = rcm_output[i + j];
        pad[2 * j + 1] = rcm_output[i + j] ^ ferret_->Delta;
      }

      ferret_->mitccrh.template hash<kOTBatchSize, 2>(pad.data());

      for (size_t j = 0; j < this_batch; ++j) {
        output[i + j] = ConvFromBlock<T>(pad[2 * j]);
        corr_output[j] = ConvFromBlock<T>(pad[2 * j + 1]);
        corr_output[j] += corr[i + j] + output[i + j];
      }

      if (pack_load > 1) {
        size_t used = ZipArray<T>({corr_output.data(), this_batch}, bit_width,
                                  absl::MakeSpan(packed_corr_output));
        SPU_ENFORCE(used == CeilDiv(this_batch, pack_load));
        io_->send_data(packed_corr_output.data(), used * sizeof(T));
      } else {
        io_->send_data(corr_output.data(), sizeof(T) * this_batch);
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

    std::vector<OtBaseTyp> rm_data(2 * n);
    auto* rm_data0 = rm_data.data();
    auto* rm_data1 = rm_data.data() + n;
    SendRandMsgRandChoice({rm_data0, n}, {rm_data1, n});

    std::transform(rm_data0, rm_data0 + n, output0.data(), [mask](OtBaseTyp x) {
      return static_cast<T>(_mm_extract_epi64(x, 0)) & mask;
    });
    std::transform(rm_data1, rm_data1 + n, output1.data(), [mask](OtBaseTyp x) {
      return static_cast<T>(_mm_extract_epi64(x, 0)) & mask;
    });
  }

  template <typename T>
  void RecvRandMsgRandChoice(absl::Span<uint8_t> choices, absl::Span<T> output,
                             size_t bit_width) {
    size_t n = choices.size();
    SPU_ENFORCE(n > 0);
    SPU_ENFORCE_EQ(n, output.size());
    const T mask = makeBitsMask<T>(bit_width);

    std::vector<OtBaseTyp> rm_data(n);
    RecvRandMsgRandChoice(choices, absl::MakeSpan(rm_data));

    std::transform(rm_data.cbegin(), rm_data.cend(), output.data(),
                   [mask](OtBaseTyp x) {
                     return static_cast<T>(_mm_extract_epi64(x, 0)) & mask;
                   });
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
    std::unique_ptr<OtBaseTyp[]> rm_data0(new OtBaseTyp[n * logN]);
    std::unique_ptr<OtBaseTyp[]> rm_data1(new OtBaseTyp[n * logN]);

    SendRandMsgChosenChoice(rm_data0.get(), rm_data1.get(), n * logN);

    std::vector<OtBaseTyp> hash_in0(N - 1);
    std::vector<OtBaseTyp> hash_in1(N - 1);
    {
      size_t idx = 0;
      for (size_t x = 0; x < logN; ++x) {
        for (size_t y = 0; y < (1UL << x); ++y) {
          hash_in0.at(idx) = emp::makeBlock(y, 0);
          hash_in1.at(idx) = emp::makeBlock((1 << x) + y, 0);
          ++idx;
        }
      }
    }

    std::vector<OtBaseTyp> hash_out0(N - 1);
    std::vector<OtBaseTyp> hash_out1(N - 1);
    std::vector<OtBaseTyp> pad(kOTBatchSize * N);

    const T msg_mask = makeBitsMask<T>(bit_width);
    size_t pack_load = 8 * sizeof(T) / bit_width;
    std::vector<T> to_send(kOTBatchSize * N);
    std::vector<T> packed_to_send;
    if (pack_load > 1) {
      // NOTE: pack bit chunks into single T element if possible
      packed_to_send.resize(CeilDiv(to_send.size(), pack_load));
    }

    for (size_t i = 0; i < n; i += kOTBatchSize) {
      size_t this_batch = std::min(kOTBatchSize, n - i);
      std::memset(pad.data(), 0, pad.size() * sizeof(OtBaseTyp));

      for (size_t j = 0; j < this_batch; ++j) {
        mitccrh_exp_.renew_ks(&rm_data0[(i + j) * logN], logN);
        mitccrh_exp_.hash_exp(hash_out0.data(), hash_in0.data(), logN);

        mitccrh_exp_.renew_ks(&rm_data1[(i + j) * logN], logN);
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

      // The i-th messages start msg_array[i * N] to msg_array[i * N + N - 1]
      for (size_t j = 0; j < this_batch; ++j) {
        const auto* this_msg = msg_array.data() + (i + j) * N;
        for (size_t k = 0; k < N; ++k) {
          to_send[j * N + k] = (ConvFromBlock<T>(pad[j * N + k]) ^ this_msg[k]);
          to_send[j * N + k] &= msg_mask;
        }
      }

      if (pack_load > 1) {
        size_t used = ZipArray<T>({to_send.data(), N * this_batch}, bit_width,
                                  absl::MakeSpan(packed_to_send));
        SPU_ENFORCE(used == CeilDiv(N * this_batch, pack_load));
        io_->send_data(packed_to_send.data(), used * sizeof(T));
      } else {
        io_->send_data(to_send.data(), N * this_batch * sizeof(T));
      }
    }
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

    // rm_data[logN * i + k] = 1-of-2 OT on the k-th bits of the i-th message
    std::vector<OtBaseTyp> rm_data(n * logN);
    RecvRandMsgChosenChoice(absl::MakeSpan(bool_choices),
                            absl::MakeSpan(rm_data));

    std::vector<OtBaseTyp> hash_in(logN);
    std::vector<OtBaseTyp> hash_out(logN);
    std::vector<OtBaseTyp> pad(kOTBatchSize);

    const T msg_mask = makeBitsMask<T>(bit_width);
    const size_t pack_load = 8 * sizeof(T) / bit_width;
    std::vector<T> recv(kOTBatchSize * N);
    std::vector<T> packed_recv(CeilDiv(recv.size(), pack_load));

    for (size_t i = 0; i < n; i += kOTBatchSize) {
      size_t this_batch = std::min(kOTBatchSize, n - i);
      size_t used = CeilDiv(N * this_batch, pack_load);
      io_->recv_data(packed_recv.data(), used * sizeof(T));
      UnzipArray<T>({packed_recv.data(), used}, bit_width,
                    {recv.data(), N * this_batch});

      std::memset(pad.data(), 0, kOTBatchSize * sizeof(OtBaseTyp));
      for (size_t j = 0; j < this_batch; ++j) {
        for (size_t s = 0; s < logN; s++) {
          auto h = choices[i + j] & makeBitsMask<uint8_t>(1 + s);
          hash_in[s] = emp::makeBlock(h, 0);
        }
        mitccrh_exp_.renew_ks(&rm_data[(i + j) * logN], logN);
        mitccrh_exp_.hash_single(hash_out.data(), hash_in.data(), logN);

        pad[j] = std::accumulate(hash_out.begin(), hash_out.end(), pad[j],
                                 std::bit_xor<OtBaseTyp>());
      }

      for (size_t j = 0; j < this_batch; ++j) {
        output[i + j] =
            (ConvFromBlock<T>(pad[j]) ^ recv[N * j + choices[i + j]]) &
            msg_mask;
      }
    }
  }
};

FerretOT::FerretOT(std::shared_ptr<Communicator> conn, bool is_sender) {
  impl_ = std::make_shared<Impl>(conn, is_sender);
}

int FerretOT::Rank() const { return impl_->Rank(); }

void FerretOT::Flush() { impl_->Flush(); }

FerretOT::~FerretOT() { impl_->Flush(); }

template <typename T>
size_t CheckBitWidth(size_t bw) {
  size_t m = 8 * sizeof(T);
  SPU_ENFORCE(bw <= m);
  if (bw == 0) {
    bw = m;
  }
  return bw;
}

#define DEF_SEND_RECV(T)                                                     \
  void FerretOT::SendCAMCC(absl::Span<const T> corr, absl::Span<T> output,   \
                           int bw) {                                         \
    impl_->SendCorrelatedMsgChosenChoice<T>(corr, output, bw);               \
  }                                                                          \
  void FerretOT::RecvCAMCC(absl::Span<const uint8_t> choices,                \
                           absl::Span<T> output, int bw) {                   \
    impl_->RecvCorrelatedMsgChosenChoice<T>(choices, output, bw);            \
  }                                                                          \
  void FerretOT::SendRMRC(absl::Span<T> output0, absl::Span<T> output1,      \
                          size_t bit_width) {                                \
    bit_width = CheckBitWidth<T>(bit_width);                                 \
    impl_->SendRandMsgRandChoice<T>(output0, output1, bit_width);            \
  }                                                                          \
  void FerretOT::RecvRMRC(absl::Span<uint8_t> choices, absl::Span<T> output, \
                          size_t bit_width) {                                \
    bit_width = CheckBitWidth<T>(bit_width);                                 \
    impl_->RecvRandMsgRandChoice<T>(choices, output, bit_width);             \
  }                                                                          \
  void FerretOT::SendCMCC(absl::Span<const T> msg_array, size_t N,           \
                          size_t bit_width) {                                \
    bit_width = CheckBitWidth<T>(bit_width);                                 \
    impl_->SendChosenMsgChosenChoice<T>(msg_array, N, bit_width);            \
  }                                                                          \
  void FerretOT::RecvCMCC(absl::Span<const uint8_t> choices, size_t N,       \
                          absl::Span<T> output, size_t bit_width) {          \
    bit_width = CheckBitWidth<T>(bit_width);                                 \
    impl_->RecvChosenMsgChosenChoice<T>(choices, N, output, bit_width);      \
  }

DEF_SEND_RECV(uint8_t)
DEF_SEND_RECV(uint32_t)
DEF_SEND_RECV(uint64_t)
DEF_SEND_RECV(uint128_t)

#undef DEF_SEND_RECV
}  // namespace spu::mpc::cheetah
