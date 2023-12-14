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

#pragma once

#include "yacl/base/dynamic_bitset.h"
#include "yacl/crypto/primitives/ot/base_ot.h"
#include "yacl/crypto/primitives/ot/ferret_ote.h"
#include "yacl/crypto/primitives/ot/iknp_ote.h"
#include "yacl/crypto/primitives/ot/ot_store.h"
#include "yacl/crypto/utils/rand.h"

#include "libspu/core/prelude.h"
#include "libspu/mpc/cheetah/ot/ot_util.h"
#include "libspu/mpc/cheetah/ot/yacl/yacl_util.h"

namespace spu::mpc::cheetah {

namespace yc = yacl::crypto;
namespace yl = yacl::link;

class YaclOTeAdapter {
 public:
  YaclOTeAdapter() = default;
  virtual ~YaclOTeAdapter() = default;
  virtual void send_rcot(absl::Span<uint128_t> data) = 0;
  virtual void recv_rcot(absl::Span<uint128_t> data,
                         absl::Span<uint8_t> choices) = 0;
  virtual void send_cot(absl::Span<uint128_t> data) = 0;
  virtual void recv_cot(absl::Span<uint128_t> data,
                        absl::Span<const uint8_t> choices) = 0;
  virtual void OneTimeSetup() = 0;

  uint128_t Delta{0};
  virtual uint128_t GetDelta() const { return Delta; }
};

class YaclFerretOTeAdapter : public YaclOTeAdapter {
 public:
  YaclFerretOTeAdapter(const std::shared_ptr<yl::Context> ctx, bool is_sender) {
    ctx_ = ctx->Spawn();  // Spawn link
    is_sender_ = is_sender;
    reserve_num_ = yc::FerretCotHelper(lpn_param_, 0);

    ot_buff_.resize(lpn_param_.n);

    id_ = yacl_id_;
    ++yacl_id_;
  }

  ~YaclFerretOTeAdapter() {
    SPDLOG_DEBUG(
        "[FerretAdapter {}]({}), comsume OT {}, total time {:.3e} ms, "
        "invoke bootstrap {} ( {:.2e} ms per bootstrap, {:.2e} ms per ot )",
        id_, (is_sender_ ? fmt::format("Sender") : fmt::format("Receiver")),
        consumed_ot_num_, bootstrap_time_, bootstrap_num_,
        bootstrap_time_ / bootstrap_num_, bootstrap_time_ / consumed_ot_num_);
  }

  void OneTimeSetup() override;

  void recv_cot(absl::Span<uint128_t> data,
                absl::Span<const uint8_t> choices) override {
    recv_cot(data, VecU8toBitset(choices));
  }

  void send_rcot(absl::Span<uint128_t> data) override { rcot(data); }

  void recv_rcot(absl::Span<uint128_t> data,
                 absl::Span<uint8_t> choices) override {
    rcot(data);
    std::transform(data.data(), data.data() + data.size(), choices.data(),
                   [](uint128_t v) { return (uint8_t)v & 0x1; });
  }

  // Correlated Cot with Chosen Choices
  void send_cot(absl::Span<uint128_t> data) override;

  void recv_cot(absl::Span<uint128_t> data,
                const yacl::dynamic_bitset<uint128_t>& choices);

  // YACL FERRET OTE ENTRY
  void rcot(absl::Span<uint128_t> data);

  uint128_t GetDelta() const override { return Delta; }

  uint128_t GetConsumed() const { return consumed_ot_num_; }

  double GetTime() const { return bootstrap_time_; }

 private:
  std::shared_ptr<yl::Context> ctx_{nullptr};

  bool is_sender_{false};

  bool is_setup_{false};

  yc::LpnParam pre_lpn_param_{470016, 32768, 918,
                              yc::LpnNoiseAsm::RegularNoise};

  yc::LpnParam lpn_param_{10485760, 452000, 1280,
                          yc::LpnNoiseAsm::RegularNoise};

  static constexpr uint128_t one =
      yacl::MakeUint128(0xffffffffffffffff, 0xfffffffffffffffe);

  uint64_t reserve_num_{0};

  uint64_t buff_used_num_{0};

  uint64_t buff_upper_bound_{0};

  yacl::AlignedVector<uint128_t> ot_buff_;  // ot buffer

  // Yacl Ferret OTe
  void Bootstrap();

  // Yacl Ferret OTe
  void BootstrapInplace(absl::Span<uint128_t> ot, absl::Span<uint128_t> data);

  // runtime record
  uint128_t consumed_ot_num_{0};
  uint128_t bootstrap_num_{0};  // number of invoke bootstrap
  double bootstrap_time_{0.0};  // ms
  uint128_t id_{0};
  static uint128_t yacl_id_;
};

class YaclIknpOTeAdapter : public YaclOTeAdapter {
 public:
  YaclIknpOTeAdapter(const std::shared_ptr<yl::Context> ctx, bool is_sender) {
    ctx_ = ctx->Spawn();  // Spawn link
    is_sender_ = is_sender;
    id_ = yacl_id_;
    ++yacl_id_;
  }

  ~YaclIknpOTeAdapter() {
    SPDLOG_INFO(
        "[IknpAdapter {}]({}), comsume OT {}, total time {:.3e} ms,"
        "invoke IKNP-OTe {} ( {:.2e} ms per iknp , {:.2e} ms per ot )",
        id_, (is_sender_ ? fmt::format("Sender") : fmt::format("Receiver")),
        consumed_ot_num_, ote_time_, ote_num_, ote_time_ / ote_num_,
        ote_time_ / consumed_ot_num_);
  }

  void OneTimeSetup() override;

  void recv_cot(absl::Span<uint128_t> data,
                absl::Span<const uint8_t> choices) override {
    recv_cot(data, VecU8toBitset(choices));
  }

  inline void send_rcot(absl::Span<uint128_t> data) override { send_cot(data); }

  inline void recv_rcot(absl::Span<uint128_t> data,
                        absl::Span<uint8_t> choices) override {
    auto _choices =
        yc::RandBits<yacl::dynamic_bitset<uint128_t>>(data.size(), true);
    BitsettoVecU8(_choices, choices);

    recv_cot(data, _choices);
  }

  // IKNP ENTRY
  // Correlated Cot with Chosen Choices
  void send_cot(absl::Span<uint128_t> data) override {
    YACL_ENFORCE(is_sender_);
    auto begin = std::chrono::high_resolution_clock::now();

    // [Warning] copy, low efficiency
    std::vector<std::array<uint128_t, 2>> send_blocks(data.size());
    yc::IknpOtExtSend(ctx_, *recv_ot_ptr_, absl::MakeSpan(send_blocks), true);
    for (uint64_t i = 0; i < data.size(); ++i) {
      data[i] = send_blocks[i][0];
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto elapse =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - begin)
            .count();
    ote_time_ += elapse * 1000;
    consumed_ot_num_ += data.size();
    ++ote_num_;
  }

  void recv_cot(absl::Span<uint128_t> data,
                const yacl::dynamic_bitset<uint128_t>& choices) {
    YACL_ENFORCE(is_sender_ == false);
    auto begin = std::chrono::high_resolution_clock::now();
    yc::IknpOtExtRecv(ctx_, *send_ot_ptr_, choices, absl::MakeSpan(data), true);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapse =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - begin)
            .count();
    ote_time_ += elapse * 1000;
    consumed_ot_num_ += data.size();
    ++ote_num_;
  }

  uint128_t GetDelta() const override { return Delta; }

  uint128_t GetConsumed() const { return consumed_ot_num_; }

  double GetTime() const { return ote_time_; }

 private:
  std::shared_ptr<yl::Context> ctx_{nullptr};

  bool is_sender_{false};

  bool is_setup_{false};

  std::unique_ptr<yc::OtSendStore> send_ot_ptr_{nullptr};

  std::unique_ptr<yc::OtRecvStore> recv_ot_ptr_{nullptr};

  // runtime record
  uint128_t consumed_ot_num_{0};
  uint128_t ote_num_{0};  // number of invoke ote protocol
  double ote_time_{0.0};  // ms
  uint128_t id_{0};
  static uint128_t yacl_id_;
};

}  // namespace spu::mpc::cheetah
