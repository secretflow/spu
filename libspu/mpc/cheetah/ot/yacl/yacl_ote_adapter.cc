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

#include "libspu/mpc/cheetah/ot/yacl/yacl_ote_adapter.h"

namespace spu::mpc::cheetah {

namespace yc = yacl::crypto;
namespace yl = yacl::link;

// ------------------------
//   FerretOTeAdapter
// ------------------------

uint128_t YaclFerretOTeAdapter::yacl_id_ = 0;

void YaclFerretOTeAdapter::OneTimeSetup() {
  if (is_setup_) {
    return;
  }

  uint128_t pre_lpn_num_ = yc::FerretCotHelper(pre_lpn_param_, 0);
  // Sender
  if (is_sender_) {
    auto ss_sender = yc::SoftspokenOtExtSender();
    ss_sender.OneTimeSetup(ctx_);

    auto ss_send_blocks =
        yacl::AlignedVector<std::array<uint128_t, 2>>(pre_lpn_num_, {0, 0});
    ss_sender.Send(ctx_, absl::MakeSpan(ss_send_blocks), true);

    auto ss_send_block0 = yacl::AlignedVector<uint128_t>(pre_lpn_num_, 0);
    std::transform(
        ss_send_blocks.cbegin(), ss_send_blocks.cend(), ss_send_block0.begin(),
        [&one = std::as_const(one)](const std::array<uint128_t, 2>& blocks) {
          return blocks[0] & one;
        });

    Delta = ss_sender.GetDelta() | ~one;
    auto pre_ferret_sent_ot =
        yc::MakeCompactOtSendStore(std::move(ss_send_block0), Delta);

    // pre ferret OTe
    yc::FerretOtExtSend_cheetah(
        ctx_, pre_ferret_sent_ot, pre_lpn_param_, pre_lpn_param_.n,
        absl::MakeSpan(ot_buff_.data<uint128_t>(), pre_lpn_param_.n));
  }
  // Receiver
  else {
    auto ss_receiver = yc::SoftspokenOtExtReceiver();
    ss_receiver.OneTimeSetup(ctx_);

    auto ss_choices =
        yc::RandBits<yacl::dynamic_bitset<uint128_t>>(pre_lpn_num_, true);
    auto ss_recv_blocks = yacl::AlignedVector<uint128_t>(pre_lpn_num_, 0);

    ss_receiver.Recv(ctx_, ss_choices, absl::MakeSpan(ss_recv_blocks), true);

    for (uint64_t i = 0; i < pre_lpn_num_; ++i) {
      ss_recv_blocks[i] = (ss_recv_blocks[i] & one) | ss_choices[i];
    }

    yc::OtRecvStore pre_ferret_recv_ot =
        yc::MakeCompactOtRecvStore(std::move(ss_recv_blocks));

    // pre ferret OTe
    yc::FerretOtExtRecv_cheetah(
        ctx_, pre_ferret_recv_ot, pre_lpn_param_, pre_lpn_param_.n,
        absl::MakeSpan(ot_buff_.data<uint128_t>(), pre_lpn_param_.n));
  }
  is_setup_ = true;
  buff_used_num_ = reserve_num_;
  buff_upper_bound_ = pre_lpn_param_.n;
  // Delay boostrap
  // Bootstrap();
}

void YaclFerretOTeAdapter::rcot(absl::Span<uint128_t> data) {
  if (is_setup_ == false) {
    OneTimeSetup();
  }

  uint64_t data_offset = 0;
  uint64_t require_num = data.size();
  uint64_t remain_num = buff_upper_bound_ - buff_used_num_;
  // When require_num is greater than lpn_param.n
  // call FerretOTe with data's subspan to avoid memory copy
  {
    uint32_t bootstrap_inplace_counter = 0;
    absl::Span<uint128_t> ot_span =
        absl::MakeSpan(ot_buff_.data<uint128_t>(), reserve_num_);
    while (require_num >= lpn_param_.n) {
      // avoid memory copy
      BootstrapInplace(ot_span, data.subspan(data_offset, lpn_param_.n));

      data_offset += (lpn_param_.n - reserve_num_);
      require_num -= (lpn_param_.n - reserve_num_);
      ++bootstrap_inplace_counter;
      consumed_ot_num_ += lpn_param_.n;
      // Next Round
      ot_span = data.subspan(data_offset, reserve_num_);
    }
    if (bootstrap_inplace_counter != 0) {
      std::memcpy(reinterpret_cast<uint128_t*>(ot_buff_.data<uint128_t>()),
                  ot_span.data(), reserve_num_ * sizeof(uint128_t));
    }
  }

  uint64_t ot_num = std::min(remain_num, require_num);

  std::memcpy(data.data() + data_offset,
              ot_buff_.data<uint128_t>() + buff_used_num_,
              ot_num * sizeof(uint128_t));

  buff_used_num_ += ot_num;
  // add state
  consumed_ot_num_ += ot_num;
  data_offset += ot_num;

  // In the case of running out of ot_buff_
  if (ot_num < require_num) {
    require_num -= ot_num;
    Bootstrap();

    // Worst Case
    // Require_num is greater then "buff_upper_bound_ - reserve_num_"
    // which means that an extra "Bootstrap" is needed.
    if (require_num > (buff_upper_bound_ - reserve_num_)) {
      SPDLOG_WARN("[YACL] Worst Case!!! current require_num {}", require_num);
      // Bootstrap would reset buff_used_num_
      memcpy(data.data() + data_offset,
             ot_buff_.data<uint128_t>() + reserve_num_,
             (buff_upper_bound_ - reserve_num_) * sizeof(uint128_t));
      require_num -= (buff_upper_bound_ - reserve_num_);
      consumed_ot_num_ += (buff_upper_bound_ - reserve_num_);
      data_offset += (buff_upper_bound_ - reserve_num_);

      // Bootstrap would reset buff_used_num_
      Bootstrap();
    }
    memcpy(data.data() + data_offset,
           ot_buff_.data<uint128_t>() + buff_used_num_,
           require_num * sizeof(uint128_t));
    buff_used_num_ += require_num;
    consumed_ot_num_ += require_num;
  }
}

void YaclFerretOTeAdapter::send_cot(absl::Span<uint128_t> data) {
  SPU_ENFORCE(is_sender_ == true);
  uint64_t num = data.size();

  rcot(data);

  auto bv = ctx_->Recv(ctx_->NextRank(), "ferret_send_cot:flip");
  auto flip_choices_span = absl::MakeSpan(
      reinterpret_cast<uint128_t*>(bv.data()), bv.size() / sizeof(uint128_t));

  yacl::dynamic_bitset<uint128_t> choices;
  choices.append(flip_choices_span.data(),
                 flip_choices_span.data() + flip_choices_span.size());

  for (uint64_t i = 0; i < num; ++i) {
    data[i] = data[i] ^ (choices[i] * Delta);
  }
}

void YaclFerretOTeAdapter::recv_cot(
    absl::Span<uint128_t> data,
    const yacl::dynamic_bitset<uint128_t>& choices) {
  SPU_ENFORCE(is_sender_ == false);
  uint64_t num = data.size();

  rcot(data);

  // Warning: low efficiency
  auto flip_choices = choices;
  for (uint64_t i = 0; i < num; ++i) {
    flip_choices[i] = choices[i] ^ (data[i] & 0x1);
  }

  auto bv =
      yacl::ByteContainerView(reinterpret_cast<uint8_t*>(flip_choices.data()),
                              flip_choices.num_blocks() * sizeof(uint128_t));

  ctx_->SendAsync(ctx_->NextRank(), bv, "ferret_recv_cot:flip");
}

void YaclFerretOTeAdapter::Bootstrap() {
  auto begin = std::chrono::high_resolution_clock::now();
  if (is_sender_) {
    yacl::AlignedVector<uint128_t> send_ot(
        ot_buff_.data<uint128_t>(), ot_buff_.data<uint128_t>() + reserve_num_);
    auto send_ot_store = yc::MakeCompactOtSendStore(std::move(send_ot), Delta);
    yc::FerretOtExtSend_cheetah(ctx_, send_ot_store, lpn_param_, lpn_param_.n,
                                MakeSpan_Uint128(ot_buff_));
  } else {
    yacl::AlignedVector<uint128_t> recv_ot(
        ot_buff_.data<uint128_t>(), ot_buff_.data<uint128_t>() + reserve_num_);
    auto recv_ot_store = yc::MakeCompactOtRecvStore(std::move(recv_ot));
    yc::FerretOtExtRecv_cheetah(ctx_, recv_ot_store, lpn_param_, lpn_param_.n,
                                MakeSpan_Uint128(ot_buff_));
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto elapse =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - begin)
          .count();
  // Notice that, we will reserve the some OT instances for boostrapping in
  // Ferret OTe protocol
  buff_used_num_ = reserve_num_;
  buff_upper_bound_ = lpn_param_.n;

  // add state
  ++bootstrap_num_;
  bootstrap_time_ += elapse * 1000;
}

void YaclFerretOTeAdapter::BootstrapInplace(absl::Span<uint128_t> ot,
                                            absl::Span<uint128_t> data) {
  YACL_ENFORCE(ot.size() == reserve_num_);
  YACL_ENFORCE(data.size() == lpn_param_.n);

  yacl::AlignedVector<uint128_t> ot_tmp(ot.data(), ot.data() + reserve_num_);

  auto begin = std::chrono::high_resolution_clock::now();
  if (is_sender_) {
    auto send_ot_store = yc::MakeCompactOtSendStore(std::move(ot_tmp), Delta);
    yc::FerretOtExtSend_cheetah(ctx_, send_ot_store, lpn_param_, lpn_param_.n,
                                data);
  } else {
    auto recv_ot_store = yc::MakeCompactOtRecvStore(std::move(ot_tmp));
    yc::FerretOtExtRecv_cheetah(ctx_, recv_ot_store, lpn_param_, lpn_param_.n,
                                data);
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto elapse =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - begin)
          .count();

  // add state
  ++bootstrap_num_;
  bootstrap_time_ += elapse * 1000;
}

// ------------------------
//   IknpOTeAdapter
// ------------------------

uint128_t YaclIknpOTeAdapter::yacl_id_ = 0;

void YaclIknpOTeAdapter::OneTimeSetup() {
  if (is_setup_) {
    return;
  }
  // Sender
  if (is_sender_) {
    auto choices = yc::RandBits<yacl::dynamic_bitset<uint128_t>>(128, true);
    // Generate BaseOT for IKNP-OTe
    auto base_ot = yc::BaseOtRecv(ctx_, choices, 128);
    recv_ot_ptr_ = std::make_unique<yc::OtRecvStore>(std::move(base_ot));
    Delta = choices.data()[0];
  }
  // Receiver
  else {
    // Generate BaseOT for IKNP-OTe
    auto base_ot = yc::BaseOtSend(ctx_, 128);
    // Random choices for IKNP-OTe
    send_ot_ptr_ = std::make_unique<yc::OtSendStore>(std::move(base_ot));
  }

  is_setup_ = true;
}

void YaclIknpOTeAdapter::send_cot(absl::Span<uint128_t> data) {
  YACL_ENFORCE(is_sender_);
  auto begin = std::chrono::high_resolution_clock::now();

  // [Warning] copy, low efficiency
  yacl::Buffer send_buf(2 * data.size() * sizeof(uint128_t));
  // std::vector<std::array<uint128_t, 2>> send_blocks(data.size());
  auto send_span = absl::MakeSpan(
      reinterpret_cast<std::array<uint128_t, 2>*>(send_buf.data()),
      data.size());
  yc::IknpOtExtSend(ctx_, *recv_ot_ptr_, send_span, true);
  std::transform(
      send_span.cbegin(), send_span.cend(), data.begin(),
      [](const std::array<uint128_t, 2>& blocks) { return blocks[0]; });

  auto end = std::chrono::high_resolution_clock::now();
  auto elapse =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - begin)
          .count();
  ote_time_ += elapse * 1000;
  consumed_ot_num_ += data.size();
  ++ote_num_;
}

void YaclIknpOTeAdapter::recv_cot(
    absl::Span<uint128_t> data,
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

// ------------------------
//   SoftspokenOTeAdapter
// ------------------------

uint128_t YaclSsOTeAdapter::yacl_id_ = 0;

void YaclSsOTeAdapter::OneTimeSetup() {
  if (is_setup_) {
    return;
  }
  if (is_sender_) {
    ss_sender_->OneTimeSetup(ctx_);
    Delta = ss_sender_->GetDelta();
  } else {
    ss_receiver_->OneTimeSetup(ctx_);
  }
}

void YaclSsOTeAdapter::send_cot(absl::Span<uint128_t> data) {
  YACL_ENFORCE(is_sender_);
  auto begin = std::chrono::high_resolution_clock::now();
  // [Warning] copy, low efficiency
  yacl::Buffer send_buf(2 * data.size() * sizeof(uint128_t));
  // std::vector<std::array<uint128_t, 2>> send_blocks(data.size());
  auto send_span = absl::MakeSpan(
      reinterpret_cast<std::array<uint128_t, 2>*>(send_buf.data()),
      data.size());

  ss_sender_->Send(ctx_, send_span, true);
  std::transform(
      send_span.cbegin(), send_span.cend(), data.begin(),
      [](const std::array<uint128_t, 2>& blocks) { return blocks[0]; });

  auto end = std::chrono::high_resolution_clock::now();
  auto elapse =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - begin)
          .count();
  ote_time_ += elapse * 1000;
  consumed_ot_num_ += data.size();
  ++ote_num_;
}

void YaclSsOTeAdapter::recv_cot(
    absl::Span<uint128_t> data,
    const yacl::dynamic_bitset<uint128_t>& choices) {
  YACL_ENFORCE(is_sender_ == false);
  auto begin = std::chrono::high_resolution_clock::now();
  ss_receiver_->Recv(ctx_, choices, data, true);
  auto end = std::chrono::high_resolution_clock::now();
  auto elapse =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - begin)
          .count();
  ote_time_ += elapse * 1000;
  consumed_ot_num_ += data.size();
  ++ote_num_;
}
};  // namespace spu::mpc::cheetah
