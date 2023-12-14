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

uint128_t YaclFerretOTeAdapter::yacl_id_ = 0;

void YaclFerretOTeAdapter::OneTimeSetup() {
  if (is_setup_) {
    return;
  }

  uint128_t pre_lpn_num_ = yc::FerretCotHelper(pre_lpn_param_, 0);

  // Sender
  if (is_sender_) {
    auto choices = yc::RandBits<yacl::dynamic_bitset<uint128_t>>(128, true);
    // In Compact mode, the last bit of delta is one
    choices.data()[0] = (choices.data()[0] | ~one);

    // Generate BaseOT for IKNP-OTe
    auto base_ot = yc::BaseOtRecv(ctx_, choices, 128);
    // Invoke IKNP-OTe to generate COT
    auto iknp_send_ot = yc::IknpOtExtSend(ctx_, base_ot, pre_lpn_num_, true);
    Delta = iknp_send_ot.GetDelta();

    // Notice !!!
    // IknpOtExt Protocol would generate Normal mode OtStore
    // But ferret OTe require COMPACT mode OtStore
    yc::OtSendStore pre_ferret_sent_ot_ =
        yc::OtSendStore(pre_lpn_num_, yc::OtStoreType::Compact);
    pre_ferret_sent_ot_.SetDelta(Delta);
    // Warning: copy, low efficiency
    for (uint64_t i = 0; i < pre_lpn_num_; ++i) {
      pre_ferret_sent_ot_.SetCompactBlock(i, iknp_send_ot.GetBlock(i, 0) & one);
    }
    // pre ferret OTe
    auto send_ot_store = yc::FerretOtExtSend(ctx_, pre_ferret_sent_ot_,
                                             pre_lpn_param_, pre_lpn_param_.n);
    // fill ot_buff_
    for (size_t i = 0; i < pre_lpn_param_.n; ++i) {
      ot_buff_[i] = send_ot_store.GetBlock(i, 0);
    }
  }
  // Receiver
  else {
    // Generate BaseOT for IKNP-OTe
    auto base_ot = yc::BaseOtSend(ctx_, 128);
    // Random choices for IKNP-OTe
    auto choices =
        yc::RandBits<yacl::dynamic_bitset<uint128_t>>(pre_lpn_num_, true);
    // Invoke IKNP-OTe to generate COT
    auto iknp_recv_ot =
        yc::IknpOtExtRecv(ctx_, base_ot, choices, pre_lpn_num_, true);

    // Notice !!!
    // IknpOtExt Protocol would generate Normal mode OtStore
    // But ferret OTe require COMPACT mode OtStore
    yc::OtRecvStore pre_ferret_recv_ot_ =
        yc::OtRecvStore(pre_lpn_num_, yc::OtStoreType::Compact);
    // Warning: copy, low efficiency
    for (uint64_t i = 0; i < pre_lpn_num_; ++i) {
      uint128_t block = (iknp_recv_ot.GetBlock(i) & one) | choices[i];
      pre_ferret_recv_ot_.SetBlock(i, block);
    }

    // pre ferret OTe
    auto recv_ot_store = yc::FerretOtExtRecv(ctx_, pre_ferret_recv_ot_,
                                             pre_lpn_param_, pre_lpn_param_.n);
    // fill ot_buff_
    for (size_t i = 0; i < pre_lpn_param_.n; ++i) {
      ot_buff_[i] = recv_ot_store.GetBlock(i);
    }
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
        absl::MakeSpan(ot_buff_.data(), reserve_num_);
    while (require_num > lpn_param_.n) {
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
      memcpy(ot_buff_.data(), ot_span.data(), reserve_num_ * sizeof(uint128_t));
    }
  }

  uint64_t ot_num = std::min(remain_num, require_num);

  memcpy(data.data() + data_offset, ot_buff_.data() + buff_used_num_,
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
      memcpy(data.data() + data_offset, ot_buff_.data() + buff_used_num_,
             (buff_upper_bound_ - reserve_num_) * sizeof(uint128_t));
      require_num -= (buff_upper_bound_ - reserve_num_);
      consumed_ot_num_ += (buff_upper_bound_ - reserve_num_);
      data_offset += (buff_upper_bound_ - reserve_num_);

      // Bootstrap would reset buff_used_num_
      Bootstrap();
    }
    memcpy(data.data() + data_offset, ot_buff_.data() + buff_used_num_,
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
    yacl::AlignedVector<uint128_t> send_ot(ot_buff_.begin(),
                                           ot_buff_.begin() + reserve_num_);
    auto send_ot_store = yc::MakeCompactOtSendStore(std::move(send_ot), Delta);
    yc::FerretOtExtSend_cheetah(ctx_, send_ot_store, lpn_param_, lpn_param_.n,
                                absl::MakeSpan(ot_buff_.data(), lpn_param_.n));
  } else {
    yacl::AlignedVector<uint128_t> recv_ot(ot_buff_.begin(),
                                           ot_buff_.begin() + reserve_num_);
    auto recv_ot_store = yc::MakeCompactOtRecvStore(std::move(recv_ot));
    yc::FerretOtExtRecv_cheetah(ctx_, recv_ot_store, lpn_param_, lpn_param_.n,
                                absl::MakeSpan(ot_buff_.data(), lpn_param_.n));
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

};  // namespace spu::mpc::cheetah
