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

#pragma once

#include <math.h>

#include <algorithm>
#include <stdexcept>

#include "cheetah_io_channel.h"
#include "emp-ot/cot.h"
#include "emp-ot/ferret/ferret_cot.h"
#include "emp-tool/utils/block.h"
#include "mitccrh_exp.h"
#include "ot-utils.h"

namespace spu {
typedef CheetahIo IO;

class SilentOT {
 public:
  std::unique_ptr<emp::FerretCOT<IO>> ferret;
  MITCCRHExp<8> mitccrh_exp{};

  SilentOT(int party, int threads, IO **ios, bool malicious = false,
           bool run_setup = true, std::string pre_file = "",
           bool warm_up = true);

  ~SilentOT() = default;

  void send_impl(const emp::block *data0, const block *data1, int64_t length) {
    send_ot_cm_cc(data0, data1, length);
  }

  void recv_impl(block *data, const bool *b, int64_t length) {
    recv_ot_cm_cc(data, b, length);
  }

  template <typename T>
  void send_impl(T **data, int length, int l) {
    send_ot_cm_cc(data, length, l);
  }

  template <typename T>
  void recv_impl(T *data, const uint8_t *b, int length, int l) {
    recv_ot_cm_cc(data, b, length, l);
  }

  template <typename T>
  void send_impl(T **data, int length, int N, int l) {
    send_ot_cm_cc(data, length, N, l);
  }

  template <typename T>
  void recv_impl(T *data, const uint8_t *b, int length, int N, int l) {
    recv_ot_cm_cc(data, b, length, N, l);
  }

  void send_cot(uint64_t *data0, const uint64_t *corr, int length, int l) {
    send_ot_cam_cc(data0, corr, length, l);
  }

  void recv_cot(uint64_t *data, const bool *b, int length, int l) {
    recv_ot_cam_cc(data, b, length, l);
  }

  template <typename T>
  void send_cot(T *data0, const T *corr, int length) {
    send_ot_cam_cc(data0, corr, length);
  }

  template <typename T>
  void recv_cot(T *data, const bool *b, int length) {
    recv_ot_cam_cc(data, b, length);
  }

  // chosen additive message, chosen choice
  // Sender chooses one message 'corr'. A correlation is defined by the addition
  // function: f(x) = x + corr Sender receives a random message 'x' as output
  // ('data0').
  template <typename T>
  void send_ot_cam_cc(T *data0, const T *corr, int64_t length);

  // chosen additive message, chosen choice
  // Receiver chooses a choice bit 'b', and
  // receives 'x' if b = 0, and 'x + corr' if b = 1
  template <typename T>
  void recv_ot_cam_cc(T *data, const bool *b, int64_t length);

  // chosen additive message, chosen choice
  // Sender chooses one message 'corr'. A correlation is defined by the addition
  // function: f(x) = x + corr Sender receives a random message 'x' as output
  // ('data0').
  void send_ot_cam_cc(uint64_t *data0, const uint64_t *corr, int64_t length,
                      int l);

  // chosen additive message, chosen choice
  // Receiver chooses a choice bit 'b', and
  // receives 'x' if b = 0, and 'x + corr' if b = 1
  void recv_ot_cam_cc(uint64_t *data, const bool *b, int64_t length, int l);

  // chosen message, chosen choice
  void send_ot_cm_cc(const block *data0, const block *data1, int64_t length);

  // chosen message, chosen choice
  void recv_ot_cm_cc(block *data, const bool *r, int64_t length);

  // chosen message, chosen choice.
  // Here, the 2nd dim of data is always 2. We use T** instead of T*[2] or two
  // arguments of T*, in order to be general and compatible with the API of
  // 1-out-of-N OT.
  template <typename T>
  void send_ot_cm_cc(T **data, int64_t length, int l);

  // chosen message, chosen choice
  // Here, r[i]'s value is always 0 or 1. We use uint8_t instead of bool, in
  // order to be general and compatible with the API of 1-out-of-N OT.
  template <typename T>
  void recv_ot_cm_cc(T *data, const uint8_t *r, int64_t length, int l);

  // random correlated message, chosen choice
  void send_ot_rcm_cc(block *data0, int64_t length);

  // random correlated message, chosen choice
  void recv_ot_rcm_cc(block *data, const bool *b, int64_t length);

  // random message, chosen choice
  void send_ot_rm_cc(block *data0, block *data1, int64_t length);

  // random message, chosen choice
  void recv_ot_rm_cc(block *data, const bool *r, int64_t length);

  // random message, random choice
  void send_ot_rm_rc(block *data0, block *data1, int64_t length);

  // random message, random choice
  void recv_ot_rm_rc(block *data, bool *r, int64_t length);

  // random message, random choice
  template <typename T>
  void send_ot_rm_rc(T *data0, T *data1, int64_t length, int l);

  // random message, random choice
  template <typename T>
  void recv_ot_rm_rc(T *data, bool *r, int64_t length, int l);

  // chosen message, chosen choice.
  // One-oo-N OT, where each message has l bits. Here, the 2nd dim of data is N.
  template <typename T>
  void send_ot_cm_cc(T **data, int64_t length, int N, int l);

  // chosen message, chosen choice
  // One-oo-N OT, where each message has l bits. Here, r[i]'s value is in [0,
  // N).
  template <typename T>
  void recv_ot_cm_cc(T *data, const uint8_t *r, int64_t length, int N, int l);

  void send_batched_got(uint64_t *data, int num_ot, int l,
                        int msgs_per_ot = 1) {
    throw std::logic_error("Not implemented");
  }

  void recv_batched_got(uint64_t *data, const uint8_t *r, int num_ot, int l,
                        int msgs_per_ot = 1) {
    throw std::logic_error("Not implemented");
  }

  void send_batched_cot(uint64_t *data0, uint64_t *corr,
                        std::vector<int> msg_len, int num_ot,
                        int msgs_per_ot = 1) {
    throw std::logic_error("Not implemented");
  }

  void recv_batched_cot(uint64_t *data, bool *b, std::vector<int> msg_len,
                        int num_ot, int msgs_per_ot = 1) {
    throw std::logic_error("Not implemented");
  }
};

class SilentOTN {
 public:
  SilentOT *silent_ot;
  int N;

  SilentOTN(SilentOT *silent_ot, int N) {
    this->silent_ot = silent_ot;
    this->N = N;
  }

  template <typename T>
  void send_impl(T **data, int length, int l) {
    silent_ot->send_ot_cm_cc(data, length, N, l);
  }

  template <typename T>
  void recv_impl(T *data, const uint8_t *b, int length, int l) {
    silent_ot->recv_ot_cm_cc(data, b, length, N, l);
  }
};

}  // namespace spu
