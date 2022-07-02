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

#include <functional>

#include "cheetah_io_channel.h"
#include "silent_ot_pack.h"

namespace spu {
using IO = CheetahIo;

template <typename type>
struct DReluConfig {
  int l{};
  int num_cmps{};
  uint8_t two_small = 1 << 1;
  uint8_t zero_small = 0;
  uint64_t mask_take_32 = -1;
  uint64_t msb_one{};
  uint64_t cut_mask{};
  uint64_t relu_comparison_rhs{};
  type mask_l;
  type relu_comparison_rhs_type;
  type cut_mask_type;
  type msb_one_type;
};

#define MILL_PARAM 4

struct MillionaireConfig {
  int l, r, log_alpha, beta, beta_pow;
  int num_digits, num_triples_corr, num_triples_std, log_num_digits;
  int num_triples;
  uint8_t mask_beta, mask_r;
};

class Triple {
 public:
  bool packed;
  std::vector<uint8_t> ai;
  std::vector<uint8_t> bi;
  std::vector<uint8_t> ci;
  int num_triples, num_bytes;

  explicit Triple(int num_triples, bool packed = true);
};

class NonlinearProtocols {
 public:
  std::shared_ptr<SilentOTPack> otpack_;
  int party_;

  explicit NonlinearProtocols(std::shared_ptr<SilentOTPack> otpack);

  ~NonlinearProtocols();

  void flush();

  template <typename T>
  void open(T *plain, const T *share, int size, std::function<T(T, T)> op,
            int bw = 0);

  void beaver_triple(uint8_t *ai, uint8_t *bi, uint8_t *ci, int num_triples,
                     bool packed = true) const;

  void beaver_triple(Triple *triples);

  template <typename T>
  void randbit(T *r, int num);

  /**
  b2a for single bit
  */
  template <typename T>
  void b2a(T *y, const uint8_t *x, int32_t size, int32_t bw_y = 0);

  /**
  b2a for multi packed bits
  */
  template <typename T>
  void b2a_full(T *y, const T *x, int32_t size, int32_t bw = 0);

  template <typename type>
  void drelu(uint8_t *drelu_res, const type *share, int num_drelu, int l = 0);

  template <typename type>
  void relu(type *result, const type *share, int num_relu,
            uint8_t *drelu_res = nullptr, int l = 0);

  template <typename T>
  void truncate(T *outB, const T *inA, int32_t dim, int32_t shift, int32_t bw,
                bool signed_arithmetic = true, uint8_t *msb_x = nullptr);

  template <typename T>
  void truncate_msb(T *outB, const T *inA, int32_t dim, int32_t shift,
                    int32_t bw, bool signed_arithmetic, uint8_t *msb_x);

  template <typename T>
  void truncate_msb0(T *outB, const T *inA, int32_t dim, int32_t shift,
                     int32_t bw, bool signed_arithmetic = true);

  // y = sel * x
  void multiplexer(uint64_t *y, const uint64_t *x, const uint8_t *sel,
                   int32_t size, int32_t bw_x, int32_t bw_y);

  template <typename T>
  void lookup_table(T *y, const T *const *spec, const T *x, int32_t size,
                    int32_t bw_x, int32_t bw_y);

  template <typename T>
  void msb(uint8_t *msb_x, const T *x, int32_t size, int32_t bw_x = 0);

  template <typename T>
  void MSB_to_Wrap(uint8_t *wrap_x, const T *x, const uint8_t *msb_x,
                   int32_t size, int32_t bw_x = 0);

  template <typename T>
  void msb0_to_wrap(uint8_t *wrap_x, const T *x, int32_t size,
                    int32_t bw_x = 0);

  template <typename T>
  void msb1_to_wrap(uint8_t *wrap_x, const T *x, int32_t size,
                    int32_t bw_x = 0);

  template <typename T>
  void compare(uint8_t *res, const T *data, int num_cmps, int bitlength = 0,
               bool greater_than = true, bool equality = false,
               int radix_base = MILL_PARAM);

 private:
  template <typename type>
  std::unique_ptr<DReluConfig<type> > configureDRelu(int l);

  std::unique_ptr<MillionaireConfig> configureMillionaire(
      int bitlength, int radix_base = MILL_PARAM);

  void set_leaf_ot_messages(uint8_t *ot_messages, uint8_t digit, int N,
                            uint8_t mask_cmp, uint8_t mask_eq,
                            bool greater_than, bool eq = true);

  void traverse_and_compute_ANDs(MillionaireConfig *config, int num_cmps,
                                 uint8_t *leaf_res_eq, uint8_t *leaf_res_cmp);

  void AND_step_1(uint8_t *ei, uint8_t *fi, uint8_t *xi, uint8_t *yi,
                  uint8_t *ai, uint8_t *bi, int num_ANDs);

  void AND_step_2(uint8_t *zi, uint8_t *e, uint8_t *f, uint8_t *ai, uint8_t *bi,
                  uint8_t *ci, int num_ANDs);
};

}  // namespace spu
