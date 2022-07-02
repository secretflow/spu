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

#include "nonlinear_protocols.h"

#include "spdlog/spdlog.h"
#include "utils.h"

namespace spu {

Triple::Triple(int num_triples, bool packed)
    : packed(packed), num_triples(num_triples) {
  if (packed) {
    assert(num_triples % 8 == 0);
    num_bytes = num_triples / 8;
  } else {
    num_bytes = num_triples;
  }
  ai.resize(num_bytes);
  bi.resize(num_bytes);
  ci.resize(num_bytes);
}

NonlinearProtocols::NonlinearProtocols(std::shared_ptr<SilentOTPack> otpack)
    : otpack_(std::move(otpack)), party_(otpack_->party_) {}

NonlinearProtocols::~NonlinearProtocols() {
  try {
    flush();
  } catch (const std::exception &e) {
    SPDLOG_ERROR("Error in flush: {}", e.what());
  }
}

void NonlinearProtocols::flush() { otpack_->io_->flush(); }

template <typename T>
void NonlinearProtocols::open(T *plain, const T *share, int size,
                              std::function<T(T, T)> op, int bw) {
  if (bw <= 0) {
    bw = sizeof(T) * 8;
  }

  otpack_->io_->send_data_partial(share, size, bw);
  otpack_->io_->recv_data_partial(plain, size, bw);

  for (int i = 0; i < size; i++) {
    plain[i] = op(plain[i], share[i]);
  }
}

void NonlinearProtocols::beaver_triple(Triple *triples) {
  beaver_triple(triples->ai.data(), triples->bi.data(), triples->ci.data(),
                triples->num_triples, triples->packed);
}

void NonlinearProtocols::beaver_triple(uint8_t *ai, uint8_t *bi, uint8_t *ci,
                                       int num_triples, bool packed) const {
  if (num_triples == 0) {
    return;
  }

  std::unique_ptr<uint8_t[]> pa;
  std::unique_ptr<uint8_t[]> pb;
  std::unique_ptr<uint8_t[]> pc;
  uint8_t *a = nullptr;
  uint8_t *b = nullptr;
  uint8_t *c = nullptr;

  if (packed) {
    pa.reset(new uint8_t[num_triples]);
    pb.reset(new uint8_t[num_triples]);
    pc.reset(new uint8_t[num_triples]);
    a = pa.get();
    b = pb.get();
    c = pc.get();
  } else {
    a = ai;
    b = bi;
    c = ci;
  }
  std::vector<uint8_t> u(num_triples);
  std::vector<uint8_t> v(num_triples);

  switch (party_) {
    case emp::ALICE: {
      otpack_->silent_ot_reversed_->template recv_ot_rm_rc<uint8_t>(
          u.data(), reinterpret_cast<bool *>(a), num_triples, 1);
      otpack_->silent_ot_->send_ot_rm_rc(v.data(), b, num_triples, 1);
      break;
    }
    case emp::BOB: {
      otpack_->silent_ot_reversed_->template send_ot_rm_rc<uint8_t>(
          v.data(), b, num_triples, 1);
      otpack_->silent_ot_->recv_ot_rm_rc(u.data(), reinterpret_cast<bool *>(a),
                                         num_triples, 1);
      break;
    }
  }
  otpack_->io_->flush();

  for (int i = 0; i < num_triples; i++) {
    b[i] = b[i] ^ v[i];
  }
  for (int i = 0; i < num_triples; i++) {
    c[i] = (a[i] & b[i]) ^ u[i] ^ v[i];
  }

  if (packed) {
    for (int i = 0; i < num_triples; i += 8) {
      ai[i / 8] = bool_to_uint8(a + i, 8);
      bi[i / 8] = bool_to_uint8(b + i, 8);
      ci[i / 8] = bool_to_uint8(c + i, 8);
    }
  }
}

template <typename T>
void NonlinearProtocols::randbit(T *r, int num) {
  // Generate random bits
  emp::PRG prg;
  std::vector<uint8_t> local_boolean_share(num);
  prg.random_bool(reinterpret_cast<bool *>(local_boolean_share.data()), num);

  b2a(r, local_boolean_share.data(), num);
}

template <typename T>
void NonlinearProtocols::b2a(T *y, const uint8_t *x, int32_t size,
                             int32_t bw_y) {
  int32_t max_bw = sizeof(T) * 8;
  if (bw_y <= 0) {
    bw_y = max_bw;
  }
  assert(bw_y <= max_bw);
  T mask = (bw_y == max_bw ? ~T(0) : ((T(1) << bw_y) - 1));

  if (party_ == emp::ALICE) {
    std::vector<T> corr_data(size);
    for (int i = 0; i < size; i++) {
      corr_data[i] = (-2 * T(x[i])) & mask;
    }
    otpack_->silent_ot_->send_ot_cam_cc(y, corr_data.data(), size);

    for (int i = 0; i < size; i++) {
      y[i] = (T(x[i]) - y[i]) & mask;
    }
  } else {  // party_ == emp::BOB
    otpack_->silent_ot_->recv_ot_cam_cc(y, reinterpret_cast<const bool *>(x),
                                        size);

    for (int i = 0; i < size; i++) {
      y[i] = (T(x[i]) + y[i]) & mask;
    }
  }
}

template <typename T>
void NonlinearProtocols::b2a_full(T *y, const T *x, int32_t size, int32_t bw) {
  if (bw <= 0) {
    bw = sizeof(T) * 8;
  }

  std::vector<T> randbits(size * bw);
  randbit(randbits.data(), size * bw);

  // open c = x ^ r
  std::vector<T> mask_x(size, T(0));
  for (int32_t i = 0; i < size; i++) {
    for (int32_t j = 0; j < bw; j++) {
      mask_x[i] += (randbits[i * bw + j] & 0x1) << j;
    }
    mask_x[i] ^= x[i];
  }

  std::vector<T> c(size);
  open<T>(c.data(), mask_x.data(), size, std::bit_xor<T>(), bw);

  // compute c + (1 - 2*c) * <r>
  memset(y, 0, sizeof(T) * size);
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < bw; j++) {
      auto c_ij = (c[i] >> j) & 0x1;
      if (party_ == emp::ALICE) {
        y[i] += (c_ij + (1 - c_ij * 2) * randbits[i * bw + j]) << j;
      } else {
        y[i] += ((1 - c_ij * 2) * randbits[i * bw + j]) << j;
      }
    }
  }
}

template <typename type>
std::unique_ptr<DReluConfig<type>> NonlinearProtocols::configureDRelu(int l) {
  std::unique_ptr<DReluConfig<type>> config =
      std::make_unique<DReluConfig<type>>();

  config->l = l;
  if (l != 32 && l != 64) {
    config->mask_l = static_cast<type>((1ULL << l) - 1);
  } else if (l == 32) {
    config->mask_l = -1;
  } else {  // l = 64
    config->mask_l = -1ULL;
  }
  if (sizeof(type) == sizeof(uint64_t)) {
    config->msb_one = (1ULL << (l - 1));
    config->relu_comparison_rhs_type = config->msb_one - 1ULL;
    config->relu_comparison_rhs = config->relu_comparison_rhs_type;
    config->cut_mask_type = config->relu_comparison_rhs_type;
    config->cut_mask = config->cut_mask_type;
  } else {
    config->msb_one_type = (1 << (l - 1));
    config->relu_comparison_rhs_type = config->msb_one_type - 1;
    config->relu_comparison_rhs = config->relu_comparison_rhs_type + 0ULL;
    config->cut_mask_type = config->relu_comparison_rhs_type;
    config->cut_mask = config->cut_mask_type + 0ULL;
  }
  return config;
}

template <typename type>
void NonlinearProtocols::drelu(uint8_t *drelu_res, const type *share,
                               int num_drelu, int l) {
  if (l <= 0) {
    l = sizeof(type) * 8;
  }
  std::unique_ptr<DReluConfig<type>> config = configureDRelu<type>(l);

  std::vector<uint8_t> msb_local_share(num_drelu);
  std::vector<uint64_t> array64(num_drelu);
  std::vector<type> array_type(num_drelu);

  config->num_cmps = num_drelu;
  std::vector<uint8_t> wrap(config->num_cmps);
  for (int i = 0; i < num_drelu; i++) {
    msb_local_share[i] = static_cast<uint8_t>(share[i] >> (l - 1));
    array_type[i] = share[i] & config->cut_mask_type;
  }

  type temp;

  if (party_ == emp::ALICE) {
    for (int i = 0; i < num_drelu; i++) {
      array64[i] = array_type[i] + 0ULL;
    }
  } else {
    for (int i = 0; i < num_drelu; i++) {
      temp = config->relu_comparison_rhs_type -
             array_type[i];  // This value is never negative.
      array64[i] = 0ULL + temp;
    }
  }

  compare(wrap.data(), array64.data(), config->num_cmps, l - 1, true, false);
  for (int i = 0; i < num_drelu; i++) {
    drelu_res[i] = (msb_local_share[i] + wrap[i]) % config->two_small;
  }

  if (party_ == emp::ALICE) {
    for (int i = 0; i < num_drelu; i++) {
      drelu_res[i] = drelu_res[i] ^ 1;
    }
  }
}

template <typename type>
void NonlinearProtocols::relu(type *result, const type *share, int num_relu,
                              uint8_t *drelu_res, int l) {
  if (l <= 0) {
    l = sizeof(type) * 8;
  }

  std::vector<uint8_t> drelu_res_temp(num_relu);
  drelu(drelu_res_temp.data(), share, num_relu);

  if (drelu_res != nullptr) {
    std::copy(drelu_res_temp.begin(), drelu_res_temp.begin() + num_relu,
              drelu_res);
  }

  multiplexer(result, share, drelu_res_temp.data(), num_relu, l, l);
}

template <typename T>
void NonlinearProtocols::truncate(T *outB, const T *inA, int32_t dim,
                                  int32_t shift, int32_t bw,
                                  bool signed_arithmetic, uint8_t *msb_x) {
  if (msb_x != nullptr) {
    return truncate_msb(outB, inA, dim, shift, bw, signed_arithmetic, msb_x);
  }

  if (shift == 0) {
    memcpy(outB, inA, sizeof(T) * dim);
    return;
  }
  assert((bw - shift) > 0 && "Truncation shouldn't truncate the full bitwidth");
  assert((signed_arithmetic && (bw - shift - 1 >= 0)) || !signed_arithmetic);
  assert(inA != outB);

  int32_t max_bw = sizeof(T) * 8;
  if (bw <= 0) {
    bw = max_bw;
  }
  assert(bw <= max_bw);

  T mask_bw = (bw == max_bw ? ~T(0) : ((T(1) << bw) - 1));
  T mask_upper =
      ((bw - shift) == max_bw ? ~T(0) : ((T(1) << (bw - shift)) - 1));

  std::vector<T> inA_temp(dim);

  if (signed_arithmetic && (party_ == emp::ALICE)) {
    for (int i = 0; i < dim; i++) {
      inA_temp[i] = ((inA[i] + (T(1) << (bw - 1))) & mask_bw);
    }
  } else {
    for (int i = 0; i < dim; i++) {
      inA_temp[i] = inA[i];
    }
  }

  std::vector<T> inA_upper(dim);
  std::vector<uint8_t> wrap_upper(dim);
  for (int i = 0; i < dim; i++) {
    inA_upper[i] = inA_temp[i] & mask_bw;
    if (party_ == emp::BOB) {
      inA_upper[i] = (mask_bw - inA_upper[i]) & mask_bw;
    }
  }
  compare(wrap_upper.data(), inA_upper.data(), dim, bw);

  std::vector<T> arith_wrap_upper(dim);
  b2a(arith_wrap_upper.data(), wrap_upper.data(), dim, shift);

  for (int i = 0; i < dim; i++) {
    outB[i] = (((inA_temp[i] >> shift) & mask_upper) -
               (T(1) << (bw - shift)) * arith_wrap_upper[i]) &
              mask_bw;
  }

  if (signed_arithmetic && (party_ == emp::ALICE)) {
    for (int i = 0; i < dim; i++) {
      outB[i] = ((outB[i] - (T(1) << (bw - shift - 1))) & mask_bw);
    }
  }
}

template <typename T>
void NonlinearProtocols::truncate_msb(T *outB, const T *inA, int32_t dim,
                                      int32_t shift, int32_t bw,
                                      bool signed_arithmetic, uint8_t *msb_x) {
  if (shift == 0) {
    memcpy(outB, inA, sizeof(T) * dim);
    return;
  }
  assert((bw - shift) > 0 && "Truncation shouldn't truncate the full bitwidth");
  assert((signed_arithmetic && (bw - shift - 1 >= 0)) || !signed_arithmetic);
  assert(inA != outB);

  int32_t max_bw = sizeof(T) * 8;
  if (bw <= 0) {
    bw = max_bw;
  }
  assert(bw <= max_bw);

  T mask_bw = (bw == max_bw ? ~T(0) : ((T(1) << bw) - 1));
  T mask_upper =
      ((bw - shift) == max_bw ? ~T(0) : ((T(1) << (bw - shift)) - 1));

  std::vector<T> inA_temp(dim);

  if (signed_arithmetic && (party_ == emp::ALICE)) {
    for (int i = 0; i < dim; i++) {
      inA_temp[i] = ((inA[i] + (T(1) << (bw - 1))) & mask_bw);
    }
  } else {
    for (int i = 0; i < dim; i++) {
      inA_temp[i] = inA[i];
    }
  }

  std::vector<T> inA_upper(dim);
  std::vector<uint8_t> wrap_upper(dim);
  for (int i = 0; i < dim; i++) {
    inA_upper[i] = (inA_temp[i] >> shift) & mask_upper;
    if (party_ == emp::BOB) {
      inA_upper[i] = (mask_upper - inA_upper[i]) & mask_upper;
    }
  }

  if (signed_arithmetic) {
    std::vector<uint8_t> inv_msb_x(dim);
    for (int i = 0; i < dim; i++) {
      inv_msb_x[i] = msb_x[i] ^ (party_ == emp::ALICE ? 1 : 0);
    }
    MSB_to_Wrap(wrap_upper.data(), inA_temp.data(), inv_msb_x.data(), dim, bw);
  } else {
    MSB_to_Wrap(wrap_upper.data(), inA_temp.data(), msb_x, dim, bw);
  }

  std::vector<T> arith_wrap_upper(dim);
  b2a(arith_wrap_upper.data(), wrap_upper.data(), dim, shift);

  for (int i = 0; i < dim; i++) {
    outB[i] = (((inA_temp[i] >> shift) & mask_upper) -
               (T(1) << (bw - shift)) * arith_wrap_upper[i]) &
              mask_bw;
  }

  if (signed_arithmetic && (party_ == emp::ALICE)) {
    for (int i = 0; i < dim; i++) {
      outB[i] = ((outB[i] - (T(1) << (bw - shift - 1))) & mask_bw);
    }
  }
}

template <typename T>
void NonlinearProtocols::truncate_msb0(T *outB, const T *inA, int32_t dim,
                                       int32_t shift, int32_t bw,
                                       bool signed_arithmetic) {
  if (shift == 0) {
    memcpy(outB, inA, sizeof(uint64_t) * dim);
    return;
  }
  assert((bw - shift) > 0 && "Truncation shouldn't truncate the full bitwidth");
  assert((signed_arithmetic && (bw - shift - 1 >= 0)) || !signed_arithmetic);
  assert(inA != outB);

  int32_t max_bw = sizeof(T) * 8;
  if (bw <= 0) {
    bw = max_bw;
  }
  assert(bw <= max_bw);

  T mask_bw = (bw == max_bw ? ~T(0) : ((T(1) << bw) - 1));
  T mask_upper =
      ((bw - shift) == max_bw ? ~T(0) : ((T(1) << (bw - shift)) - 1));

  std::vector<T> inA_temp(dim);

  if (signed_arithmetic && (party_ == emp::ALICE)) {
    for (int i = 0; i < dim; i++) {
      inA_temp[i] = ((inA[i] + (T(1) << (bw - 1))) & mask_bw);
    }
  } else {
    for (int i = 0; i < dim; i++) {
      inA_temp[i] = inA[i];
    }
  }

  std::vector<uint8_t> wrap_upper(dim);

  if (signed_arithmetic) {
    msb1_to_wrap(wrap_upper.data(), inA_temp.data(), dim, bw);
  } else {
    msb0_to_wrap(wrap_upper.data(), inA_temp.data(), dim, bw);
  }

  std::vector<T> arith_wrap_upper(dim);
  b2a(arith_wrap_upper.data(), wrap_upper.data(), dim, shift);

  for (int i = 0; i < dim; i++) {
    outB[i] = (((inA_temp[i] >> shift) & mask_upper) -
               (T(1) << (bw - shift)) * arith_wrap_upper[i]) &
              mask_bw;
  }

  if (signed_arithmetic && (party_ == emp::ALICE)) {
    for (int i = 0; i < dim; i++) {
      outB[i] = ((outB[i] - (T(1) << (bw - shift - 1))) & mask_bw);
    }
  }
}

void NonlinearProtocols::multiplexer(uint64_t *y, const uint64_t *x,
                                     const uint8_t *sel, int32_t size,
                                     int32_t bw_x, int32_t bw_y) {
  assert(bw_x <= 64 && bw_y <= 64 && bw_y <= bw_x);
  uint64_t mask_y = (bw_y == 64 ? -1 : ((1ULL << bw_y) - 1));

  std::vector<uint64_t> corr_data(size);
  std::vector<uint64_t> data_S(size);
  std::vector<uint64_t> data_R(size);

  for (int i = 0; i < size; i++) {
    corr_data[i] = (x[i] * (1 - 2 * uint64_t(sel[i]))) & mask_y;
  }
  if (party_ == emp::ALICE) {
    otpack_->silent_ot_->send_cot(data_S.data(), corr_data.data(), size, bw_y);
    otpack_->silent_ot_reversed_->recv_cot(
        data_R.data(), reinterpret_cast<const bool *>(sel), size, bw_y);
  } else {  // party_ == emp::BOB
    otpack_->silent_ot_->recv_cot(
        data_R.data(), reinterpret_cast<const bool *>(sel), size, bw_y);
    otpack_->silent_ot_reversed_->send_cot(data_S.data(), corr_data.data(),
                                           size, bw_y);
  }
  for (int i = 0; i < size; i++) {
    y[i] = ((x[i] * uint64_t(sel[i]) + data_R[i] - data_S[i]) & mask_y);
  }
}

template <typename T>
void NonlinearProtocols::lookup_table(T *y, const T *const *spec, const T *x,
                                      int32_t size, int32_t bw_x,
                                      int32_t bw_y) {
  if (party_ == emp::ALICE) {
    assert(x == nullptr);
    assert(y == nullptr);
  } else {  // party_ == emp::BOB
    assert(spec == nullptr);
  }
  assert(bw_x <= 8 && bw_x >= 2);
  int32_t T_size = sizeof(T) * 8;
  assert(bw_y <= T_size);

  T mask_x = (bw_x == T_size ? -1 : ((1ULL << bw_x) - 1));
  auto N = 1ULL << bw_x;

  if (party_ == emp::ALICE) {
    PRG prg;
    std::vector<T> data_data(N * size);
    std::vector<T *> data(size);
    for (int i = 0; i < size; i++) {
      data[i] = &data_data[i * N];
      for (uint64_t j = 0; j < N; j++) {
        data[i][j] = spec[i][j];
      }
    }
    otpack_->kkot_[bw_x - 1]->send_impl(data.data(), size, bw_y);
  } else {  // party_ == emp::BOB
    std::vector<uint8_t> choice(size);
    for (int i = 0; i < size; i++) {
      choice[i] = x[i] & mask_x;
    }
    otpack_->kkot_[bw_x - 1]->recv_impl(y, choice.data(), size, bw_y);
  }
}

template <typename T>
void NonlinearProtocols::msb(uint8_t *msb_x, const T *x, int32_t size,
                             int32_t bw_x) {
  int32_t max_bw = sizeof(T) * 8;
  if (bw_x <= 0) {
    bw_x = max_bw;
  }
  assert(bw_x <= max_bw);
  int32_t shift = bw_x - 1;
  YASL_ENFORCE_LE(shift, 63);
  uint64_t shift_mask = (1ULL << shift) - 1;

  std::vector<T> tmp_x(size);
  std::vector<uint8_t> msb_xb(size);
  for (int i = 0; i < size; i++) {
    tmp_x[i] = x[i] & shift_mask;
    msb_xb[i] = (x[i] >> shift) & 1;
    if (party_ == emp::BOB) {
      tmp_x[i] = (shift_mask - tmp_x[i]) & shift_mask;
    }
  }

  compare(msb_x, tmp_x.data(), size, bw_x - 1, true);  // computing greater_than

  for (int i = 0; i < size; i++) {
    msb_x[i] = msb_x[i] ^ msb_xb[i];
  }
}

template <typename T>
void NonlinearProtocols::MSB_to_Wrap(uint8_t *wrap_x, const T *x,
                                     const uint8_t *msb_x, int32_t size,
                                     int32_t bw_x) {
  int32_t max_bw = sizeof(T) * 8;
  if (bw_x <= 0) {
    bw_x = max_bw;
  }
  assert(bw_x <= max_bw);

  if (party_ == emp::ALICE) {
    PRG prg;
    prg.random_bool(reinterpret_cast<bool *>(wrap_x), size);
    std::vector<uint8_t> spec_data(size * 4);
    std::vector<uint8_t *> spec(size);
    for (int i = 0; i < size; i++) {
      spec[i] = &spec_data[i * 4];
      uint8_t msb_xb = (x[i] >> (bw_x - 1)) & 1;
      for (int j = 0; j < 4; j++) {
        uint8_t bits_j[2];  // j0 || j1 (LSB to MSB)
        uint8_to_bool(bits_j, j, 2);
        spec[i][j] = (((1 ^ msb_x[i] ^ bits_j[0]) * (msb_xb ^ bits_j[1])) ^
                      (msb_xb * bits_j[1]) ^ wrap_x[i]) &
                     1;
      }
    }
    lookup_table<uint8_t>(nullptr, spec.data(), nullptr, size, 2, 1);
  } else {  // party_ == emp::BOB
    std::vector<uint8_t> lut_in(size);
    for (int i = 0; i < size; i++) {
      lut_in[i] = (((x[i] >> (bw_x - 1)) & 1) << 1) | msb_x[i];
    }
    lookup_table<uint8_t>(wrap_x, nullptr, lut_in.data(), size, 2, 1);
  }
}

template <typename T>
void NonlinearProtocols::msb0_to_wrap(uint8_t *wrap_x, const T *x, int32_t size,
                                      int32_t bw_x) {
  int32_t max_bw = sizeof(T) * 8;
  if (bw_x <= 0) {
    bw_x = max_bw;
  }
  assert(bw_x <= max_bw);

  if (party_ == emp::ALICE) {
    PRG prg;
    prg.random_bool(reinterpret_cast<bool *>(wrap_x), size);
    std::vector<uint8_t> spec_data(size * 2);
    std::vector<uint8_t *> spec(size);
    for (int i = 0; i < size; i++) {
      spec[i] = &spec_data[2 * i];
      uint8_t msb_xb = (x[i] >> (bw_x - 1)) & 1;
      spec[i][0] = wrap_x[i] ^ msb_xb;
      spec[i][1] = wrap_x[i] ^ 1;
    }
    otpack_->silent_ot_->template send_ot_cm_cc<uint8_t>(spec.data(), size, 1);
  } else {  // party_ == emp::BOB
    std::vector<uint8_t> msb_xb(size);
    for (int i = 0; i < size; i++) {
      msb_xb[i] = (x[i] >> (bw_x - 1)) & 1;
    }
    otpack_->silent_ot_->template recv_ot_cm_cc<uint8_t>(wrap_x, msb_xb.data(),
                                                         size, 1);
  }
}

template <typename T>
void NonlinearProtocols::msb1_to_wrap(uint8_t *wrap_x, const T *x, int32_t size,
                                      int32_t bw_x) {
  int32_t max_bw = sizeof(T) * 8;
  if (bw_x <= 0) {
    bw_x = max_bw;
  }
  assert(bw_x <= max_bw);

  if (party_ == emp::ALICE) {
    PRG prg;
    prg.random_bool(reinterpret_cast<bool *>(wrap_x), size);
    std::vector<uint8_t> spec_data(2 * size);
    std::vector<uint8_t *> spec(size);
    for (int i = 0; i < size; i++) {
      spec[i] = &spec_data[2 * i];
      uint8_t msb_xb = (x[i] >> (bw_x - 1)) & 1;
      spec[i][0] = wrap_x[i];
      spec[i][1] = wrap_x[i] ^ msb_xb;
    }
    otpack_->silent_ot_->template send_ot_cm_cc<uint8_t>(spec.data(), size, 1);
  } else {  // party_ == emp::BOB
    std::vector<uint8_t> msb_xb(size);
    for (int i = 0; i < size; i++) {
      msb_xb[i] = (x[i] >> (bw_x - 1)) & 1;
    }
    otpack_->silent_ot_->template recv_ot_cm_cc<uint8_t>(wrap_x, msb_xb.data(),
                                                         size, 1);
  }
}

std::unique_ptr<MillionaireConfig> NonlinearProtocols::configureMillionaire(
    int bitlength, int radix_base) {
  assert(radix_base <= 8);
  std::unique_ptr<MillionaireConfig> config =
      std::make_unique<MillionaireConfig>();

  config->l = bitlength;
  config->beta = radix_base;

  if (bitlength <= config->beta) {
    throw std::invalid_argument("Invalid millionaire parameters");
  }

  config->num_digits = ceil(static_cast<double>(config->l) / config->beta);
  config->r = config->l % config->beta;
  config->log_num_digits = bitlen(config->num_digits);
  config->num_triples_corr =
      2 * config->num_digits - 2 - 2 * config->log_num_digits;
  config->num_triples_std = config->log_num_digits;
  config->num_triples = config->num_triples_std + config->num_triples_corr;
  if (config->beta == 8) {
    config->mask_beta = -1;
  } else {
    config->mask_beta = (1 << config->beta) - 1;
  }
  config->mask_r = (1 << config->r) - 1;
  config->beta_pow = 1 << config->beta;

  return config;
}

void NonlinearProtocols::set_leaf_ot_messages(uint8_t *ot_messages,
                                              uint8_t digit, int N,
                                              uint8_t mask_cmp, uint8_t mask_eq,
                                              bool greater_than, bool eq) {
  for (int i = 0; i < N; i++) {
    if (greater_than) {
      ot_messages[i] = ((digit > i) ^ mask_cmp);
    } else {
      ot_messages[i] = ((digit < i) ^ mask_cmp);
    }
    if (eq) {
      ot_messages[i] = (ot_messages[i] << 1) | ((digit == i) ^ mask_eq);
    }
  }
}

/**************************************************************************************************
 *                         AND computation related functions
 **************************************************************************************************/

void NonlinearProtocols::AND_step_1(uint8_t *ei,  // evaluates batch of 8 ANDs
                                    uint8_t *fi, uint8_t *xi, uint8_t *yi,
                                    uint8_t *ai, uint8_t *bi, int num_ANDs) {
  assert(num_ANDs % 8 == 0);
  for (int i = 0; i < num_ANDs; i += 8) {
    ei[i / 8] = ai[i / 8];
    fi[i / 8] = bi[i / 8];
    ei[i / 8] ^= bool_to_uint8(xi + i, 8);
    fi[i / 8] ^= bool_to_uint8(yi + i, 8);
  }
}

void NonlinearProtocols::AND_step_2(uint8_t *zi,  // evaluates batch of 8 ANDs
                                    uint8_t *e, uint8_t *f, uint8_t *ai,
                                    uint8_t *bi, uint8_t *ci, int num_ANDs) {
  assert(num_ANDs % 8 == 0);
  for (int i = 0; i < num_ANDs; i += 8) {
    uint8_t temp_z;
    if (party_ == emp::ALICE) {
      temp_z = e[i / 8] & f[i / 8];
    } else {
      temp_z = 0;
    }
    temp_z ^= f[i / 8] & ai[i / 8];
    temp_z ^= e[i / 8] & bi[i / 8];
    temp_z ^= ci[i / 8];
    uint8_to_bool(zi + i, temp_z, 8);
  }
}

void NonlinearProtocols::traverse_and_compute_ANDs(MillionaireConfig *config,
                                                   int num_cmps,
                                                   uint8_t *leaf_res_eq,
                                                   uint8_t *leaf_res_cmp) {
  int num_triples = config->num_triples;
  int num_digits = config->num_digits;
  int num_triples_std = config->num_triples_std;

  Triple triples_std(num_triples * num_cmps, true);

  // Generate required Bit-Triples
  beaver_triple(&triples_std);

  // Combine leaf OT results in a bottom-up fashion
  int counter_std = 0;
  int old_counter_std = 0;
  int counter_corr = 0;
  int old_counter_corr = 0;
  int counter_combined = 0;
  int old_counter_combined = 0;
  std::vector<uint8_t> ei((num_triples * num_cmps) / 8);
  std::vector<uint8_t> fi((num_triples * num_cmps) / 8);
  std::vector<uint8_t> e((num_triples * num_cmps) / 8);
  std::vector<uint8_t> f((num_triples * num_cmps) / 8);

  for (int i = 1; i < num_digits; i *= 2) {
    // Mask: ei = xi + ai, fi = yi + bi
    for (int j = 0; j < num_digits and j + i < num_digits; j += 2 * i) {
      // leftmost branch of the tree
      if (j == 0) {
        AND_step_1(ei.data() + (counter_std * num_cmps) / 8,
                   fi.data() + (counter_std * num_cmps) / 8,
                   leaf_res_cmp + j * num_cmps,
                   leaf_res_eq + (j + i) * num_cmps,
                   triples_std.ai.data() + (counter_combined * num_cmps) / 8,
                   triples_std.bi.data() + (counter_combined * num_cmps) / 8,
                   num_cmps);
        counter_std++;
        counter_combined++;

      }
      // other branches
      else {
        AND_step_1(
            ei.data() + ((num_triples_std + 2 * counter_corr) * num_cmps) / 8,
            fi.data() + ((num_triples_std + 2 * counter_corr) * num_cmps) / 8,
            leaf_res_cmp + j * num_cmps, leaf_res_eq + (j + i) * num_cmps,
            triples_std.ai.data() + (counter_combined * num_cmps) / 8,
            triples_std.bi.data() + (counter_combined * num_cmps) / 8,
            num_cmps);
        counter_combined++;
        AND_step_1(
            ei.data() +
                ((num_triples_std + (2 * counter_corr + 1)) * num_cmps) / 8,
            fi.data() +
                ((num_triples_std + (2 * counter_corr + 1)) * num_cmps) / 8,
            leaf_res_eq + j * num_cmps, leaf_res_eq + (j + i) * num_cmps,
            triples_std.ai.data() + (counter_combined * num_cmps) / 8,
            triples_std.bi.data() + (counter_combined * num_cmps) / 8,
            num_cmps);
        counter_combined++;
        counter_corr++;
      }
    }
    int offset_std = (old_counter_std * num_cmps) / 8;
    int size_std = ((counter_std - old_counter_std) * num_cmps) / 8;
    int offset_corr = ((num_triples_std + 2 * old_counter_corr) * num_cmps) / 8;
    int size_corr = (2 * (counter_corr - old_counter_corr) * num_cmps) / 8;

    // Reveal e and f
    if (party_ == emp::ALICE) {
      otpack_->io_->send_data(ei.data() + offset_std, size_std);
      otpack_->io_->send_data(ei.data() + offset_corr, size_corr);
      otpack_->io_->send_data(fi.data() + offset_std, size_std);
      otpack_->io_->send_data(fi.data() + offset_corr, size_corr);
      otpack_->io_->recv_data(e.data() + offset_std, size_std);
      otpack_->io_->recv_data(e.data() + offset_corr, size_corr);
      otpack_->io_->recv_data(f.data() + offset_std, size_std);
      otpack_->io_->recv_data(f.data() + offset_corr, size_corr);
    } else {
      otpack_->io_->recv_data(e.data() + offset_std, size_std);
      otpack_->io_->recv_data(e.data() + offset_corr, size_corr);
      otpack_->io_->recv_data(f.data() + offset_std, size_std);
      otpack_->io_->recv_data(f.data() + offset_corr, size_corr);
      otpack_->io_->send_data(ei.data() + offset_std, size_std);
      otpack_->io_->send_data(ei.data() + offset_corr, size_corr);
      otpack_->io_->send_data(fi.data() + offset_std, size_std);
      otpack_->io_->send_data(fi.data() + offset_corr, size_corr);
    }
    for (int i = 0; i < size_std; i++) {
      e[i + offset_std] ^= ei[i + offset_std];
      f[i + offset_std] ^= fi[i + offset_std];
    }
    for (int i = 0; i < size_corr; i++) {
      e[i + offset_corr] ^= ei[i + offset_corr];
      f[i + offset_corr] ^= fi[i + offset_corr];
    }

    // zi = i*e*f + f*ai + e*bi + ci
    counter_std = old_counter_std;
    counter_corr = old_counter_corr;
    counter_combined = old_counter_combined;
    for (int j = 0; j < num_digits and j + i < num_digits; j += 2 * i) {
      if (j == 0) {
        AND_step_2(leaf_res_cmp + j * num_cmps,
                   e.data() + (counter_std * num_cmps) / 8,
                   f.data() + (counter_std * num_cmps) / 8,
                   triples_std.ai.data() + (counter_combined * num_cmps) / 8,
                   triples_std.bi.data() + (counter_combined * num_cmps) / 8,
                   triples_std.ci.data() + (counter_combined * num_cmps) / 8,
                   num_cmps);
        counter_combined++;

        for (int k = 0; k < num_cmps; k++) {
          leaf_res_cmp[j * num_cmps + k] ^=
              leaf_res_cmp[(j + i) * num_cmps + k];
        }
        counter_std++;
      } else {
        AND_step_2(
            leaf_res_cmp + j * num_cmps,
            e.data() + ((num_triples_std + 2 * counter_corr) * num_cmps) / 8,
            f.data() + ((num_triples_std + 2 * counter_corr) * num_cmps) / 8,
            triples_std.ai.data() + (counter_combined * num_cmps) / 8,
            triples_std.bi.data() + (counter_combined * num_cmps) / 8,
            triples_std.ci.data() + (counter_combined * num_cmps) / 8,
            num_cmps);
        counter_combined++;
        AND_step_2(
            leaf_res_eq + j * num_cmps,
            e.data() +
                ((num_triples_std + (2 * counter_corr + 1)) * num_cmps) / 8,
            f.data() +
                ((num_triples_std + (2 * counter_corr + 1)) * num_cmps) / 8,
            triples_std.ai.data() + (counter_combined * num_cmps) / 8,
            triples_std.bi.data() + (counter_combined * num_cmps) / 8,
            triples_std.ci.data() + (counter_combined * num_cmps) / 8,
            num_cmps);
        counter_combined++;

        for (int k = 0; k < num_cmps; k++) {
          leaf_res_cmp[j * num_cmps + k] ^=
              leaf_res_cmp[(j + i) * num_cmps + k];
        }
        counter_corr++;
      }
    }
    old_counter_std = counter_std;
    old_counter_corr = counter_corr;
    old_counter_combined = counter_combined;
  }

  assert(counter_combined == num_triples);
}

template <typename T>
void NonlinearProtocols::compare(uint8_t *res, const T *data, int num_cmps,
                                 int bitlength, bool greater_than,
                                 bool equality, int radix_base) {
  if (bitlength <= 0) bitlength = sizeof(T) * 8;

  std::unique_ptr<MillionaireConfig> config =
      configureMillionaire(bitlength, radix_base);
  const int num_digits = config->num_digits;
  const int r = config->r;
  const int beta = config->beta;
  const int mask_r = config->mask_r;
  const int mask_beta = config->mask_beta;
  const int beta_pow = config->beta_pow;

  int old_num_cmps = num_cmps;
  // num_cmps should be a multiple of 8
  num_cmps = ceil(num_cmps / 8.0) * 8;

  std::vector<T> data_ext(num_cmps, T(0));
  memcpy(data_ext.data(), data, old_num_cmps * sizeof(T));

  std::vector<uint8_t> digits(num_digits * num_cmps);  // num_digits * num_cmps
  std::vector<uint8_t> leaf_res_cmp(num_digits *
                                    num_cmps);  // num_digits * num_cmps
  std::vector<uint8_t> leaf_res_eq(num_digits *
                                   num_cmps);  // num_digits * num_cmps

  // Extract radix-digits from data
  for (int i = 0; i < num_digits; i++) {  // Stored from LSB to MSB
    for (int j = 0; j < num_cmps; j++) {
      if ((i == num_digits - 1) && (r != 0)) {
        digits[i * num_cmps + j] =
            static_cast<uint8_t>(data_ext[j] >> i * beta) & mask_r;
      } else {
        digits[i * num_cmps + j] =
            static_cast<uint8_t>(data_ext[j] >> i * beta) & mask_beta;
      }
    }
  }

  if (party_ == emp::ALICE) {
    std::vector<uint8_t> leaf_ot_messages_data(
        num_digits * num_cmps *
        beta_pow);  // (num_digits * num_cmps) X beta_pow (=2^beta)
    std::vector<uint8_t *> leaf_ot_messages(num_digits * num_cmps);
    for (int i = 0; i < num_digits * num_cmps; i++) {
      leaf_ot_messages[i] = &leaf_ot_messages_data[i * beta_pow];
    }

    PRG prg;
    // Set Leaf OT messages
    prg.random_bool(reinterpret_cast<bool *>(leaf_res_cmp.data()),
                    num_digits * num_cmps);
    prg.random_bool(reinterpret_cast<bool *>(leaf_res_eq.data()),
                    num_digits * num_cmps);

    for (int i = 0; i < num_digits; i++) {
      for (int j = 0; j < num_cmps; j++) {
        if (i == 0) {
          set_leaf_ot_messages(
              leaf_ot_messages[i * num_cmps + j], digits[i * num_cmps + j],
              beta_pow, leaf_res_cmp[i * num_cmps + j], 0, greater_than, false);
        } else if (i == (num_digits - 1) && (r > 0)) {
          set_leaf_ot_messages(leaf_ot_messages[i * num_cmps + j],
                               digits[i * num_cmps + j], beta_pow,
                               leaf_res_cmp[i * num_cmps + j],
                               leaf_res_eq[i * num_cmps + j], greater_than);

        } else {
          set_leaf_ot_messages(leaf_ot_messages[i * num_cmps + j],
                               digits[i * num_cmps + j], beta_pow,
                               leaf_res_cmp[i * num_cmps + j],
                               leaf_res_eq[i * num_cmps + j], greater_than);
        }
      }
    }

    // Perform Leaf OTs
    otpack_->kkot_[beta - 1]->send_impl(leaf_ot_messages.data(),
                                        num_cmps * (num_digits), 2);
  } else {
    // Perform Leaf OTs
    otpack_->kkot_[beta - 1]->recv_impl(leaf_res_cmp.data(), digits.data(),
                                        num_cmps * (num_digits), 2);

    // Extract equality result from leaf_res_cmp
    for (int i = num_cmps; i < num_digits * num_cmps; i++) {
      leaf_res_eq[i] = leaf_res_cmp[i] & 1;
      leaf_res_cmp[i] >>= 1;
    }
  }

  traverse_and_compute_ANDs(config.get(), num_cmps, leaf_res_eq.data(),
                            leaf_res_cmp.data());

  for (int i = 0; i < old_num_cmps; i++) {
    res[i] = leaf_res_cmp[i];
  }
}

template void NonlinearProtocols::open<uint32_t>(
    uint32_t *plain, const uint32_t *share, int size,
    std::function<uint32_t(uint32_t, uint32_t)> op, int bw);
template void NonlinearProtocols::open<uint64_t>(
    uint64_t *plain, const uint64_t *share, int size,
    std::function<uint64_t(uint64_t, uint64_t)> op, int bw);
template void NonlinearProtocols::open<uint128_t>(
    uint128_t *plain, const uint128_t *share, int size,
    std::function<uint128_t(uint128_t, uint128_t)> op, int bw);

template void NonlinearProtocols::randbit<uint32_t>(uint32_t *r, int num);
template void NonlinearProtocols::randbit<uint64_t>(uint64_t *r, int num);
template void NonlinearProtocols::randbit<uint128_t>(uint128_t *r, int num);

template void NonlinearProtocols::b2a<uint32_t>(uint32_t *y, const uint8_t *x,
                                                int32_t size, int32_t bw_y);
template void NonlinearProtocols::b2a<uint64_t>(uint64_t *y, const uint8_t *x,
                                                int32_t size, int32_t bw_y);
template void NonlinearProtocols::b2a<uint128_t>(uint128_t *y, const uint8_t *x,
                                                 int32_t size, int32_t bw_y);

template void NonlinearProtocols::b2a_full<uint32_t>(uint32_t *y,
                                                     const uint32_t *x,
                                                     int32_t size, int32_t bw);
template void NonlinearProtocols::b2a_full<uint64_t>(uint64_t *y,
                                                     const uint64_t *x,
                                                     int32_t size, int32_t bw);
template void NonlinearProtocols::b2a_full<uint128_t>(uint128_t *y,
                                                      const uint128_t *x,
                                                      int32_t size, int32_t bw);

template void NonlinearProtocols::msb<uint32_t>(uint8_t *msb_x,
                                                const uint32_t *x, int32_t size,
                                                int32_t bw_x);
template void NonlinearProtocols::msb<uint64_t>(uint8_t *msb_x,
                                                const uint64_t *x, int32_t size,
                                                int32_t bw_x);
template void NonlinearProtocols::msb<uint128_t>(uint8_t *msb_x,
                                                 const uint128_t *x,
                                                 int32_t size, int32_t bw_x);

template void NonlinearProtocols::MSB_to_Wrap<uint32_t>(uint8_t *wrap_x,
                                                        const uint32_t *x,
                                                        const uint8_t *msb_x,
                                                        int32_t size,
                                                        int32_t bw_x);
template void NonlinearProtocols::MSB_to_Wrap<uint64_t>(uint8_t *wrap_x,
                                                        const uint64_t *x,
                                                        const uint8_t *msb_x,
                                                        int32_t size,
                                                        int32_t bw_x);
template void NonlinearProtocols::MSB_to_Wrap<uint128_t>(uint8_t *wrap_x,
                                                         const uint128_t *x,
                                                         const uint8_t *msb_x,
                                                         int32_t size,
                                                         int32_t bw_x);

template void NonlinearProtocols::msb0_to_wrap<uint32_t>(uint8_t *wrap_x,
                                                         const uint32_t *x,
                                                         int32_t size,
                                                         int32_t bw_x);
template void NonlinearProtocols::msb0_to_wrap<uint64_t>(uint8_t *wrap_x,
                                                         const uint64_t *x,
                                                         int32_t size,
                                                         int32_t bw_x);
template void NonlinearProtocols::msb0_to_wrap<uint128_t>(uint8_t *wrap_x,
                                                          const uint128_t *x,
                                                          int32_t size,
                                                          int32_t bw_x);

template void NonlinearProtocols::msb1_to_wrap<uint32_t>(uint8_t *wrap_x,
                                                         const uint32_t *x,
                                                         int32_t size,
                                                         int32_t bw_x);
template void NonlinearProtocols::msb1_to_wrap<uint64_t>(uint8_t *wrap_x,
                                                         const uint64_t *x,
                                                         int32_t size,
                                                         int32_t bw_x);
template void NonlinearProtocols::msb1_to_wrap<uint128_t>(uint8_t *wrap_x,
                                                          const uint128_t *x,
                                                          int32_t size,
                                                          int32_t bw_x);

template void NonlinearProtocols::compare<uint32_t>(
    uint8_t *res, const uint32_t *data, int num_cmps, int bitlength,
    bool greater_than, bool equality, int radix_base);
template void NonlinearProtocols::compare<uint64_t>(
    uint8_t *res, const uint64_t *data, int num_cmps, int bitlength,
    bool greater_than, bool equality, int radix_base);
template void NonlinearProtocols::compare<uint128_t>(
    uint8_t *res, const uint128_t *data, int num_cmps, int bitlength,
    bool greater_than, bool equality, int radix_base);

template void NonlinearProtocols::truncate<uint32_t>(
    uint32_t *outB, const uint32_t *inA, int32_t dim, int32_t shift, int32_t bw,
    bool signed_arithmetic, uint8_t *msb_x);
template void NonlinearProtocols::truncate<uint64_t>(
    uint64_t *outB, const uint64_t *inA, int32_t dim, int32_t shift, int32_t bw,
    bool signed_arithmetic, uint8_t *msb_x);
template void NonlinearProtocols::truncate<uint128_t>(
    uint128_t *outB, const uint128_t *inA, int32_t dim, int32_t shift,
    int32_t bw, bool signed_arithmetic, uint8_t *msb_x);

template void NonlinearProtocols::truncate_msb<uint32_t>(
    uint32_t *outB, const uint32_t *inA, int32_t dim, int32_t shift, int32_t bw,
    bool signed_arithmetic, uint8_t *msb_x);
template void NonlinearProtocols::truncate_msb<uint64_t>(
    uint64_t *outB, const uint64_t *inA, int32_t dim, int32_t shift, int32_t bw,
    bool signed_arithmetic, uint8_t *msb_x);
template void NonlinearProtocols::truncate_msb<uint128_t>(
    uint128_t *outB, const uint128_t *inA, int32_t dim, int32_t shift,
    int32_t bw, bool signed_arithmetic, uint8_t *msb_x);

template void NonlinearProtocols::truncate_msb0<uint32_t>(
    uint32_t *outB, const uint32_t *inA, int32_t dim, int32_t shift, int32_t bw,
    bool signed_arithmetic);
template void NonlinearProtocols::truncate_msb0<uint64_t>(
    uint64_t *outB, const uint64_t *inA, int32_t dim, int32_t shift, int32_t bw,
    bool signed_arithmetic);
template void NonlinearProtocols::truncate_msb0<uint128_t>(
    uint128_t *outB, const uint128_t *inA, int32_t dim, int32_t shift,
    int32_t bw, bool signed_arithmetic);

}  // namespace spu
