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

#include "silent_ot.h"

#include <utility>
#include <vector>

#include "utils.h"

// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=69884
#if defined __GNUC__ && __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

namespace spu {

SilentOT::SilentOT(int party, int threads, IO** ios, bool malicious,
                   bool run_setup, std::string pre_file, bool warm_up) {
  ferret = std::make_unique<FerretCOT<IO>>(party, threads, ios, malicious,
                                           run_setup, std::move(pre_file));
  if (warm_up) {
    block tmp;
    ferret->rcot(&tmp, 1);
  }

  block s;
  if (party == emp::ALICE) {
    ferret->prg.random_block(&s, 1);
    ferret->io->send_block(&s, 1);
    ferret->mitccrh.setS(s);
    // Need to flush?
    ferret->io->flush();
  } else {
    ferret->io->recv_block(&s, 1);
    ferret->mitccrh.setS(s);
  }
}

// chosen additive message, chosen choice
// Sender chooses one message 'corr'. A correlation is defined by the addition
// function: f(x) = x + corr Sender receives a random message 'x' as output
// ('data0').
template <typename T>
void SilentOT::send_ot_cam_cc(T* data0, const T* corr, int64_t length) {
  std::vector<block> rcm_data(length);
  send_ot_rcm_cc(rcm_data.data(), length);

  block pad[2 * ot_bsize];
  uint32_t corrected_bsize;
  T corr_data[ot_bsize];

  for (int64_t i = 0; i < length; i += ot_bsize) {
    for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
      pad[2 * (j - i)] = rcm_data[j];
      pad[2 * (j - i) + 1] = rcm_data[j] ^ ferret->Delta;
    }

    ferret->mitccrh.template hash<ot_bsize, 2>(pad);

    for (int j = i; j < i + ot_bsize and j < length; ++j) {
      from_block(data0[j], pad[2 * (j - i)]);
      from_block(corr_data[j - i], pad[2 * (j - i) + 1]);
      corr_data[j - i] += corr[j] + data0[j];
    }
    corrected_bsize = std::min(ot_bsize, length - i);

    ferret->io->send_data(corr_data, sizeof(T) * (corrected_bsize));
  }
}

// chosen additive message, chosen choice
// Receiver chooses a choice bit 'b', and
// receives 'x' if b = 0, and 'x + corr' if b = 1
template <typename T>
void SilentOT::recv_ot_cam_cc(T* data, const bool* b, int64_t length) {
  std::vector<block> rcm_data(length);
  recv_ot_rcm_cc(rcm_data.data(), b, length);

  block pad[ot_bsize];

  uint32_t corrected_bsize;
  T corr_data[ot_bsize];

  for (int64_t i = 0; i < length; i += ot_bsize) {
    corrected_bsize = std::min(ot_bsize, length - i);

    memcpy(pad, rcm_data.data() + i, corrected_bsize * sizeof(block));
    ferret->mitccrh.template hash<ot_bsize, 1>(pad);

    ferret->io->recv_data(corr_data, sizeof(T) * corrected_bsize);

    for (int64_t j = i; j < i + ot_bsize and j < length; ++j) {
      from_block(data[j], pad[j - i]);
      if (b[j]) {
        data[j] = corr_data[j - i] - data[j];
      }
    }
  }
}

// chosen additive message, chosen choice
// Sender chooses one message 'corr'. A correlation is defined by the addition
// function: f(x) = x + corr Sender receives a random message 'x' as output
// ('data0').
void SilentOT::send_ot_cam_cc(uint64_t* data0, const uint64_t* corr,
                              int64_t length, int l) {
  uint64_t modulo_mask = (1ULL << l) - 1;
  if (l == 64) {
    modulo_mask = -1;
  }
  std::unique_ptr<block> rcm_data(new block[length]);
  send_ot_rcm_cc(rcm_data.get(), length);

  block pad[2 * ot_bsize];
  auto y_size = static_cast<uint32_t>(ceil((ot_bsize * l) / (float(64))));
  uint32_t corrected_y_size;
  uint32_t corrected_bsize;
  uint64_t y[y_size];
  uint64_t corr_data[ot_bsize];

  for (int64_t i = 0; i < length; i += ot_bsize) {
    for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
      pad[2 * (j - i)] = rcm_data.get()[j];
      pad[2 * (j - i) + 1] = rcm_data.get()[j] ^ ferret->Delta;
    }

    ferret->mitccrh.template hash<ot_bsize, 2>(pad);

    for (int j = i; j < i + ot_bsize and j < length; ++j) {
      data0[j] = _mm_extract_epi64(pad[2 * (j - i)], 0) & modulo_mask;
      corr_data[j - i] =
          (corr[j] + data0[j] + _mm_extract_epi64(pad[2 * (j - i) + 1], 0)) &
          modulo_mask;
    }
    corrected_y_size =
        static_cast<uint32_t>(ceil((std::min(ot_bsize, length - i) * l) /
                                   (static_cast<float>(sizeof(uint64_t)) * 8)));
    corrected_bsize = std::min(ot_bsize, length - i);

    pack_cot_messages(y, corr_data, corrected_y_size, corrected_bsize, l);
    ferret->io->send_data(y, sizeof(uint64_t) * (corrected_y_size));
  }
}

// chosen additive message, chosen choice
// Receiver chooses a choice bit 'b', and
// receives 'x' if b = 0, and 'x + corr' if b = 1
void SilentOT::recv_ot_cam_cc(uint64_t* data, const bool* b, int64_t length,
                              int l) {
  uint64_t modulo_mask = (1ULL << l) - 1;
  if (l == 64) {
    modulo_mask = -1;
  }

  std::vector<block> rcm_data(length);
  recv_ot_rcm_cc(rcm_data.data(), b, length);

  block pad[ot_bsize];

  auto recvd_size = static_cast<uint32_t>(ceil((ot_bsize * l) / (float(64))));
  uint32_t corrected_recvd_size;
  uint32_t corrected_bsize;
  uint64_t corr_data[ot_bsize];
  uint64_t recvd[recvd_size];

  for (int64_t i = 0; i < length; i += ot_bsize) {
    corrected_recvd_size = static_cast<uint32_t>(
        ceil((std::min(ot_bsize, length - i) * l) / (float(64))));
    corrected_bsize = std::min(ot_bsize, length - i);

    memcpy(pad, rcm_data.data() + i,
           std::min(ot_bsize, length - i) * sizeof(block));
    ferret->mitccrh.template hash<ot_bsize, 1>(pad);

    ferret->io->recv_data(recvd, sizeof(uint64_t) * corrected_recvd_size);

    unpack_cot_messages(corr_data, recvd, corrected_bsize, l);

    for (int j = i; j < i + ot_bsize and j < length; ++j) {
      if (b[j]) {
        data[j] =
            (corr_data[j - i] - _mm_extract_epi64(pad[j - i], 0)) & modulo_mask;
      } else {
        data[j] = _mm_extract_epi64(pad[j - i], 0) & modulo_mask;
      }
    }
  }
}

// chosen message, chosen choice
void SilentOT::send_ot_cm_cc(const block* data0, const block* data1,
                             int64_t length) {
  std::vector<block> data(length);
  send_ot_rcm_cc(data.data(), length);

  block pad[2 * ot_bsize];
  for (int64_t i = 0; i < length; i += ot_bsize) {
    for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
      pad[2 * (j - i)] = data[j];
      pad[2 * (j - i) + 1] = data[j] ^ ferret->Delta;
    }
    // here, ferret depends on the template parameter "IO", making mitccrh also
    // dependent, hence we have to explicitly tell the compiler that "hash" is a
    // template function. See:
    // https://stackoverflow.com/questions/7397934/calling-template-function-within-template-class
    ferret->mitccrh.template hash<ot_bsize, 2>(pad);
    for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
      pad[2 * (j - i)] = pad[2 * (j - i)] ^ data0[j];
      pad[2 * (j - i) + 1] = pad[2 * (j - i) + 1] ^ data1[j];
    }
    ferret->io->send_data(pad,
                          2 * sizeof(block) * std::min(ot_bsize, length - i));
  }
}

// chosen message, chosen choice
void SilentOT::recv_ot_cm_cc(block* data, const bool* r, int64_t length) {
  recv_ot_rcm_cc(data, r, length);

  block res[2 * ot_bsize];
  block pad[ot_bsize];
  for (int64_t i = 0; i < length; i += ot_bsize) {
    memcpy(pad, data + i, std::min(ot_bsize, length - i) * sizeof(block));
    ferret->mitccrh.template hash<ot_bsize, 1>(pad);
    ferret->io->recv_data(res,
                          2 * sizeof(block) * std::min(ot_bsize, length - i));
    for (int64_t j = 0; j < ot_bsize and j < length - i; ++j) {
      data[i + j] = res[2 * j + r[i + j]] ^ pad[j];
    }
  }
}

// chosen message, chosen choice.
// Here, the 2nd dim of data is always 2. We use T** instead of T*[2] or two
// arguments of T*, in order to be general and compatible with the API of
// 1-out-of-N OT.
template <typename T>
void SilentOT::send_ot_cm_cc(T** data, int64_t length, int l) {
  std::vector<block> rcm_data(length);
  send_ot_rcm_cc(rcm_data.data(), length);

  block pad[2 * ot_bsize];
  auto y_size =
      static_cast<uint32_t>(ceil((2 * ot_bsize * l) / ((float)sizeof(T) * 8)));
  uint32_t corrected_y_size;
  uint32_t corrected_bsize;
  T y[y_size];

  for (int64_t i = 0; i < length; i += ot_bsize) {
    for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
      pad[2 * (j - i)] = rcm_data[j];
      pad[2 * (j - i) + 1] = rcm_data[j] ^ ferret->Delta;
    }
    // here, ferret depends on the template parameter "IO", making mitccrh also
    // dependent, hence we have to explicitly tell the compiler that "hash" is a
    // template function. See:
    // https://stackoverflow.com/questions/7397934/calling-template-function-within-template-class
    ferret->mitccrh.template hash<ot_bsize, 2>(pad);

    corrected_y_size =
        static_cast<uint32_t>(ceil((2 * std::min(ot_bsize, length - i) * l) /
                                   (static_cast<float>(sizeof(T) * 8))));
    corrected_bsize = std::min(ot_bsize, length - i);

    pack_ot_messages<T>(&y[0], data + i, pad, corrected_y_size, corrected_bsize,
                        l, 2);

    ferret->io->send_data(y, sizeof(T) * (corrected_y_size));
  }
}

// chosen message, chosen choice
// Here, r[i]'s value is always 0 or 1. We use uint8_t instead of bool, in order
// to be general and compatible with the API of 1-out-of-N OT.
template <typename T>
void SilentOT::recv_ot_cm_cc(T* data, const uint8_t* r, int64_t length, int l) {
  std::vector<block> rcm_data(length);
  recv_ot_rcm_cc(rcm_data.data(), reinterpret_cast<const bool*>(r), length);

  block pad[ot_bsize];

  auto recvd_size = static_cast<uint32_t>(
      ceil((2 * ot_bsize * l) / (static_cast<float>(sizeof(T) * 8))));
  uint32_t corrected_recvd_size;
  uint32_t corrected_bsize;
  T recvd[recvd_size];

  for (int64_t i = 0; i < length; i += ot_bsize) {
    corrected_recvd_size = (uint32_t)ceil(
        (2 * std::min(ot_bsize, length - i) * l) / ((float)sizeof(T) * 8));
    corrected_bsize = std::min(ot_bsize, length - i);

    ferret->io->recv_data(recvd, sizeof(T) * (corrected_recvd_size));

    memcpy(pad, rcm_data.data() + i,
           std::min(ot_bsize, length - i) * sizeof(block));
    ferret->mitccrh.template hash<ot_bsize, 1>(pad);

    unpack_ot_messages<T>(data + i, r + i, (T*)recvd, pad, corrected_bsize, l,
                          2);
  }
}

// random correlated message, chosen choice
void SilentOT::send_ot_rcm_cc(block* data0, int64_t length) {
  ferret->send_cot(data0, length);
}

// random correlated message, chosen choice
void SilentOT::recv_ot_rcm_cc(block* data, const bool* b, int64_t length) {
  ferret->recv_cot(data, b, length);
}

// random message, chosen choice
void SilentOT::send_ot_rm_cc(block* data0, block* data1, int64_t length) {
  send_ot_rcm_cc(data0, length);

  block pad[ot_bsize * 2];
  for (int64_t i = 0; i < length; i += ot_bsize) {
    for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
      pad[2 * (j - i)] = data0[j];
      pad[2 * (j - i) + 1] = data0[j] ^ ferret->Delta;
    }
    ferret->mitccrh.template hash<ot_bsize, 2>(pad);
    for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
      data0[j] = pad[2 * (j - i)];
      data1[j] = pad[2 * (j - i) + 1];
    }
  }
}

// random message, chosen choice
void SilentOT::recv_ot_rm_cc(block* data, const bool* r, int64_t length) {
  recv_ot_rcm_cc(data, r, length);

  block pad[ot_bsize];
  for (int64_t i = 0; i < length; i += ot_bsize) {
    memcpy(pad, data + i, std::min(ot_bsize, length - i) * sizeof(block));
    ferret->mitccrh.template hash<ot_bsize, 1>(pad);
    memcpy(data + i, pad, std::min(ot_bsize, length - i) * sizeof(block));
  }
}

// random message, random choice
void SilentOT::send_ot_rm_rc(block* data0, block* data1, int64_t length) {
  ferret->rcot(data0, length);

  block pad[ot_bsize * 2];
  for (int64_t i = 0; i < length; i += ot_bsize) {
    for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
      pad[2 * (j - i)] = data0[j];
      pad[2 * (j - i) + 1] = data0[j] ^ ferret->Delta;
    }
    ferret->mitccrh.template hash<ot_bsize, 2>(pad);
    for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
      data0[j] = pad[2 * (j - i)];
      data1[j] = pad[2 * (j - i) + 1];
    }
  }
}

// random message, random choice
void SilentOT::recv_ot_rm_rc(block* data, bool* r, int64_t length) {
  ferret->rcot(data, length);
  for (int64_t i = 0; i < length; i++) {
    r[i] = getLSB(data[i]);
  }

  block pad[ot_bsize];
  for (int64_t i = 0; i < length; i += ot_bsize) {
    memcpy(pad, data + i, std::min(ot_bsize, length - i) * sizeof(block));
    ferret->mitccrh.template hash<ot_bsize, 1>(pad);
    memcpy(data + i, pad, std::min(ot_bsize, length - i) * sizeof(block));
  }
}

// random message, random choice
template <typename T>
void SilentOT::send_ot_rm_rc(T* data0, T* data1, int64_t length, int l) {
  std::vector<block> rm_data0(length);
  std::vector<block> rm_data1(length);
  send_ot_rm_rc(rm_data0.data(), rm_data1.data(), length);

  T mask = static_cast<T>((1ULL << l) - 1ULL);

  for (int64_t i = 0; i < length; i++) {
    data0[i] = (static_cast<T>(_mm_extract_epi64(rm_data0[i], 0))) & mask;
    data1[i] = (static_cast<T>(_mm_extract_epi64(rm_data1[i], 0))) & mask;
  }
}

// random message, random choice
template <typename T>
void SilentOT::recv_ot_rm_rc(T* data, bool* r, int64_t length, int l) {
  std::vector<block> rm_data(length);
  recv_ot_rm_rc(rm_data.data(), r, length);

  T mask = static_cast<T>((1ULL << l) - 1ULL);

  for (int64_t i = 0; i < length; i++) {
    data[i] = (static_cast<T>(_mm_extract_epi64(rm_data[i], 0))) & mask;
  }
}

// chosen message, chosen choice.
// One-oo-N OT, where each message has l bits. Here, the 2nd dim of data is N.
template <typename T>
void SilentOT::send_ot_cm_cc(T** data, int64_t length, int N, int l) {
  int logN = static_cast<int>(ceil(log2(N)));

  std::vector<block> rm_data0(length * logN);
  std::vector<block> rm_data1(length * logN);
  send_ot_rm_cc(rm_data0.data(), rm_data1.data(), length * logN);

  block pad[ot_bsize * N];
  auto y_size =
      static_cast<uint32_t>(ceil((ot_bsize * N * l) / (float(sizeof(T) * 8))));
  uint32_t corrected_y_size;
  uint32_t corrected_bsize;
  T y[y_size];

  std::vector<block> hash_in0(N - 1);
  std::vector<block> hash_in1(N - 1);
  std::vector<block> hash_out(2 * N - 2);
  int idx = 0;
  for (int x = 0; x < logN; x++) {
    for (int y = 0; y < (1 << x); y++) {
      hash_in0[idx] = makeBlock(y, 0);
      hash_in1[idx] = makeBlock((1 << x) + y, 0);
      idx++;
    }
  }

  for (int64_t i = 0; i < length; i += ot_bsize) {
    memset(pad, 0, sizeof(block) * N * ot_bsize);
    for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
      mitccrh_exp.renew_ks(rm_data0.data() + j * logN, logN);
      mitccrh_exp.hash_exp(hash_out.data(), hash_in0.data(), logN);
      mitccrh_exp.renew_ks(rm_data1.data() + j * logN, logN);
      mitccrh_exp.hash_exp(hash_out.data() + N - 1, hash_in1.data(), logN);

      for (int64_t k = 0; k < N; k++) {
        idx = 0;
        for (int64_t s = 0; s < logN; s++) {
          int mask = (1 << s) - 1;
          int pref = k & mask;
          if ((k & (1 << s)) == 0) {
            pad[(j - i) * N + k] ^= hash_out[idx + pref];
          } else {
            pad[(j - i) * N + k] ^= hash_out[idx + N - 1 + pref];
          }
          idx += 1 << s;
        }
      }
    }

    corrected_y_size = static_cast<uint32_t>(ceil(
        (std::min(ot_bsize, length - i) * N * l) / (float(sizeof(T) * 8))));
    corrected_bsize = std::min(ot_bsize, length - i);

    pack_ot_messages<T>(&y[0], data + i, pad, corrected_y_size, corrected_bsize,
                        l, N);

    ferret->io->send_data(y, sizeof(T) * (corrected_y_size));
  }
}

// chosen message, chosen choice
// One-oo-N OT, where each message has l bits. Here, r[i]'s value is in [0, N).
template <typename T>
void SilentOT::recv_ot_cm_cc(T* data, const uint8_t* r, int64_t length, int N,
                             int l) {
  int logN = static_cast<int>(ceil(log2(N)));

  std::vector<block> rm_data(length * logN);
  std::vector<uint8_t> b_choices(length * logN);
  for (int64_t i = 0; i < length; i++) {
    for (int64_t j = 0; j < logN; j++) {
      b_choices[i * logN + j] = static_cast<uint8_t>((r[i] & (1 << j)) >> j);
    }
  }
  recv_ot_rm_cc(rm_data.data(), reinterpret_cast<const bool*>(b_choices.data()),
                length * logN);

  block pad[ot_bsize];

  auto recvd_size =
      static_cast<uint32_t>(ceil((ot_bsize * N * l) / (float(sizeof(T) * 8))));
  uint32_t corrected_recvd_size;
  uint32_t corrected_bsize;
  T recvd[recvd_size];

  std::vector<block> hash_out(logN);
  std::vector<block> hash_in(logN);

  for (int64_t i = 0; i < length; i += ot_bsize) {
    corrected_recvd_size = static_cast<uint32_t>(ceil(
        (std::min(ot_bsize, length - i) * N * l) / ((float)sizeof(T) * 8)));
    corrected_bsize = std::min(ot_bsize, length - i);

    ferret->io->recv_data(recvd, sizeof(T) * (corrected_recvd_size));

    memset(pad, 0, sizeof(block) * ot_bsize);
    for (int64_t j = i; j < std::min(i + ot_bsize, length); ++j) {
      for (int64_t s = 0; s < logN; s++) {
        hash_in[s] = makeBlock(r[j] & ((1 << (s + 1)) - 1), 0);
      }
      mitccrh_exp.renew_ks(&rm_data[j * logN], logN);
      mitccrh_exp.hash_single(hash_out.data(), hash_in.data(), logN);

      for (int64_t s = 0; s < logN; s++) {
        pad[j - i] ^= hash_out[s];
      }
    }

    unpack_ot_messages<T>(data + i, r + i, &recvd[0], pad, corrected_bsize, l,
                          N);
  }
}

template void SilentOT::send_ot_cam_cc<uint8_t>(uint8_t* data0,
                                                const uint8_t* corr,
                                                int64_t length);
template void SilentOT::send_ot_cam_cc<uint32_t>(uint32_t* data0,
                                                 const uint32_t* corr,
                                                 int64_t length);
template void SilentOT::send_ot_cam_cc<uint64_t>(uint64_t* data0,
                                                 const uint64_t* corr,
                                                 int64_t length);
template void SilentOT::send_ot_cam_cc<uint128_t>(uint128_t* data0,
                                                  const uint128_t* corr,
                                                  int64_t length);

template void SilentOT::recv_ot_cam_cc<uint8_t>(uint8_t* data, const bool* b,
                                                int64_t length);
template void SilentOT::recv_ot_cam_cc<uint32_t>(uint32_t* data, const bool* b,
                                                 int64_t length);
template void SilentOT::recv_ot_cam_cc<uint64_t>(uint64_t* data, const bool* b,
                                                 int64_t length);
template void SilentOT::recv_ot_cam_cc<uint128_t>(uint128_t* data,
                                                  const bool* b,
                                                  int64_t length);

template void SilentOT::send_ot_cm_cc<uint64_t>(uint64_t** data, int64_t length,
                                                int l);
template void SilentOT::send_ot_cm_cc<uint8_t>(uint8_t** data, int64_t length,
                                               int l);

template void SilentOT::recv_ot_cm_cc<uint64_t>(uint64_t* data,
                                                const uint8_t* r,
                                                int64_t length, int l);
template void SilentOT::recv_ot_cm_cc<uint8_t>(uint8_t* data, const uint8_t* r,
                                               int64_t length, int l);

template void SilentOT::send_ot_rm_rc<uint64_t>(uint64_t* data0,
                                                uint64_t* data1, int64_t length,
                                                int l);
template void SilentOT::send_ot_rm_rc<uint8_t>(uint8_t* data0, uint8_t* data1,
                                               int64_t length, int l);

template void SilentOT::recv_ot_rm_rc<uint64_t>(uint64_t* data, bool* r,
                                                int64_t length, int l);
template void SilentOT::recv_ot_rm_rc<uint8_t>(uint8_t* data, bool* r,
                                               int64_t length, int l);

template void SilentOT::send_ot_cm_cc<uint64_t>(uint64_t** data, int64_t length,
                                                int N, int l);
template void SilentOT::send_ot_cm_cc<uint8_t>(uint8_t** data, int64_t length,
                                               int N, int l);

template void SilentOT::recv_ot_cm_cc<uint64_t>(uint64_t* data,
                                                const uint8_t* r,
                                                int64_t length, int N, int l);
template void SilentOT::recv_ot_cm_cc<uint8_t>(uint8_t* data, const uint8_t* r,
                                               int64_t length, int N, int l);
}  // namespace spu
