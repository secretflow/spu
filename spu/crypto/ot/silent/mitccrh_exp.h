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

#include <stdio.h>

#include <stdexcept>
#include <string>

#include "emp-tool/utils/aes_opt.h"

#ifdef __aarch64__
#include "sse2neon.h"
#endif

namespace spu {
/*
 * With numKeys keys:
 * key 0 encrypts 1 block;
 * key 1 encrypts 2 blocks;
 * key 2 encrypts 4 blocks;
 * ...
 * key i encrypts 2^i blocks
 */
template <int numKeys>
static inline void ParaEncExp(block* blks, AES_KEY* keys) {
  block* first = blks;
  for (int i = 0; i < numKeys; ++i) {
    block K = keys[i].rd_key[0];
    int numEncs = 1 << i;
    for (int j = 0; j < numEncs; ++j) {
      *blks = *blks ^ K;
      ++blks;
    }
  }

  for (unsigned int r = 1; r < 10; ++r) {
    blks = first;
    for (int i = 0; i < numKeys; ++i) {
      block K = keys[i].rd_key[r];
      int numEncs = 1 << i;
      for (int j = 0; j < numEncs; ++j) {
        *blks = _mm_aesenc_si128(*blks, K);
        ++blks;
      }
    }
  }

  blks = first;
  for (int i = 0; i < numKeys; ++i) {
    block K = keys[i].rd_key[10];
    int numEncs = 1 << i;
    for (int j = 0; j < numEncs; ++j) {
      *blks = _mm_aesenclast_si128(*blks, K);
      ++blks;
    }
  }
}

/*
 * [REF] Implementation of "Better Concrete Security for Half-Gates Garbling (in
 * the Multi-Instance Setting)" https://eprint.iacr.org/2019/1168.pdf
 */

template <int BatchSize = 8>
class MITCCRHExp {
 public:
  AES_KEY scheduled_key[BatchSize];
  block keys[BatchSize];

  /**
  Renew with n explicitly specified AES keys.
  */
  void renew_ks(block* new_keys, int n) {
    for (int i = 0; i < n; ++i) keys[i] = new_keys[i];
    switch (n) {
      case 1:
        AES_opt_key_schedule<1>(keys, scheduled_key);
        break;
      case 2:
        AES_opt_key_schedule<2>(keys, scheduled_key);
        break;
      case 3:
        AES_opt_key_schedule<3>(keys, scheduled_key);
        break;
      case 4:
        AES_opt_key_schedule<4>(keys, scheduled_key);
        break;
      case 8:
        AES_opt_key_schedule<8>(keys, scheduled_key);
        break;
      default:
        throw std::invalid_argument(string("MITCCRH not implemented: ") +
                                    std::to_string(n));
    }
  }

  /**
  Hash (2^n - 1) blocks, using key i to hash 2^i blocks.
  */
  void hash_exp(block* out, const block* in, int n) {
    int n_blks = (1 << n) - 1;
    for (int i = 0; i < n_blks; ++i) out[i] = in[i];

    switch (n) {
      case 1:
        ParaEncExp<1>(out, scheduled_key);
        break;
      case 2:
        ParaEncExp<2>(out, scheduled_key);
        break;
      case 3:
        ParaEncExp<3>(out, scheduled_key);
        break;
      case 4:
        ParaEncExp<4>(out, scheduled_key);
        break;
      case 8:
        ParaEncExp<8>(out, scheduled_key);
        break;
      default:
        throw std::invalid_argument(string("MITCCRH not implemented: ") +
                                    std::to_string(n));
    }

    for (int i = 0; i < n_blks; ++i) out[i] = in[i] ^ out[i];
  }

  /**
  Hash n blocks, using each key to hash only one block.
  */
  void hash_single(block* out, const block* in, int n) {
    int n_blks = n;
    for (int i = 0; i < n_blks; ++i) out[i] = in[i];

    switch (n) {
      case 1:
        ParaEnc<1, 1>(out, scheduled_key);
        break;
      case 2:
        ParaEnc<2, 1>(out, scheduled_key);
        break;
      case 3:
        ParaEnc<3, 1>(out, scheduled_key);
        break;
      case 4:
        ParaEnc<4, 1>(out, scheduled_key);
        break;
      case 8:
        ParaEnc<8, 1>(out, scheduled_key);
        break;
      default:
        throw std::invalid_argument(string("MITCCRH not implemented: ") +
                                    std::to_string(n));
    }

    for (int i = 0; i < n_blks; ++i) out[i] = in[i] ^ out[i];
  }
};
}  // namespace spu
