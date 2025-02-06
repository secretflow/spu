// Copyright 2024 Ant Group Co., Ltd.
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

#include "yacl/crypto/tools/prg.h"

#include "libspu/core/ndarray_ref.h"

namespace spu::mpc {

// ref: Ciphers for MPC and FHE
// https://eprint.iacr.org/2016/687.pdf
class LowMC {
  using KeyType = uint128_t;  // key size should always be 128, at least >= 80

 public:
  // To prevent the user to set the wrong parameters, we decide the inner
  // parameters of LowMC insides. These parameters are deduced from the 5
  // attacks in the origin LowMC paper.
  // Note: currently, we only support encryption functionality.
  //
  // d: data complexity, the log2 of the numbers of data_blocks
  explicit LowMC(FieldType field, uint128_t seed, int64_t d,
                 uint64_t key_size = 128, bool need_decrypt = false);

  // plaintext set key procedure, debug only
  void set_key(KeyType key);

  ///
  /// encrypt/decrypt api for plaintext data, debug only now
  ///

  NdArrayRef encrypt(const NdArrayRef& plaintext);

  NdArrayRef decrypt(const NdArrayRef& ciphertext);

  std::vector<NdArrayRef> Lmat() const { return lin_matrices_; }

  std::vector<NdArrayRef> RoundConstants() const { return round_constants_; }

  std::vector<NdArrayRef> Kmat() const { return key_matrices_; }

  int64_t rounds() const { return rounds_; }

  int64_t number_of_boxes() const { return number_of_boxes_; }

  int64_t data_block_size() const { return block_size_; }

 private:
  // utils functions

  // S-boxes implementation with lookup table
  NdArrayRef Substitution(const NdArrayRef& data,
                          absl::Span<uint64_t const> sbox) const;

  // key filling functions
  void fill_matrixes(bool need_decrypt);

  // random blocks helper functions
  // generate public and replay rand array.
  NdArrayRef replay_ring_rand(FieldType field, const Shape& shape);

  // Note: To save memory, we compress k bits into a single uint64_t
  // or uint128_t number. So for n*k binary matrixes, we store it with an
  // shape (n,) NdArrayRef, each element (k bits) is a row of matrix.
  NdArrayRef get_pub_rand_blocks(FieldType field, int64_t n_blocks,
                                 int64_t desire_rank = -1);

  // some meta infos of the lowmc
  static constexpr int kSboxBits = 3;
  uint64_t block_size_;       // Data size in bits
  FieldType field_;           // field of data block
  uint64_t number_of_boxes_;  // Number of S-boxes in each round
  uint64_t identity_size_;    // Size of the identity part in the Sbox layer
  uint64_t key_size_;         // Key size in bits
  uint64_t rounds_;
  bool need_decrypt_;
  bool key_been_set_ = false;

  // random values related
  uint128_t seed_;  // seed to generate random matrixes and keys
  static constexpr yacl::crypto::SymmetricCrypto::CryptoType kCryptoType =
      yacl::crypto::SymmetricCrypto::CryptoType::AES128_ECB;
  uint128_t iv_ = 0;
  uint64_t cnt_ = 0;

  // inner matrixes and keys
  // Stores the binary matrices for each round.
  // each array, shape: (block_size_,)
  // each element is a ROW of matrix, i.e. block_size_ bits
  std::vector<NdArrayRef> lin_matrices_;
  // Stores the round constants
  // each array, shape: (1,)
  // each element is block_size_ bits
  std::vector<NdArrayRef> round_constants_;
  // Stores the matrices that generate the round keys
  // each array, shape: (block_size_,)
  // each element is a ROW of matrix, i.e. key_size_ bits
  std::vector<NdArrayRef> key_matrices_;
  // Stores the round keys
  // each array, shape: (1,)
  // each element is block_size_ bits
  std::vector<NdArrayRef> round_keys_;

  // some matrixes for decrypt, valid only for testing
  // Stores the inverses of LinMatrices
  // each array, shape: (block_size_,)
  // each element is a ROW of matrix, i.e. block_size_ bits
  std::vector<NdArrayRef> inv_lin_matrices_;

  // The Sbox and its inverse
  // The plaintext implementations of the Sbox and its inverse are based on
  // Look-Up tables.
  static constexpr std::array<uint64_t, 8> kSBox = {0x00, 0x01, 0x03, 0x06,
                                                    0x07, 0x04, 0x05, 0x02};
  static constexpr std::array<uint64_t, 8> kInvSBox = {0x00, 0x01, 0x07, 0x02,
                                                       0x05, 0x06, 0x03, 0x04};
};

}  // namespace spu::mpc
