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

#include "libspu/mpc/utils/lowmc.h"

#include "libspu/mpc/utils/lowmc_utils.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc {

namespace {

template <typename T>
bool get_bit(const T x, int i) {
  return (x >> i) & (1);
}

// Some linear algebra helper functions
uint64_t rank_of_matrix(const NdArrayRef& matrix) {
  SPU_ENFORCE(matrix.shape().size() == 1, "matrix should be a 1-D array");

  const auto n_rows = static_cast<uint64_t>(matrix.numel());
  auto mat = matrix.clone();
  const auto field = mat.eltype().as<RingTy>()->field();

  // Do Gaussian elimination, and count the non-zero rows
  uint64_t row = 0;

  DISPATCH_ALL_FIELDS(field, [&]() {
    using block_type = ring2k_t;
    NdArrayView<block_type> _mat(mat);

    // can be `block_size_` or `key_size_`, column size of matrix
    const auto size = sizeof(block_type) * 8;
    const auto max_rank = std::min(n_rows, size);

    // we try to transform matrix to its upper triangular form
    for (uint64_t col = 1; col <= size; ++col) {
      // if the pivot is zero, then find the first non-zero row and swap it
      if (!get_bit(_mat[row], size - col)) {
        uint64_t r = row;
        while (r < n_rows && !get_bit(_mat[r], size - col)) {
          ++r;
        }
        // all rows in this column are zero, skip it
        if (r >= n_rows) {
          continue;
        } else {
          auto temp = _mat[row];
          _mat[row] = _mat[r];
          _mat[r] = temp;
        }
      }
      for (uint64_t i = row + 1; i < n_rows; ++i) {
        if (get_bit(_mat[i], size - col)) {
          _mat[i] ^= _mat[row];
        }
      }
      ++row;
      if (row == max_rank) {
        break;
      }
    }
  });

  return row;
}

// Computing the inv of matrix without checking the rank of matrix by
// Gaussian elimination algorithm: [M | I] -> [I | inv(M)]
NdArrayRef invert_matrix(const NdArrayRef& matrix) {
  SPU_ENFORCE(matrix.shape().size() == 1, "matrix should be a 1-D array");

  const auto n_rows = static_cast<uint64_t>(matrix.numel());
  auto mat = matrix.clone();
  const auto field = mat.eltype().as<RingTy>()->field();

  auto inv_mat = NdArrayRef(matrix.eltype(), matrix.shape());

  DISPATCH_ALL_FIELDS(field, [&]() {
    using block_type = ring2k_t;
    const auto size = sizeof(block_type) * 8;
    SPU_ENFORCE(n_rows == size, "Not a square matrix.");

    NdArrayView<block_type> _mat(mat);
    NdArrayView<block_type> _inv_mat(inv_mat);

    // init inv_mat as identity matrix
    pforeach(0, n_rows, [&](int64_t idx) {  //
      _inv_mat[idx] = (static_cast<block_type>(1) << idx);
    });

    // Transform to upper triangular matrix first
    uint64_t row = 0;
    for (uint64_t col = 0; col < size; ++col) {
      // if the pivot is zero, then find the first non-zero row and swap it
      if (!get_bit(_mat[row], col)) {
        uint64_t r = row + 1;
        while (r < n_rows && !get_bit(_mat[r], col)) {
          ++r;
        }
        if (r >= n_rows) {
          continue;
        } else {
          auto temp = _mat[row];
          _mat[row] = _mat[r];
          _mat[r] = temp;

          temp = _inv_mat[row];
          _inv_mat[row] = _inv_mat[r];
          _inv_mat[r] = temp;
        }
      }
      for (uint64_t i = row + 1; i < n_rows; ++i) {
        if (get_bit(_mat[i], col)) {
          _mat[i] ^= _mat[row];
          _inv_mat[i] ^= _inv_mat[row];
        }
      }
      ++row;
    }

    // Transform to identity matrix
    for (uint64_t col = size; col > 0; --col) {
      for (uint64_t r = 0; r < col - 1; ++r) {
        if (get_bit(_mat[r], col - 1)) {
          _mat[r] ^= _mat[col - 1];
          _inv_mat[r] ^= _inv_mat[col - 1];
        }
      }
    }
  });

  return inv_mat;
}

}  // namespace

/// public api implementation

LowMC::LowMC(FieldType field, uint128_t seed, int64_t d, uint64_t key_size,
             bool need_decrypt) {
  SPU_ENFORCE(key_size == 128, "key size should always be 128 now");

  int64_t n_boxes;
  int64_t rounds;
  if (field == FM32) {
    SPU_ENFORCE(d < 32,
                "Support at most 2^32 blocks to encrypt for 32-bit blocks.");
    // d=20 or d=30 has the same parameter setting.
    n_boxes = 9;
    rounds = 15;
  } else if (field == FM64) {
    switch (d) {
      case 20:
        n_boxes = 15;
        rounds = 11;
        break;
      case 30:
        n_boxes = 13;
        rounds = 12;
        break;
      case 40:
        n_boxes = 13;
        rounds = 13;
        break;
      default:
        SPU_THROW("Not supported data complexity.");
    }
  } else if (field == FM128) {
    switch (d) {
      case 20:
        n_boxes = 25;
        rounds = 10;
        break;
      case 30:
        n_boxes = 25;
        rounds = 11;
        break;
      case 40:
        n_boxes = 25;
        rounds = 12;
        break;
      default:
        SPU_THROW("Not supported data complexity.");
    }
  } else {
    SPU_THROW("Should not be here.");
  }

  field_ = field;
  seed_ = seed;
  number_of_boxes_ = n_boxes;
  rounds_ = rounds;
  key_size_ = key_size;
  need_decrypt_ = need_decrypt;
  block_size_ = SizeOf(field) * 8;
  SPU_ENFORCE(block_size_ <= 128,
              "data size should be no more than 128 bits now.");

  // S-boxes of LowMC has 3 bits
  identity_size_ = block_size_ - number_of_boxes_ * kSboxBits;

  // fill some key-irrelevant random matrixes
  fill_matrixes(need_decrypt);
}

void LowMC::set_key(KeyType key) {
  if (key_been_set_) {
    return;
  }

  round_keys_ = generate_round_keys(key_matrices_, key, rounds_, field_);
  key_been_set_ = true;
}

NdArrayRef LowMC::encrypt(const NdArrayRef& plaintext) {
  SPU_ENFORCE(key_been_set_, "key not set.");
  SPU_ENFORCE(plaintext.eltype().as<RingTy>()->field() == field_,
              "field mismatch");
  const auto& shape = plaintext.shape();

  // 1. key whiten
  auto c = ring_xor(plaintext, round_keys_[0].broadcast_to(shape, {}));

  // 2. round loop
  for (uint64_t r = 1; r <= rounds_; r++) {
    // S-boxes
    c = Substitution(c, kSBox);

    // affine layer
    c = dot_product_gf2(lin_matrices_[r - 1], c, field_);
    ring_xor_(c, round_constants_[r - 1].broadcast_to(shape, {}));

    // round key xor
    ring_xor_(c, round_keys_[r].broadcast_to(shape, {}));
  }

  return c;
}

NdArrayRef LowMC::decrypt(const NdArrayRef& ciphertext) {
  SPU_ENFORCE(key_been_set_, "key not set.");
  SPU_ENFORCE(ciphertext.eltype().as<RingTy>()->field() == field_,
              "field mismatch");
  const auto& shape = ciphertext.shape();

  // just the inverse procedure of encrypt
  auto c = ciphertext;
  for (uint64_t r = rounds_; r > 0; r--) {
    ring_xor_(c, round_keys_[r].broadcast_to(shape, {}));

    ring_xor_(c, round_constants_[r - 1].broadcast_to(shape, {}));
    c = dot_product_gf2(inv_lin_matrices_[r - 1], c, field_);
    c = Substitution(c, kInvSBox);
  }

  ring_xor_(c, round_keys_[0].broadcast_to(shape, {}));

  return c;
}

/// private api implementation

NdArrayRef LowMC::Substitution(const NdArrayRef& data,
                               absl::Span<uint64_t const> sbox) const {
  NdArrayRef ret(data.eltype(), data.shape());

  DISPATCH_ALL_FIELDS(ret.eltype().as<RingTy>()->field(), [&]() {
    using block_type = ring2k_t;
    NdArrayView<block_type> _data(data);
    NdArrayView<block_type> _ret(ret);

    pforeach(0, data.numel(), [&](int64_t idx) {
      block_type tmp = 0;

      // Get the identity part of the data
      tmp ^= (_data[idx] >> (3 * number_of_boxes_));

      // Get the rest through the Sboxes
      for (uint64_t i = 1; i <= number_of_boxes_; ++i) {
        tmp <<= 3;
        auto ind = ((_data[idx] >> 3 * (number_of_boxes_ - i)) & 0x7);
        tmp ^= static_cast<block_type>(sbox[ind]);
      }

      _ret[idx] = tmp;
    });
  });

  return ret;
}

void LowMC::fill_matrixes(bool need_decrypt) {
  // 1. create Lmatrixes
  lin_matrices_.reserve(rounds_);
  // -1 means no rank checking
  int64_t desire_rank = -1;
  if (need_decrypt) {
    inv_lin_matrices_.reserve(rounds_);
    // Note: we force block_size_ <= key_size_ = 128, so we can just use the
    // same ranks for all Lmatices and key matrices.
    desire_rank = block_size_;
  }

  for (uint64_t i = 0; i < rounds_; i++) {
    auto mat = get_pub_rand_blocks(field_, block_size_, desire_rank);
    lin_matrices_.push_back(mat);

    if (need_decrypt) {
      inv_lin_matrices_.push_back(invert_matrix(mat));
    }
  }

  // 2. create round constants
  round_constants_.reserve(rounds_);
  for (uint64_t i = 0; i < rounds_; i++) {
    round_constants_.push_back(get_pub_rand_blocks(field_, 1));
  }

  // 3. create key matrices
  key_matrices_.reserve(rounds_ + 1);  // first element is for initial whiten
  for (uint64_t i = 0; i < rounds_ + 1; i++) {
    // we force the key_size = 128 for safety consideration.
    key_matrices_.push_back(
        get_pub_rand_blocks(FM128, block_size_, desire_rank));
  }
}

NdArrayRef LowMC::replay_ring_rand(FieldType field, const Shape& shape) {
  NdArrayRef res(makeType<RingTy>(field), shape);

  cnt_ = yacl::crypto::FillPRand(
      kCryptoType, seed_, iv_, cnt_,
      absl::MakeSpan(res.data<char>(), res.buf()->size()));

  return res;
}

NdArrayRef LowMC::get_pub_rand_blocks(FieldType field, int64_t n_blocks,
                                      int64_t desire_rank) {
  const auto ring_ty = makeType<RingTy>(field);
  auto rand = replay_ring_rand(field, {n_blocks});

  // check the rank for the inverse process (debug only now)
  if (desire_rank > 0) {
    // The simple constant rounds algorithm to generate invertible or full
    // row-rank matrixes:
    // e.g. For nxn matrices M, we just fill M
    // with random bits, and it's not hard to prove that: P(det(M) != 0) =
    // (1-1/2) * (1-1/4) * ... * (1-1/2^n) ~= 0.2888 (when n->inf), so the
    // expected repeat times are no more than 4.
    while (rank_of_matrix(rand) != static_cast<uint64_t>(desire_rank)) {
      rand = replay_ring_rand(field, {n_blocks});
    }
  }

  return rand.as(ring_ty);
}

}  // namespace spu::mpc
