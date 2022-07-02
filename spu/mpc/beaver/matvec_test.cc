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

#include "spu/mpc/beaver/matvec.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "seal/seal.h"

#include "spu/core/xt_helper.h"
#include "spu/mpc/util/ring_ops.h"
#include "spu/mpc/util/seal_help.h"

namespace spu::mpc {
class MatVecTest : public ::testing::TestWithParam<std::tuple<size_t, size_t>> {
 public:
  MatVecTest() {
    using namespace seal;
    size_t poly_deg = 128;  // NOTE(juhou) use a tiny parameter for testing
    EncryptionParameters parms(scheme_type::bfv);
    std::vector<int> modulus_bits{59, 55, 60};
    parms.set_poly_modulus_degree(poly_deg);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_deg, modulus_bits));
    parms.set_plain_modulus(CoeffModulus::Create(poly_deg, {22})[0]);
    plain_ = parms.plain_modulus();

    context_ = std::make_shared<SEALContext>(parms, true, sec_level_type::none);

    KeyGenerator keygen(*context_);
    sk_ = std::make_shared<SecretKey>(keygen.secret_key());
    sym_encryptor_ = std::make_shared<Encryptor>(*context_, *sk_);
    decryptor_ = std::make_shared<Decryptor>(*context_, *sk_);
    evaluator_ = std::make_shared<Evaluator>(*context_);

    rot_keys_ = std::make_shared<GaloisKeys>();
    keygen.create_galois_keys(*rot_keys_);
  }

  template <typename T>
  void RandomPlain(T* dst, size_t sze) {
    std::mt19937 rdv(std::time(0));
    std::uniform_int_distribution<T> uniform(0, plain_.value() - 1);
    std::generate_n(dst, sze, [&]() { return uniform(rdv); });
  }

  seal::Modulus plain_;
  std::shared_ptr<seal::SEALContext> context_;

  std::shared_ptr<seal::SecretKey> sk_;
  std::shared_ptr<seal::Encryptor> sym_encryptor_;
  std::shared_ptr<seal::Evaluator> evaluator_;
  std::shared_ptr<seal::Decryptor> decryptor_;
  std::shared_ptr<seal::GaloisKeys> rot_keys_;
};

TEST_P(MatVecTest, Simple) {
  FieldType field = FieldType::FM64;
  size_t nrows = std::get<0>(GetParam());
  size_t ncols = std::get<1>(GetParam());

  auto mat = ring_zeros(field, nrows * ncols);
  auto vec = ring_zeros(field, ncols);
  auto xvec = xt_mutable_adapt<uint64_t>(vec);
  auto xmat = xt_mutable_adapt<uint64_t>(mat);
  xmat.reshape({nrows, ncols});
  RandomPlain(xvec.data(), xvec.size());
  RandomPlain(xmat.data(), xmat.size());

  ArrayRef _ground = ring_mmul(mat, vec, nrows, 1, ncols);
  auto ground = xt_mutable_adapt<uint64_t>(_ground);
  std::transform(ground.begin(), ground.end(), ground.begin(), [&](int64_t u) {
    auto uu = static_cast<uint64_t>(u);
    return seal::util::barrett_reduce_64(uu, plain_);
  });

  // Protocol starts
  MatVecHelper::MatViewMeta meta;
  meta.is_transposed = false;
  meta.num_rows = nrows;
  meta.num_cols = ncols;
  meta.row_start = 0;
  meta.col_start = 0;
  meta.row_extent = nrows;
  meta.col_extent = ncols;

  MatVecProtocol matvec_prot(*rot_keys_, *context_);
  yasl::Buffer buffer = matvec_prot.EncryptVector(vec, meta, *sym_encryptor_);

  seal::Ciphertext enc_vec;
  DecodeSEALObject(buffer, *context_, &enc_vec);

  std::vector<seal::Plaintext> ecd_diags;
  matvec_prot.EncodeSubMatrix(mat, meta, &ecd_diags);

  seal::Ciphertext matvec_ct;
  absl::Span<const seal::Plaintext> ecd_mat(ecd_diags.data(), ecd_diags.size());

  matvec_prot.Compute(enc_vec, ecd_mat, meta, &matvec_ct);

  seal::Plaintext pt;
  decryptor_->decrypt(matvec_ct, pt);
  std::vector<uint64_t> slots;
  seal::BatchEncoder encoder(*context_);
  encoder.decode(pt, slots);
  // Protocol ends

  std::vector<int64_t> shape{(int64_t)ground.size()};
  std::vector<int64_t> strides{1};
  auto computed = xt::adapt(slots.data(), ground.size(), xt::no_ownership(),
                            shape, strides);
  EXPECT_EQ(ground, computed);
}

TEST_P(MatVecTest, Transpose) {
  FieldType field = FieldType::FM64;
  size_t nrows = std::get<0>(GetParam());
  size_t ncols = std::get<1>(GetParam());

  auto mat = ring_zeros(field, nrows * ncols);
  auto vec = ring_zeros(field, nrows);
  auto xvec = xt_mutable_adapt<uint64_t>(vec);
  auto xmat = xt_mutable_adapt<uint64_t>(mat);
  xmat.reshape({nrows, ncols});
  RandomPlain(xvec.data(), xvec.size());
  RandomPlain(xmat.data(), xmat.size());

  ArrayRef _ground = [&]() -> ArrayRef {
    auto matT = ring_zeros(field, xmat.size());

    auto xmatT = xt::eval(xt::transpose(xmat));
    std::copy_n(xmatT.begin(), xmat.size(),
                reinterpret_cast<uint64_t*>(matT.data()));
    return ring_mmul(matT, vec, ncols, 1, nrows);
  }();

  auto ground = xt_mutable_adapt<uint64_t>(_ground);
  std::transform(ground.begin(), ground.end(), ground.begin(), [&](int64_t u) {
    auto uu = static_cast<uint64_t>(u);
    return seal::util::barrett_reduce_64(uu, plain_);
  });

  // Protocol starts
  MatVecHelper::MatViewMeta meta;
  meta.is_transposed = true;
  meta.num_rows = nrows;
  meta.num_cols = ncols;
  meta.row_start = 0;
  meta.col_start = 0;
  meta.row_extent = nrows;
  meta.col_extent = ncols;

  MatVecProtocol matvec_prot(*rot_keys_, *context_);
  yasl::Buffer buffer = matvec_prot.EncryptVector(vec, meta, *sym_encryptor_);

  seal::Ciphertext enc_vec;
  DecodeSEALObject(buffer, *context_, &enc_vec);

  std::vector<seal::Plaintext> ecd_diags;
  matvec_prot.EncodeSubMatrix(mat, meta, &ecd_diags);

  seal::Ciphertext matvec_ct;
  absl::Span<const seal::Plaintext> ecd_mat(ecd_diags.data(), ecd_diags.size());
  matvec_prot.Compute(enc_vec, ecd_mat, meta, &matvec_ct);

  seal::Plaintext pt;
  decryptor_->decrypt(matvec_ct, pt);
  std::vector<uint64_t> slots;
  seal::BatchEncoder encoder(*context_);
  encoder.decode(pt, slots);
  // Protocol ends

  std::vector<int64_t> shape{(int64_t)ground.size()};
  std::vector<int64_t> strides{1};
  auto computed = xt::adapt(slots.data(), ground.size(), xt::no_ownership(),
                            shape, strides);
  EXPECT_EQ(ground, computed);
}

INSTANTIATE_TEST_SUITE_P(MatVecTestInstance, MatVecTest,
                         testing::Values(
                             // square matrix
                             std::make_tuple(16, 16), std::make_tuple(32, 32),
                             // tall matrix
                             std::make_tuple(32, 8), std::make_tuple(128, 8),
                             // short matrix
                             std::make_tuple(8, 32), std::make_tuple(8, 128),
                             // half load
                             std::make_tuple(64, 64),
                             // full load
                             std::make_tuple(64, 128),
                             // full load
                             std::make_tuple(128, 64),
                             // some boundary cases
                             std::make_tuple(1, 1), std::make_tuple(128, 1),
                             std::make_tuple(1, 128)));

}  // namespace spu::mpc
