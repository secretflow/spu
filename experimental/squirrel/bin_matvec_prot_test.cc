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

#include "experimental/squirrel/bin_matvec_prot.h"

#include <random>

#include "gtest/gtest.h"

#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace squirrel::test {

class BinMatVecProtTest
    : public ::testing::TestWithParam<
          std::tuple<spu::FieldType, std::tuple<size_t, size_t>>> {
 public:
  void PrepareBinaryMat(StlSparseMatrix &bin_mat, size_t rows, size_t cols,
                        uint64_t seed) {
    std::default_random_engine eng(seed);
    std::uniform_int_distribution<size_t> uniform(0, -1);

    size_t nnz = rows * cols;
    // sparsity 5%
    nnz = std::max(1UL, static_cast<size_t>(nnz * 0.05));
    // row major
    bin_mat.rows_data_.resize(rows);
    for (size_t i = 0; i < nnz; ++i) {
      size_t r = static_cast<size_t>(uniform(eng) % rows);
      size_t c = static_cast<size_t>(uniform(eng) % cols);
      bin_mat.rows_data_[r].insert(c);
    }

    bin_mat.cols_ = cols;
  }

  // plaintext BinMatVec
  template <typename T>
  std::vector<T> BinAccumuate(
      spu::NdArrayView<T> long_vector, const StlSparseMatrix &bin_mat,
      std::optional<absl::Span<const uint8_t>> indicator = std::nullopt) {
    size_t num_bins = bin_mat.rows();
    std::vector<T> feature_hist(num_bins, static_cast<T>(0));

    for (int64_t k = 0; k < bin_mat.rows(); ++k) {
      for (auto itr = bin_mat.iterate_row_begin(k);
           itr != bin_mat.iterate_row_end(k); ++itr) {
        if (!indicator || (*indicator)[*itr]) {
          feature_hist.at(k) += long_vector[*itr];
        }
      }
    }
    return feature_hist;
  }
};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, BinMatVecProtTest,
    testing::Combine(testing::Values(spu::FM32, spu::FM64, spu::FM128),
                     testing::Values(std::make_tuple<size_t>(11, 496),
                                     std::make_tuple<size_t>(421, 134),
                                     std::make_tuple<size_t>(32, 100))),
    [](const testing::TestParamInfo<BinMatVecProtTest::ParamType> &p) {
      return fmt::format("{}x{}x{}", std::get<0>(p.param),
                         std::get<0>(std::get<1>(p.param)),
                         std::get<1>(std::get<1>(p.param)));
    });

TEST_P(BinMatVecProtTest, Basic) {
  using namespace spu;
  using namespace spu::mpc;
  constexpr size_t kWorldSize = 2;

  FieldType field = std::get<0>(GetParam());
  int64_t dim_in = std::get<0>(std::get<1>(GetParam()));
  int64_t dim_out = std::get<1>(std::get<1>(GetParam()));

  StlSparseMatrix mat;
  PrepareBinaryMat(mat, dim_out, dim_in, 0);

  NdArrayRef vec_shr[2];
  vec_shr[0] = ring_rand(field, {dim_in})
                   .as(spu::makeType<spu::mpc::cheetah::AShrTy>(field));
  vec_shr[1] = ring_rand(field, {dim_in})
                   .as(spu::makeType<spu::mpc::cheetah::AShrTy>(field));

  NdArrayRef vec = ring_add(vec_shr[0], vec_shr[1]);

  NdArrayRef out_shr[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> lctx) {
    BinMatVecProtocol binmat_prot(SizeOf(field) * 8, lctx);
    if (0 == lctx->Rank()) {
      out_shr[0] = binmat_prot.Send(vec_shr[0], dim_out, dim_in);
    } else {
      out_shr[1] = binmat_prot.Recv(vec_shr[1], dim_out, dim_in, mat);
    }
  });
  NdArrayRef reveal = ring_add(out_shr[0], out_shr[1]);

  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _vec(vec);
    auto expected = BinAccumuate<ring2k_t>(_vec, mat);
    NdArrayView<ring2k_t> got(reveal);

    EXPECT_EQ(expected.size(), (size_t)got.numel());
    for (int64_t i = 0; i < dim_out; ++i) {
      EXPECT_NEAR(expected[i], got[i], 1);
    }
  });
}

TEST_P(BinMatVecProtTest, WithIndicator) {
  using namespace spu;
  using namespace spu::mpc;
  constexpr size_t kWorldSize = 2;

  FieldType field = std::get<0>(GetParam());
  int64_t dim_in = std::get<0>(std::get<1>(GetParam()));
  int64_t dim_out = std::get<1>(std::get<1>(GetParam()));

  StlSparseMatrix mat;
  PrepareBinaryMat(mat, dim_out, dim_in, 0);

  NdArrayRef vec_shr[2];
  vec_shr[0] = ring_rand(field, {dim_in})
                   .as(spu::makeType<spu::mpc::cheetah::AShrTy>(field));
  vec_shr[1] = ring_rand(field, {dim_in})
                   .as(spu::makeType<spu::mpc::cheetah::AShrTy>(field));
  std::vector<uint8_t> indicator(dim_in);
  std::default_random_engine rdv;
  std::uniform_int_distribution<uint8_t> dist(0, 10);
  // 80% density
  std::generate_n(indicator.data(), indicator.size(),
                  [&]() { return dist(rdv) > 2; });

  NdArrayRef vec = ring_add(vec_shr[0], vec_shr[1]);

  NdArrayRef out_shr[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> lctx) {
    BinMatVecProtocol binmat_prot(SizeOf(field) * 8, lctx);
    if (0 == lctx->Rank()) {
      out_shr[0] = binmat_prot.Send(vec_shr[0], dim_out, dim_in);
    } else {
      out_shr[1] = binmat_prot.Recv(vec_shr[1], dim_out, dim_in, mat,
                                    absl::MakeConstSpan(indicator));
    }
  });
  NdArrayRef reveal = ring_add(out_shr[0], out_shr[1]);

  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _vec(vec);
    auto expected =
        BinAccumuate<ring2k_t>(_vec, mat, absl::MakeConstSpan(indicator));
    NdArrayView<ring2k_t> got(reveal);

    EXPECT_EQ(expected.size(), (size_t)got.numel());
    for (int64_t i = 0; i < dim_out; ++i) {
      EXPECT_NEAR(expected[i], got[i], 1);
    }
  });
}

TEST_P(BinMatVecProtTest, EmptyMat) {
  using namespace spu;
  using namespace spu::mpc;
  constexpr size_t kWorldSize = 2;

  FieldType field = std::get<0>(GetParam());
  int64_t dim_in = std::get<0>(std::get<1>(GetParam()));
  int64_t dim_out = std::get<1>(std::get<1>(GetParam()));

  StlSparseMatrix mat;
  mat.rows_data_.resize(dim_out);
  mat.cols_ = dim_in;

  NdArrayRef vec_shr[2];
  vec_shr[0] = ring_rand(field, {dim_in})
                   .as(spu::makeType<spu::mpc::cheetah::AShrTy>(field));
  vec_shr[1] = ring_rand(field, {dim_in})
                   .as(spu::makeType<spu::mpc::cheetah::AShrTy>(field));

  NdArrayRef vec = ring_add(vec_shr[0], vec_shr[1]);

  NdArrayRef out_shr[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> lctx) {
    BinMatVecProtocol binmat_prot(SizeOf(field) * 8, lctx);
    if (0 == lctx->Rank()) {
      out_shr[0] = binmat_prot.Send(vec_shr[0], dim_out, dim_in);
    } else {
      out_shr[1] = binmat_prot.Recv(vec_shr[1], dim_out, dim_in, mat);
    }
  });
  NdArrayRef reveal = ring_add(out_shr[0], out_shr[1]);

  DISPATCH_ALL_FIELDS(field, "", [&]() {
    using sT = std::make_signed<ring2k_t>::type;
    NdArrayView<sT> _vec(vec);
    auto expected = BinAccumuate<sT>(_vec, mat);
    NdArrayView<sT> got(reveal);

    EXPECT_EQ(expected.size(), (size_t)got.numel());
    for (int64_t i = 0; i < dim_out; ++i) {
      EXPECT_NEAR(expected[i], got[i], 1);
    }
  });
}

TEST_P(BinMatVecProtTest, WithEmptyIndicator) {
  using namespace spu;
  using namespace spu::mpc;
  constexpr size_t kWorldSize = 2;

  FieldType field = std::get<0>(GetParam());
  int64_t dim_in = std::get<0>(std::get<1>(GetParam()));
  int64_t dim_out = std::get<1>(std::get<1>(GetParam()));

  StlSparseMatrix mat;
  PrepareBinaryMat(mat, dim_out, dim_in, 0);

  NdArrayRef vec_shr[2];
  vec_shr[0] = ring_rand(field, {dim_in})
                   .as(spu::makeType<spu::mpc::cheetah::AShrTy>(field));
  vec_shr[1] = ring_rand(field, {dim_in})
                   .as(spu::makeType<spu::mpc::cheetah::AShrTy>(field));
  std::vector<uint8_t> indicator(dim_in);
  std::default_random_engine rdv;
  std::uniform_int_distribution<uint8_t> dist(0, 10);
  // empty indicator
  std::fill_n(indicator.data(), indicator.size(), 0);

  NdArrayRef vec = ring_add(vec_shr[0], vec_shr[1]);

  NdArrayRef out_shr[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> lctx) {
    BinMatVecProtocol binmat_prot(SizeOf(field) * 8, lctx);
    if (0 == lctx->Rank()) {
      out_shr[0] = binmat_prot.Send(vec_shr[0], dim_out, dim_in);
    } else {
      out_shr[1] = binmat_prot.Recv(vec_shr[1], dim_out, dim_in, mat,
                                    absl::MakeConstSpan(indicator));
    }
  });
  NdArrayRef reveal = ring_add(out_shr[0], out_shr[1]);

  DISPATCH_ALL_FIELDS(field, "", [&]() {
    using sT = std::make_signed<ring2k_t>::type;
    NdArrayView<sT> _vec(vec);
    NdArrayView<sT> got(reveal);
    auto expected = BinAccumuate<sT>(_vec, mat, absl::MakeConstSpan(indicator));

    EXPECT_EQ(expected.size(), (size_t)got.numel());
    for (int64_t i = 0; i < dim_out; ++i) {
      EXPECT_NEAR(expected[i], got[i], 1);
    }
  });
}

}  // namespace squirrel::test
