// Copyright 2022 Ant Group Co., Ltd.
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

#include "libspu/mpc/cheetah/arith/batch_matmul.h"

#include "gtest/gtest.h"
#include "yacl/utils/elapsed_timer.h"

#include "libspu/core/type_util.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah {

class BatchMatMulTest
    : public ::testing::TestWithParam<std::tuple<FieldType, Shape4D, bool>> {};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, BatchMatMulTest,
    testing::Combine(
        testing::Values(FieldType::FM32, FieldType::FM64, FieldType::FM128),
        testing::Values(Shape4D{4, 1, 2048, 768}, Shape4D{4, 18, 768, 78},
                        Shape4D{4, 1024, 16, 16}),
        testing::Values(false)),
    [](const testing::TestParamInfo<BatchMatMulTest::ParamType>& p) {
      return fmt::format(
          "{}_{}x{}x{}x{}_{}", std::get<0>(p.param),
          std::get<0>(std::get<1>(p.param)), std::get<1>(std::get<1>(p.param)),
          std::get<2>(std::get<1>(p.param)), std::get<3>(std::get<1>(p.param)),
          std::get<2>(p.param) ? "Approx" : "Exact");
    });

TEST_P(BatchMatMulTest, Basic) {
  size_t kWorldSize = 2;
  auto field = std::get<0>(GetParam());
  auto dim4 = std::get<1>(GetParam());
  bool allow_approx = std::get<2>(GetParam());

  std::vector<NdArrayRef> input(kWorldSize);
  NdArrayRef weight;

  // TODO: inputs should be multi-dim tensors.
  input[0] = ring_rand(field, {dim4[0], dim4[1], dim4[2]});
  input[1] = ring_rand(field, {dim4[0], dim4[1], dim4[2]});
  weight = ring_rand(field, {dim4[0], dim4[2], dim4[3]});

  std::vector<NdArrayRef> result(kWorldSize);
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> lctx) {
    int rank = lctx->Rank();
    auto matmul = std::make_shared<BatchMatMul>(lctx, allow_approx);
    matmul->LazyInitKeys(field);

    if (rank == 0) {
      // client
      result[rank] = matmul->MatMulClient(input[0], dim4);
    } else {
      // server
      result[rank] = matmul->MatMulServer(input[1], weight, dim4);
    }
  });

  auto computed = ring_add(result[0], result[1]);

  // compute expected result
  NdArrayRef expected;
  expected = ring_zeros(field, {dim4[0], dim4[1], dim4[3]});
  NdArrayRef released_input;
  released_input = ring_add(input[0], input[1]);

  for (int64_t b = 0; b < dim4[0]; b++) {
    auto lhs =
        released_input.slice({b, 0, 0}, {b + 1, dim4[1], dim4[2]}, {1, 1, 1})
            .reshape({dim4[1], dim4[2]});
    auto rhs = weight.slice({b, 0, 0}, {b + 1, dim4[2], dim4[3]}, {1, 1, 1})
                   .reshape({dim4[2], dim4[3]});
    auto slice = expected.slice({b, 0, 0}, {b + 1, dim4[1], dim4[3]}, {1, 1, 1})
                     .reshape({dim4[1], dim4[3]});
    ring_mmul_(slice, lhs, rhs);
  }

  EXPECT_EQ(expected.numel(), computed.numel());
  DISPATCH_ALL_FIELDS(field, [&]() {
    auto e = NdArrayView<ring2k_t>(expected);
    auto c = NdArrayView<ring2k_t>(computed);

    for (auto idx = 0; idx < expected.numel(); idx++) {
      // only exact version supported now.
      SPU_ENFORCE(e[idx] == c[idx], "expected {}, got {}, at {}", e[idx],
                  c[idx], idx);
    }
  });
}

}  // namespace spu::mpc::cheetah