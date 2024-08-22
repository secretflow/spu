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

#include "libspu/mpc/cheetah/ot/ot_util.h"

#include "gtest/gtest.h"

#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah::test {

class OtUtilTest : public ::testing::TestWithParam<FieldType> {
  void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, OtUtilTest,
    testing::Values(FieldType::FM32, FieldType::FM64, FieldType::FM128),
    [](const testing::TestParamInfo<OtUtilTest::ParamType> &p) {
      return fmt::format("{}", p.param);
    });

TEST_P(OtUtilTest, ZipArray) {
  const int64_t n = 200;
  const auto field = GetParam();
  const size_t elsze = SizeOf(field);

  auto unzip = ring_zeros(field, {n});

  DISPATCH_ALL_FIELDS(field, [&]() {
    for (size_t bw : {1, 2, 4, 7, 15, 16}) {
      int64_t pack_load = elsze * 8 / bw;
      auto zip = ring_zeros(field, {(n + pack_load - 1) / pack_load});
      auto array = ring_rand(field, {n});
      auto inp = absl::MakeSpan(&array.at<ring2k_t>(0), array.numel());
      auto mask = makeBitsMask<ring2k_t>(bw);
      std::transform(inp.begin(), inp.end(), inp.data(),
                     [&](auto v) { return v & mask; });

      auto _zip = absl::MakeSpan(&zip.at<ring2k_t>(0), zip.numel());
      auto _unzip = absl::MakeSpan(&unzip.at<ring2k_t>(0), unzip.numel());
      (void)ZipArray<ring2k_t>(inp, bw, _zip);

      UnzipArray<ring2k_t>(_zip, bw, _unzip);

      for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(inp[i], _unzip[i]);
      }
    }
  });
}

TEST_P(OtUtilTest, ZipArrayBit) {
  const size_t n = 1000;
  const auto field = GetParam();

  auto unzip = ring_zeros(field, {n});

  DISPATCH_ALL_FIELDS(field, [&]() {
    const size_t elsze = SizeOf(field);
    for (size_t bw : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
      size_t width = elsze * 8;
      size_t pack_sze = CeilDiv(bw * n, width);

      auto zip = ring_zeros(field, {(int)pack_sze});
      auto array = ring_rand(field, {n});
      auto mask = makeBitsMask<ring2k_t>(bw);

      auto inp = absl::MakeSpan(&array.at<ring2k_t>(0), array.numel());
      auto _zip = absl::MakeSpan(&zip.at<ring2k_t>(0), zip.numel());
      auto _unzip = absl::MakeSpan(&unzip.at<ring2k_t>(0), unzip.numel());
      pforeach(0, array.numel(), [&](int64_t i) { inp[i] &= mask; });
      size_t zip_sze = ZipArrayBit<ring2k_t>(inp, bw, _zip);
      SPU_ENFORCE(zip_sze == pack_sze);

      if (((n * bw) % width) != 0) {
        // add some noises
        _zip[pack_sze - 1] |= (static_cast<ring2k_t>(1) << (width - 1));
      }

      UnzipArrayBit<ring2k_t>(_zip, bw, _unzip);

      for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(inp[i], _unzip[i]);
      }
    }
  });
}

template <typename T>
T makeBitsMask(size_t nbits) {
  size_t max = sizeof(T) * 8;
  if (nbits == 0) {
    nbits = max;
  }
  SPU_ENFORCE(nbits <= max);
  T mask = static_cast<T>(-1);
  if (nbits < max) {
    mask = (static_cast<T>(1) << nbits) - 1;
  }
  return mask;
}

void MaskArray(NdArrayRef array, FieldType field, size_t bw) {
  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> view(array);
    auto msk = makeBitsMask<ring2k_t>(bw);
    for (int64_t i = 0; i < view.numel(); ++i) {
      view[i] &= msk;
    }
  });
}

TEST_P(OtUtilTest, OpenShare_ADD) {
  const auto field = GetParam();
  Shape shape = {1000L};

  for (size_t bw_offset : {0, 15, 17}) {
    size_t bw = SizeOf(field) * 8 - bw_offset;
    NdArrayRef inp[2];
    utils::simulate(2, [&](std::shared_ptr<yacl::link::Context> ctx) {
      int rank = ctx->Rank();

      inp[rank] = ring_rand(field, shape);
      MaskArray(inp[rank], field, bw);

      auto conn = std::make_shared<Communicator>(ctx);
      auto opened = OpenShare(inp[rank], ReduceOp::ADD, bw, conn);
      if (rank == 0) return;
      auto expected = ring_add(inp[0], inp[1]);
      MaskArray(expected, field, bw);

      ASSERT_TRUE(std::memcmp(&opened.at<uint8_t>(0), &expected.at<uint8_t>(0),
                              opened.elsize() * opened.numel()) == 0);
    });
  }
}

TEST_P(OtUtilTest, OpenShare_XOR) {
  const auto field = GetParam();
  Shape shape = {1000L};

  for (size_t bw_offset : {0, 3, 15}) {
    size_t bw = SizeOf(field) * 8 - bw_offset;
    NdArrayRef inp[2];
    utils::simulate(2, [&](std::shared_ptr<yacl::link::Context> ctx) {
      int rank = ctx->Rank();

      inp[rank] = ring_rand(field, shape);
      MaskArray(inp[rank], field, bw);

      auto conn = std::make_shared<Communicator>(ctx);
      auto opened = OpenShare(inp[rank], ReduceOp::XOR, bw, conn);
      if (rank == 0) return;
      auto expected = ring_xor(inp[0], inp[1]);

      ASSERT_TRUE(std::memcmp(&opened.at<uint8_t>(0), &expected.at<uint8_t>(0),
                              opened.elsize() * opened.numel()) == 0);
    });
  }
}

}  // namespace spu::mpc::cheetah::test
