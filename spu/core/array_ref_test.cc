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

#include "spu/core/array_ref.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace spu {

TEST(ArrayRefTest, Empty) {
  ArrayRef a;
  EXPECT_EQ(a.numel(), 0);
  EXPECT_EQ(a.stride(), 0);
  EXPECT_EQ(a.elsize(), 0);
  EXPECT_EQ(a.offset(), 0);
}

TEST(ArrayRefTest, Simple) {
  constexpr size_t kBufSize = 100;
  auto mem = std::make_shared<yasl::Buffer>(kBufSize);

  ArrayRef a(mem, I32, 100 / 4, 1, 0);

  EXPECT_EQ(a.numel(), kBufSize / I32.size());
  EXPECT_EQ(a.stride(), 1);
  EXPECT_EQ(a.elsize(), I32.size());
  EXPECT_EQ(a.offset(), 0);

  ArrayRef b(mem, I32, 10, 2, 20);
  EXPECT_EQ(b.numel(), 10);
  EXPECT_EQ(b.stride(), 2);
  EXPECT_EQ(b.elsize(), 4);
  EXPECT_EQ(b.offset(), 20);

  EXPECT_EQ(a.buf(), b.buf());

  // test packing
  {
    EXPECT_EQ(hasSimdTrait<ArrayRef>::value, true);
    EXPECT_EQ(hasSimdTrait<size_t>::value, false);
    //
    SimdTrait<ArrayRef>::PackInfo pi;
    std::vector<ArrayRef> l = {a, b};
    ArrayRef packed = SimdTrait<ArrayRef>::pack(l.begin(), l.end(), pi);

    EXPECT_EQ(packed.numel(), a.numel() + b.numel());
    EXPECT_EQ(packed.stride(), 1);
    EXPECT_EQ(packed.elsize(), I32.size());
    EXPECT_EQ(packed.offset(), 0);

    for (int64_t idx = 0; idx < packed.numel(); idx++) {
      if (idx < a.numel()) {
        EXPECT_EQ(memcmp(&packed.at(idx), &a.at(idx), a.elsize()), 0);
      } else {
        EXPECT_EQ(memcmp(&packed.at(idx), &b.at(idx - a.numel()), b.elsize()),
                  0);
      }
    }

    std::vector<ArrayRef> unpacked;
    SimdTrait<ArrayRef>::unpack(packed, std::back_inserter(unpacked), pi);

    EXPECT_EQ(unpacked.size(), 2);
    EXPECT_EQ(unpacked[0].numel(), a.numel());
    EXPECT_EQ(unpacked[0].eltype(), a.eltype());
    EXPECT_EQ(unpacked[1].numel(), b.numel());
    EXPECT_EQ(unpacked[1].eltype(), b.eltype());

    for (int64_t idx = 0; idx < a.numel(); idx++) {
      EXPECT_EQ(memcmp(&unpacked[0].at(idx), &a.at(idx), a.elsize()), 0);
    }

    for (int64_t idx = 0; idx < b.numel(); idx++) {
      EXPECT_EQ(memcmp(&unpacked[1].at(idx), &b.at(idx), b.elsize()), 0);
    }
  }
}

TEST(ArrayRefTest, Slice) {
  constexpr size_t kBufSize = 100;
  auto mem = std::make_shared<yasl::Buffer>(kBufSize);

  // GIVEN
  ArrayRef a(mem, I32, 100 / 4, 1, 0);
  EXPECT_EQ(a.numel(), kBufSize / I32.size());
  EXPECT_EQ(a.stride(), 1);
  EXPECT_EQ(a.elsize(), I32.size());
  EXPECT_EQ(a.offset(), 0);
  for (int32_t i = 0; i < 25; i++) {
    a.at<int32_t>(i) = i;
  }

  // WHEN
  ArrayRef b = a.slice(10, 20);
  // THEN
  {
    EXPECT_EQ(a.buf(), b.buf());
    EXPECT_EQ(b.numel(), 10);
    EXPECT_EQ(b.stride(), 1);
    EXPECT_EQ(b.elsize(), 4);
    EXPECT_EQ(b.offset(), 40);
    for (int32_t i = 0; i < 10; i++) {
      EXPECT_EQ(b.at<int32_t>(i), 10 + i);
    }
  }

  // WHEN
  ArrayRef c = b.slice(5, 10, 2);
  // THEN
  {
    EXPECT_EQ(a.buf(), c.buf());
    EXPECT_EQ(c.numel(), 3);
    EXPECT_EQ(c.stride(), 2);
    EXPECT_EQ(c.elsize(), 4);
    EXPECT_EQ(c.offset(), 60);
    for (int32_t i = 0; i < 3; i++) {
      EXPECT_EQ(c.at<int32_t>(i), 15 + 2 * i);
    }
  }
}

TEST(ArrayRefTest, Strides) {
  // Make 3 element, strides = 2 array
  ArrayRef a(std::make_shared<yasl::Buffer>(6 * sizeof(int32_t)),
             makePtType(PT_I32), 3, 2, 0);

  EXPECT_EQ(a.numel(), 3);
  EXPECT_EQ(a.stride(), 2);
  EXPECT_EQ(a.elsize(), sizeof(int32_t));

  // Fill array with 0 1 2 3 4 5
  std::iota(static_cast<int32_t*>(a.data()),
            reinterpret_cast<int32_t*>(reinterpret_cast<std::byte*>(a.data()) +
                                       a.buf()->size()),
            0);

  EXPECT_EQ(a.at<int32_t>(0), 0);
  EXPECT_EQ(a.at<int32_t>(1), 2);
  EXPECT_EQ(a.at<int32_t>(2), 4);

  // Make a compact clone
  auto b = a.clone();

  EXPECT_TRUE(b.isCompact());
  EXPECT_EQ(b.numel(), 3);
  EXPECT_EQ(b.stride(), 1);

  EXPECT_EQ(b.at<int32_t>(0), 0);
  EXPECT_EQ(b.at<int32_t>(1), 2);
  EXPECT_EQ(b.at<int32_t>(2), 4);
}

}  // namespace spu
