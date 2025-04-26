// Copyright 2024 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "libspu/mpc/utils/waksman_net.h"

#include "gtest/gtest.h"

#include "libspu/mpc/utils/permute.h"

namespace spu::mpc {

IntegerPermutation makeRandomPermutation(size_t size, uint64_t seed) {
  uint64_t counter_ = 0;
  return IntegerPermutation(genRandomPerm(size, seed, &counter_));
}

void TestContent(const IntegerPermutation &p,
                 absl::Span<PermEleType const> data, PermEleType offset = 0) {
  for (size_t i = 0; i < p.size(); i++) {
    EXPECT_EQ(p[i + offset], data[i]);
  }
}

bool valid_as_waksman_routing(const IntegerPermutation &permutation,
                              const AsWaksmanRouting &routing) {
  const size_t num_packets = permutation.size();
  const size_t width = routing.size();
  auto neighbors = generate_as_waksman_topology(num_packets);

  IntegerPermutation curperm(num_packets);

  for (size_t column_idx = 0; column_idx < width; ++column_idx) {
    // each column gives one of permutation composed.
    // i.e. permutation = p_{k-1} o ... o p_0
    IntegerPermutation nextperm(num_packets);
    for (size_t packet_idx = 0; packet_idx < num_packets; ++packet_idx) {
      // the packet_idx in the next column
      size_t routed_packet_idx;
      if (neighbors[column_idx][packet_idx].first ==
          neighbors[column_idx][packet_idx].second) {
        // straight line
        routed_packet_idx = neighbors[column_idx][packet_idx].first;
      } else {
        // cross line
        auto it = routing[column_idx].find(packet_idx);
        auto it2 = routing[column_idx].find(packet_idx - 1);
        // can not find routing in both packet_idx and packet_idx-1
        SPU_ENFORCE((it != routing[column_idx].end()) ^
                    (it2 != routing[column_idx].end()));

        const bool switch_setting =
            (it != routing[column_idx].end() ? it->second : it2->second);

        routed_packet_idx =
            (switch_setting ? neighbors[column_idx][packet_idx].second
                            : neighbors[column_idx][packet_idx].first);
      }

      nextperm[routed_packet_idx] = curperm[packet_idx];
    }

    curperm = nextperm;
  }

  return (curperm == permutation.inverse());
}

TEST(IntegerPermutationTest, Work) {
  // 1. test constructor
  {
    auto p1 = IntegerPermutation(5);
    auto p2 = IntegerPermutation(2, 7);

    EXPECT_EQ(p1.size(), 5);
    TestContent(p1, {0, 1, 2, 3, 4});
    EXPECT_EQ(p2.size(), 6);
    TestContent(p2, {2, 3, 4, 5, 6, 7}, 2);
  }

  {
    // copy
    Index p = {0, 1, 2, 3, 4, 5};
    auto p1 = IntegerPermutation(p);
    TestContent(p1, p);

    p[0] = 10;
    EXPECT_EQ(p1[0], 0);
  }

  {
    // move
    auto p1 = IntegerPermutation({0, 1, 2, 3, 4, 5});
    TestContent(p1, {0, 1, 2, 3, 4, 5});
  }

  // 2. test get, set, slice
  {
    auto p1 = IntegerPermutation(2, 7);
    PermEleType offset = 2;

    EXPECT_EQ(p1[offset + 0], 2);
    EXPECT_EQ(p1[offset + 2], 4);

    p1[offset + 0] = 3;
    p1[offset + 1] = 2;
    EXPECT_EQ(p1[offset + 0], 3);
    EXPECT_EQ(p1[offset + 2], 4);

    auto p2 = p1.slice(4, 7);
    EXPECT_EQ(p2.size(), 4);
    TestContent(p2, {4, 5, 6, 7}, 4);
  }

  // 3. misc
  {
    auto p1 = IntegerPermutation(2, 7);
    PermEleType offset = 2;

    EXPECT_TRUE(p1.is_valid());

    auto flag = p1.next_permutation();
    flag = p1.next_permutation();
    EXPECT_TRUE(flag);
    EXPECT_TRUE(p1.is_valid());
    TestContent(p1, {2, 3, 4, 6, 5, 7}, offset);

    auto p2 = p1.inverse();
    TestContent(p2, {2, 3, 4, 6, 5, 7}, offset);
  }
}

TEST(AsWaksmanNetTest, Work) {
  // small permutation
  {
    // even case:
    const size_t n1 = 4;
    IntegerPermutation perm1(n1);

    do {
      const auto routing = get_as_waksman_routing(perm1);
      EXPECT_TRUE(valid_as_waksman_routing(perm1, routing));
    } while (perm1.next_permutation());
  }

  // odd case:
  const size_t n2 = 5;
  IntegerPermutation perm2(n2);

  do {
    auto routing = get_as_waksman_routing(perm2);
    EXPECT_TRUE(valid_as_waksman_routing(perm2, routing));
  } while (perm2.next_permutation());

  // large permutation
  {
    const size_t n = 1000;
    uint64_t seed = 107;
    auto perm = makeRandomPermutation(n, seed);
    auto routing = get_as_waksman_routing(perm);
    EXPECT_TRUE(valid_as_waksman_routing(perm, routing));
  }
}

}  // namespace spu::mpc
