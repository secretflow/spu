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

#include "spu/psi/core/kkrt_psi.h"

#include <future>
#include <iostream>

#include "gtest/gtest.h"
#include "yasl/base/exception.h"
#include "yasl/crypto/hash_util.h"
#include "yasl/link/test_util.h"

struct TestParams {
  std::vector<uint128_t> items_a;
  std::vector<uint128_t> items_b;
};

namespace spu::psi {

void KkrtPsiSend(const std::shared_ptr<yasl::link::Context>& link_ctx,
                 const std::vector<uint128_t>& items_hash) {
  yasl::BaseRecvOptions recv_opts;

  GetKkrtOtSenderOptions(link_ctx, 512, &recv_opts);

  return KkrtPsiSend(link_ctx, recv_opts, items_hash);
}

std::vector<std::size_t> KkrtPsiRecv(
    const std::shared_ptr<yasl::link::Context>& link_ctx,
    const std::vector<uint128_t>& items_hash) {
  yasl::BaseSendOptions send_opts;

  GetKkrtOtReceiverOptions(link_ctx, 512, &send_opts);

  return KkrtPsiRecv(link_ctx, send_opts, items_hash);
}

std::vector<size_t> GetIntersection(const TestParams& params) {
  std::set<uint128_t> seta(params.items_a.begin(), params.items_a.end());
  std::vector<size_t> ret;
  std::set<size_t> ret_set;
  size_t idx = 0;
  for (const auto& s : params.items_b) {
    if (seta.count(s) != 0) {
      ret_set.insert(idx);
    }
    idx++;
  }
  ret.assign(ret_set.begin(), ret_set.end());
  return ret;
}

class KkrtPsiTest : public testing::TestWithParam<TestParams> {};

TEST_P(KkrtPsiTest, Works) {
  auto params = GetParam();
  const int kWorldSize = 2;
  auto contexts = yasl::link::test::SetupWorld(kWorldSize);

  std::future<void> kkrtPsi_sender =
      std::async([&] { return KkrtPsiSend(contexts[0], params.items_a); });
  std::future<std::vector<std::size_t>> kkrtPsi_receiver =
      std::async([&] { return KkrtPsiRecv(contexts[1], params.items_b); });

  if ((params.items_a.size() == 0) || (params.items_b.size() == 0)) {
    EXPECT_THROW(kkrtPsi_sender.get(), ::yasl::EnforceNotMet);
    EXPECT_THROW(kkrtPsi_receiver.get(), ::yasl::EnforceNotMet);
    return;
  }
  kkrtPsi_sender.get();
  auto psi_idx_result = kkrtPsi_receiver.get();

  std::sort(psi_idx_result.begin(), psi_idx_result.end());

  auto intersection = GetIntersection(params);

  EXPECT_EQ(psi_idx_result, intersection);
}

std::vector<uint128_t> CreateRangeItems(size_t begin, size_t size) {
  std::vector<uint128_t> ret;
  for (size_t i = 0; i < size; i++) {
    ret.push_back(yasl::crypto::Blake3_128(std::to_string(begin + i)));
  }
  return ret;
}

INSTANTIATE_TEST_SUITE_P(
    Works_Instances, KkrtPsiTest,
    testing::Values(
        TestParams{
            {yasl::crypto::Blake3_128("a"), yasl::crypto::Blake3_128("b")},
            {yasl::crypto::Blake3_128("b"), yasl::crypto::Blake3_128("c")}},  //
        //
        TestParams{
            {yasl::crypto::Blake3_128("a"), yasl::crypto::Blake3_128("b")},
            {yasl::crypto::Blake3_128("c"), yasl::crypto::Blake3_128("d")}},
        // size not equal
        TestParams{
            {yasl::crypto::Blake3_128("a"), yasl::crypto::Blake3_128("b"),
             yasl::crypto::Blake3_128("c")},
            {yasl::crypto::Blake3_128("c"), yasl::crypto::Blake3_128("d")}},  //
        TestParams{
            {yasl::crypto::Blake3_128("a"), yasl::crypto::Blake3_128("b")},
            {yasl::crypto::Blake3_128("b"), yasl::crypto::Blake3_128("c"),
             yasl::crypto::Blake3_128("d")}},  //
        //
        TestParams{{}, {yasl::crypto::Blake3_128("a")}},  //
        //
        TestParams{{yasl::crypto::Blake3_128("a")}, {}},  //
        // less than one batch
        TestParams{CreateRangeItems(0, 1000), CreateRangeItems(1, 1000)},  //
        TestParams{CreateRangeItems(0, 1000), CreateRangeItems(1, 800)},   //
        TestParams{CreateRangeItems(0, 800), CreateRangeItems(1, 1000)},   //
        // exactly one batch
        TestParams{CreateRangeItems(0, 1024), CreateRangeItems(1, 1024)},  //
        // more than one batch
        TestParams{CreateRangeItems(0, 4095), CreateRangeItems(1, 4095)},  //
        //
        TestParams{{}, {}}  //
        ));

}  // namespace spu::psi
