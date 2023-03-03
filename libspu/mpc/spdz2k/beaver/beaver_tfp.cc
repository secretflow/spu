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

#include "libspu/mpc/spdz2k/beaver/beaver_tfp.h"

#include <random>
#include <utility>

#include "yacl/crypto/utils/rand.h"
#include "yacl/link/link.h"
#include "yacl/utils/serialize.h"

#include "libspu/mpc/common/prg_tensor.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::spdz2k {

BeaverTfpUnsafe::BeaverTfpUnsafe(std::shared_ptr<yacl::link::Context> lctx)
    : lctx_(std::move(std::move(lctx))),
      seed_(yacl::crypto::RandSeed(true)),
      counter_(0) {
  auto buf = yacl::SerializeUint128(seed_);
  std::vector<yacl::Buffer> all_bufs =
      yacl::link::Gather(lctx_, buf, 0, "BEAVER_TFP:SYNC_SEEDS");

  if (lctx_->Rank() == 0) {
    // Collects seeds from all parties.
    for (size_t rank = 0; rank < lctx_->WorldSize(); ++rank) {
      PrgSeed seed = yacl::DeserializeUint128(all_bufs[rank]);
      tp_.setSeed(rank, lctx_->WorldSize(), seed);
    }
  }
}

uint128_t BeaverTfpUnsafe::GetSpdzKey(FieldType field, size_t s) {
  PrgArrayDesc desc{};
  const size_t size = 1;
  auto a = prgCreateArray(field, size, seed_, &counter_, &desc);

  if (lctx_->Rank() == 0) {
    auto t = tp_.adjustSpdzKey(desc);
    global_key_ = yacl::crypto::RandSeed(true);
    global_key_ &= (static_cast<uint128_t>(1) << s) - 1;
    a.at<uint128_t>(0) += global_key_ - t.at<uint128_t>(0);
  }

  return a.at<uint128_t>(0);
}

BeaverTfpUnsafe::Pair BeaverTfpUnsafe::AuthCoinTossing(FieldType field,
                                                       size_t size, size_t s) {
  PrgArrayDesc desc{};
  PrgArrayDesc mac_desc{};

  auto x = prgCreateArray(field, size, seed_, &counter_, &desc);
  auto x_mac = prgCreateArray(field, size, seed_, &counter_, &mac_desc);

  if (lctx_->Rank() == 0) {
    auto v = tp_.adjustAuthCoinTossing(desc, mac_desc, global_key_, s);
    x = v[0];
    x_mac = v[1];
  }

  return {x, x_mac};
}

BeaverTfpUnsafe::Triple_Pair BeaverTfpUnsafe::AuthMul(FieldType field,
                                                      size_t size) {
  std::vector<PrgArrayDesc> descs(3);
  std::vector<PrgArrayDesc> mac_descs(3);

  auto a = prgCreateArray(field, size, seed_, &counter_, descs.data());
  auto b = prgCreateArray(field, size, seed_, &counter_, &descs[1]);
  auto c = prgCreateArray(field, size, seed_, &counter_, &descs[2]);

  auto a_mac = prgCreateArray(field, size, seed_, &counter_, mac_descs.data());
  auto b_mac = prgCreateArray(field, size, seed_, &counter_, &mac_descs[1]);
  auto c_mac = prgCreateArray(field, size, seed_, &counter_, &mac_descs[2]);

  if (lctx_->Rank() == 0) {
    auto v = tp_.adjustAuthMul(descs, mac_descs, global_key_);
    c = v[0];
    a_mac = v[1];
    b_mac = v[2];
    c_mac = v[3];
  }

  return {{a, b, c}, {a_mac, b_mac, c_mac}};
}

BeaverTfpUnsafe::Triple_Pair BeaverTfpUnsafe::AuthDot(FieldType field, size_t m,
                                                      size_t n, size_t k) {
  std::vector<PrgArrayDesc> descs(3);
  std::vector<PrgArrayDesc> mac_descs(3);

  auto a = prgCreateArray(field, m * k, seed_, &counter_, descs.data());
  auto b = prgCreateArray(field, k * n, seed_, &counter_, &descs[1]);
  auto c = prgCreateArray(field, m * n, seed_, &counter_, &descs[2]);

  auto a_mac = prgCreateArray(field, m * k, seed_, &counter_, mac_descs.data());
  auto b_mac = prgCreateArray(field, k * n, seed_, &counter_, &mac_descs[1]);
  auto c_mac = prgCreateArray(field, m * n, seed_, &counter_, &mac_descs[2]);

  if (lctx_->Rank() == 0) {
    auto v = tp_.adjustAuthDot(descs, mac_descs, m, n, k, global_key_);
    c = v[0];
    a_mac = v[1];
    b_mac = v[2];
    c_mac = v[3];
  }

  return {{a, b, c}, {a_mac, b_mac, c_mac}};
}

BeaverTfpUnsafe::Pair_Pair BeaverTfpUnsafe::AuthTrunc(FieldType field,
                                                      size_t size,
                                                      size_t bits) {
  std::vector<PrgArrayDesc> descs(2);
  std::vector<PrgArrayDesc> mac_descs(2);

  auto a = prgCreateArray(field, size, seed_, &counter_, descs.data());
  auto b = prgCreateArray(field, size, seed_, &counter_, &descs[1]);
  auto a_mac = prgCreateArray(field, size, seed_, &counter_, mac_descs.data());
  auto b_mac = prgCreateArray(field, size, seed_, &counter_, &mac_descs[1]);

  if (lctx_->Rank() == 0) {
    auto v = tp_.adjustAuthTrunc(descs, mac_descs, bits, global_key_);
    b = v[0];
    a_mac = v[1];
    b_mac = v[2];
  }

  return {{a, b}, {a_mac, b_mac}};
}

}  // namespace spu::mpc::spdz2k
