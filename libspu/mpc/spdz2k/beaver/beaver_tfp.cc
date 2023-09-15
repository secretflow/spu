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
#include "libspu/mpc/spdz2k/commitment.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::spdz2k {

BeaverTfpUnsafe::BeaverTfpUnsafe(std::shared_ptr<yacl::link::Context> lctx)
    : seed_(yacl::crypto::SecureRandSeed()), counter_(0) {
  comm_ = std::make_unique<Communicator>(lctx);
  auto buf = yacl::SerializeUint128(seed_);
  std::vector<yacl::Buffer> all_bufs =
      yacl::link::Gather(lctx, buf, 0, "BEAVER_TFP:SYNC_SEEDS");

  if (comm_->getRank() == 0) {
    // Collects seeds from all parties.
    for (size_t rank = 0; rank < comm_->getWorldSize(); ++rank) {
      PrgSeed seed = yacl::DeserializeUint128(all_bufs[rank]);
      tp_.setSeed(rank, comm_->getWorldSize(), seed);
    }
  }
}

uint128_t BeaverTfpUnsafe::InitSpdzKey(FieldType field, size_t s) {
  PrgArrayDesc desc{};
  const int64_t size = 1;
  auto a = prgCreateArray(field, {size}, seed_, &counter_, &desc);

  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    NdArrayView<ring2k_t> _a(a);
    if (comm_->getRank() == 0) {
      auto t = tp_.adjustSpdzKey(desc);
      NdArrayView<ring2k_t> _t(t);
      global_key_ = yacl::crypto::SecureRandSeed();
      global_key_ &= (static_cast<uint128_t>(1) << s) - 1;

      _a[0] += global_key_ - _t[0];
    }

    spdz_key_ = _a[0];

    return spdz_key_;
  });
}

NdArrayRef BeaverTfpUnsafe::AuthArrayRef(const NdArrayRef& value,
                                         FieldType field, size_t k, size_t s) {
  auto [r, r_mac] = AuthCoinTossing(field, value.shape(), k, s);
  auto x_r =
      comm_->reduce(ReduceOp::ADD, ring_sub(value, r), 0, "auth_arrayref");

  if (comm_->getRank() == 0) {
    ring_add_(r_mac, ring_mul(x_r, global_key_));
  }

  return r_mac;
}

BeaverTfpUnsafe::Pair BeaverTfpUnsafe::AuthCoinTossing(FieldType field,
                                                       const Shape& shape,
                                                       size_t k, size_t s) {
  PrgArrayDesc desc{};
  PrgArrayDesc mac_desc{};

  auto x = prgCreateArray(field, shape, seed_, &counter_, &desc);
  auto x_mac = prgCreateArray(field, shape, seed_, &counter_, &mac_desc);

  if (comm_->getRank() == 0) {
    auto v = tp_.adjustAuthCoinTossing(desc, mac_desc, global_key_, k, s);
    x = v[0];
    x_mac = v[1];
  }

  return {x, x_mac};
}

BeaverTfpUnsafe::Pair BeaverTfpUnsafe::AuthRandBit(FieldType field,
                                                   const Shape& shape,
                                                   size_t /*k*/, size_t s) {
  PrgArrayDesc desc{};
  PrgArrayDesc mac_desc{};

  auto x = prgCreateArray(field, shape, seed_, &counter_, &desc);
  auto x_mac = prgCreateArray(field, shape, seed_, &counter_, &mac_desc);

  if (comm_->getRank() == 0) {
    auto v = tp_.adjustAuthRandBit(desc, mac_desc, global_key_, s);
    x = v[0];
    x_mac = v[1];
  }

  return {x, x_mac};
}

BeaverTfpUnsafe::Triple_Pair BeaverTfpUnsafe::AuthMul(FieldType field,
                                                      const Shape& shape,
                                                      size_t /*k*/,
                                                      size_t /*s*/) {
  std::vector<PrgArrayDesc> descs(3);
  std::vector<PrgArrayDesc> mac_descs(3);

  auto a = prgCreateArray(field, shape, seed_, &counter_, descs.data());
  auto b = prgCreateArray(field, shape, seed_, &counter_, &descs[1]);
  auto c = prgCreateArray(field, shape, seed_, &counter_, &descs[2]);

  auto a_mac = prgCreateArray(field, shape, seed_, &counter_, mac_descs.data());
  auto b_mac = prgCreateArray(field, shape, seed_, &counter_, &mac_descs[1]);
  auto c_mac = prgCreateArray(field, shape, seed_, &counter_, &mac_descs[2]);

  if (comm_->getRank() == 0) {
    auto v = tp_.adjustAuthMul(descs, mac_descs, global_key_);
    c = v[0];
    a_mac = v[1];
    b_mac = v[2];
    c_mac = v[3];
  }

  return {{a, b, c}, {a_mac, b_mac, c_mac}};
}

BeaverTfpUnsafe::Triple_Pair BeaverTfpUnsafe::AuthDot(FieldType field,
                                                      int64_t m, int64_t n,
                                                      int64_t k,
                                                      size_t /*k_bits*/,
                                                      size_t /*s_bits*/) {
  std::vector<PrgArrayDesc> descs(3);
  std::vector<PrgArrayDesc> mac_descs(3);

  auto a = prgCreateArray(field, {m, k}, seed_, &counter_, descs.data());
  auto b = prgCreateArray(field, {k, n}, seed_, &counter_, &descs[1]);
  auto c = prgCreateArray(field, {m, n}, seed_, &counter_, &descs[2]);

  auto a_mac =
      prgCreateArray(field, {m, k}, seed_, &counter_, mac_descs.data());
  auto b_mac = prgCreateArray(field, {k, n}, seed_, &counter_, &mac_descs[1]);
  auto c_mac = prgCreateArray(field, {m, n}, seed_, &counter_, &mac_descs[2]);

  if (comm_->getRank() == 0) {
    auto v = tp_.adjustAuthDot(descs, mac_descs, m, n, k, global_key_);
    c = v[0];
    a_mac = v[1];
    b_mac = v[2];
    c_mac = v[3];
  }

  return {{a, b, c}, {a_mac, b_mac, c_mac}};
}

BeaverTfpUnsafe::Triple_Pair BeaverTfpUnsafe::AuthAnd(FieldType field,
                                                      const Shape& shape,
                                                      size_t /*s*/) {
  std::vector<PrgArrayDesc> descs(3);
  std::vector<PrgArrayDesc> mac_descs(3);

  auto a = prgCreateArray(field, shape, seed_, &counter_, descs.data());
  auto b = prgCreateArray(field, shape, seed_, &counter_, &descs[1]);
  auto c = prgCreateArray(field, shape, seed_, &counter_, &descs[2]);

  auto a_mac = prgCreateArray(field, shape, seed_, &counter_, mac_descs.data());
  auto b_mac = prgCreateArray(field, shape, seed_, &counter_, &mac_descs[1]);
  auto c_mac = prgCreateArray(field, shape, seed_, &counter_, &mac_descs[2]);

  if (comm_->getRank() == 0) {
    auto v = tp_.adjustAuthAnd(descs, mac_descs, global_key_);
    c = v[0];
    a_mac = v[1];
    b_mac = v[2];
    c_mac = v[3];
  }

  return {{a, b, c}, {a_mac, b_mac, c_mac}};
}

BeaverTfpUnsafe::Pair_Pair BeaverTfpUnsafe::AuthTrunc(FieldType field,
                                                      const Shape& shape,
                                                      size_t bits, size_t k,
                                                      size_t s) {
  std::vector<PrgArrayDesc> descs(2);
  std::vector<PrgArrayDesc> mac_descs(2);

  auto a = prgCreateArray(field, shape, seed_, &counter_, descs.data());
  auto b = prgCreateArray(field, shape, seed_, &counter_, &descs[1]);
  auto a_mac = prgCreateArray(field, shape, seed_, &counter_, mac_descs.data());
  auto b_mac = prgCreateArray(field, shape, seed_, &counter_, &mac_descs[1]);

  if (comm_->getRank() == 0) {
    auto v = tp_.adjustAuthTrunc(descs, mac_descs, bits, global_key_, k, s);
    a = v[0];
    b = v[1];
    a_mac = v[2];
    b_mac = v[3];
  }

  return {{a, b}, {a_mac, b_mac}};
}

NdArrayRef BeaverTfpUnsafe::genPublCoin(FieldType field, int64_t numel) {
  NdArrayRef res(makeType<RingTy>(field), {numel});

  // generate new seed
  uint128_t self_pk = yacl::crypto::SecureRandSeed();
  std::vector<std::string> all_strs;

  std::string self_pk_str(reinterpret_cast<char*>(&self_pk), sizeof(self_pk));
  SPU_ENFORCE(commit_and_open(comm_->lctx(), self_pk_str, &all_strs));

  uint128_t public_seed = 0;
  for (const auto& str : all_strs) {
    uint128_t seed = *(reinterpret_cast<const uint128_t*>(str.data()));
    public_seed += seed;
  }

  auto kAesType = yacl::crypto::SymmetricCrypto::CryptoType::AES128_CTR;
  yacl::crypto::FillPRand(kAesType, public_seed, 0, 0,
                          absl::MakeSpan(res.data<char>(), res.buf()->size()));

  return res;
}

// Refer to:
// Procedure BatchCheck, 3.2 Batch MAC Checking with Random Linear
// Combinations, SPDZ2k: Efficient MPC mod 2k for Dishonest Majority
// - https://eprint.iacr.org/2018/482.pdf
//
// Open the value only
// Notice return { open_val , zero_mac = open_val * \sum spdz_key_ }
// the last kth bits in open_val is valid
std::pair<NdArrayRef, NdArrayRef> BeaverTfpUnsafe::BatchOpen(
    const NdArrayRef& value, const NdArrayRef& mac, size_t k, size_t s) {
  static constexpr char kBindName[] = "batch_open";
  SPU_ENFORCE(value.shape() == mac.shape());

  const auto field = value.eltype().as<Ring2k>()->field();

  auto [r_val, r_mac] = AuthCoinTossing(field, value.shape(), k, s);

  // Open the low k_bits only
  // value = value + r_val * 2^k
  // mac = mac + r_mac * 2^k
  auto masked_val = ring_add(value, ring_lshift(r_val, k));
  auto masked_mac = ring_add(mac, ring_lshift(r_mac, k));

  auto open_val = comm_->allReduce(ReduceOp::ADD, masked_val, kBindName);
  return {open_val, masked_mac};
}

// Refer to:
// Procedure BatchCheck, 3.2 Batch MAC Checking with Random Linear
// Combinations, SPDZ2k: Efficient MPC mod 2k for Dishonest Majority
// - https://eprint.iacr.org/2018/482.pdf
//
// Check the opened value only
bool BeaverTfpUnsafe::BatchMacCheck(const NdArrayRef& open_value,
                                    const NdArrayRef& mac, size_t k, size_t s) {
  const auto field = open_value.eltype().as<Ring2k>()->field();
  const int64_t numel = open_value.numel();
  const size_t mac_bits = k + s;

  auto* comm = comm_.get();
  const auto& lctx = comm_->lctx();
  const auto key = spdz_key_;

  // 1. get l public random values, compute plain y
  auto pub_r = genPublCoin(field, numel).reshape({1, numel});
  ring_bitmask_(pub_r, 0, s);

  // 2. check_value = pub_r * open_value
  //    check_mac = pub_r * mac
  auto check_value = ring_mmul(pub_r, open_value.reshape({numel, 1}));
  auto check_mac = ring_mmul(pub_r, mac.reshape({numel, 1}));

  // 3. compute z, commit and open z
  auto z = ring_sub(check_mac, ring_mul(check_value, key));

  std::string z_str(z.data<char>(), z.numel() * z.elsize());
  std::vector<std::string> z_strs;
  SPU_ENFORCE(commit_and_open(lctx, z_str, &z_strs));
  SPU_ENFORCE(z_strs.size() == comm->getWorldSize());

  // since the commit size in commit_and_open is independent with numel, we
  // ignore it
  comm->addCommStatsManually(1, 0);
  // since the random string size in commit_and_open is independent with numel,
  // we ignore it
  comm->addCommStatsManually(1,
                             z_str.size() / numel * (comm->getWorldSize() - 1));

  // 4. verify whether plain z is zero
  auto plain_z = ring_zeros(field, {1});
  for (size_t i = 0; i < comm->getWorldSize(); ++i) {
    const auto& _z_str = z_strs[i];
    auto mem = std::make_shared<yacl::Buffer>(_z_str.data(), _z_str.size());
    NdArrayRef a(mem, plain_z.eltype(),
                 {static_cast<int64_t>(_z_str.size() / SizeOf(field))}, {1}, 0);
    ring_add_(plain_z, a);
  }

  if (mac_bits != 0) {
    ring_bitmask_(plain_z, 0, mac_bits);
  }

  return ring_all_equal(plain_z, ring_zeros(field, {1}));
}

}  // namespace spu::mpc::spdz2k
