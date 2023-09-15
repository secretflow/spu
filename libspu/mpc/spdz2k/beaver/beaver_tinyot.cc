// Copyright 2023 Ant Group Co., Ltd.
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

#include "libspu/mpc/spdz2k/beaver/beaver_tinyot.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <random>

#include "yacl/base/dynamic_bitset.h"
#include "yacl/crypto/primitives/ot/ot_store.h"
#include "yacl/crypto/tools/prg.h"
#include "yacl/crypto/utils/rand.h"
#include "yacl/link/link.h"
#include "yacl/utils/matrix_utils.h"
#include "yacl/utils/serialize.h"

#include "libspu/mpc/common/prg_tensor.h"
#include "libspu/mpc/spdz2k/commitment.h"
#include "libspu/mpc/spdz2k/ot/kos_ote.h"
#include "libspu/mpc/spdz2k/ot/tiny_ot.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::spdz2k {
namespace {

// sqrt2k algorithm find the smallest root for residue in ring2K
// Polynomial time algorithm to find the root
// reference
// https://github.com/sagemath/sage/blob/2114066f877a28b7473bf9242b1bb11931f3ec3e/src/sage/rings/finite_rings/integer_mod.pyx#L3943
uint128_t inline Sqrt2k(uint128_t residue, uint128_t bits) {
  uint128_t x = 1;
  uint128_t N = residue;
  SPU_ENFORCE((N & 7) == 1);
  while (x < 8 && (N & 31) != ((x * x) & 31)) {
    x += 2;
  }
  uint128_t t = (N - x * x) >> 5;
  for (size_t i = 4; i < bits; ++i) {
    if (t & 1) {
      x |= (uint128_t)1 << i;
      t -= x - ((uint128_t)1 << (i - 1));
    }
    t >>= 1;
  }

  uint128_t half_mod = (uint128_t)1 << (bits - 1);
  uint128_t mask = half_mod + (half_mod - 1);
  auto l = [&mask](uint128_t val) { return val & mask; };
  return std::min({l(x), l(x + half_mod), l(-x), l(-x + half_mod)});
}

NdArrayRef ring_sqrt2k(const NdArrayRef& x, size_t bits = 0) {
  const auto field = x.eltype().as<Ring2k>()->field();
  const auto numel = x.numel();
  if (bits == 0) {
    bits = SizeOf(field) * 8;
  }

  auto ret = ring_zeros(field, x.shape());
  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using U = std::make_unsigned<ring2k_t>::type;
    NdArrayView<U> _ret(ret);
    NdArrayView<U> _x(x);
    yacl::parallel_for(0, numel, 4096, [&](int64_t beg, int64_t end) {
      for (int64_t idx = beg; idx < end; ++idx) {
        _ret[idx] = Sqrt2k(_x[idx], bits);
      }
    });
  });
  return ret;
}

// reference https://github.com/data61/MP-SPDZ/blob/master/Math/Z2k.hpp
uint128_t inline Invert2k(const uint128_t value, const size_t bits) {
  SPU_ENFORCE((value & 1) == 1);
  uint128_t ret = 1;
  for (size_t i = 0; i < bits; ++i) {
    if (!((value * ret >> i) & 1)) {
      ret += (uint128_t)1 << i;
    }
  }
  return ret;
}

NdArrayRef ring_inv2k(const NdArrayRef& x, size_t bits = 0) {
  const auto field = x.eltype().as<Ring2k>()->field();
  const auto numel = x.numel();
  if (bits == 0) {
    bits = SizeOf(field) * 8;
  }

  auto ret = ring_zeros(field, x.shape());
  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using U = std::make_unsigned<ring2k_t>::type;
    NdArrayView<U> _ret(ret);
    NdArrayView<U> _x(x);
    yacl::parallel_for(0, numel, 4096, [&](int64_t beg, int64_t end) {
      for (int64_t idx = beg; idx < end; ++idx) {
        _ret[idx] = Invert2k(_x[idx], bits);
      }
    });
  });
  return ret;
}

std::vector<bool> ring_cast_vector_boolean(const NdArrayRef& x) {
  const auto field = x.eltype().as<Ring2k>()->field();

  std::vector<bool> res(x.numel());
  DISPATCH_ALL_FIELDS(field, "RingOps", [&]() {
    NdArrayView<ring2k_t> _x(x);
    yacl::parallel_for(0, x.numel(), 4096, [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        res[i] = static_cast<bool>(_x[i] & 0x1);
      }
    });
  });
  return res;
}

}  // namespace

BeaverTinyOt::BeaverTinyOt(std::shared_ptr<yacl::link::Context> lctx)
    : seed_(yacl::crypto::SecureRandSeed()) {
  comm_ = std::make_shared<Communicator>(lctx);
  prg_state_ = std::make_shared<PrgState>(lctx);
  spdz2k_ot_primitives_ = std::make_shared<BasicOTProtocols>(comm_);

  auto buf = yacl::SerializeUint128(seed_);
  std::vector<yacl::Buffer> all_bufs =
      yacl::link::Gather(lctx, buf, 0, "BEAVER_TINY:SYNC_SEEDS");

  if (comm_->getRank() == 0) {
    // Collects seeds from all parties.
    for (size_t rank = 0; rank < comm_->getWorldSize(); ++rank) {
      PrgSeed seed = yacl::DeserializeUint128(all_bufs[rank]);
      tp_.setSeed(rank, comm_->getWorldSize(), seed);
    }
  }

  auto recv_opts_choices = yacl::dynamic_bitset<uint128_t>(kappa_);
  auto recv_opts_blocks = std::vector<uint128_t>(kappa_);

  auto send_opts_blocks = std::vector<std::array<uint128_t, 2>>(kappa_);

  if (comm_->getRank() == 0) {
    yacl::crypto::BaseOtRecv(comm_->lctx(), recv_opts_choices,
                             absl::MakeSpan(recv_opts_blocks));
    yacl::crypto::BaseOtSend(comm_->lctx(), absl::MakeSpan(send_opts_blocks));
  } else {
    yacl::crypto::BaseOtSend(comm_->lctx(), absl::MakeSpan(send_opts_blocks));
    yacl::crypto::BaseOtRecv(comm_->lctx(), recv_opts_choices,
                             absl::MakeSpan(recv_opts_blocks));
  }

  recv_opts_ = std::make_shared<yacl::crypto::OtRecvStore>(
      yacl::crypto::MakeOtRecvStore(recv_opts_choices, recv_opts_blocks));

  send_opts_ = std::make_shared<yacl::crypto::OtSendStore>(
      yacl::crypto::MakeOtSendStore(send_opts_blocks));

  // the choices of BaseOT options would be the delta in delta OT
  // which means that delta is the "key" in TinyOT
  tinyot_key_ = 0;
  for (size_t k = 0; k < kappa_; ++k) {
    if (recv_opts_->GetChoice(k)) {
      tinyot_key_ |= (uint128_t)1 << k;
    }
  }
}

uint128_t BeaverTinyOt::InitSpdzKey(FieldType, size_t s) {
  spdz_key_ = yacl::crypto::SecureRandSeed();
  spdz_key_ &= ((uint128_t)1 << s) - 1;
  return spdz_key_;
}

// Refer to:
// Fig. 11 Protocol for authenticating secret-shared values
// SPDZ2k: Efficient MPC mod 2k for Dishonest Majority
// - https://eprint.iacr.org/2018/482.pdf
NdArrayRef BeaverTinyOt::AuthArrayRef(const NdArrayRef& x, FieldType field,
                                      size_t k, size_t s) {
  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using T = ring2k_t;

    // 1. l_ = max(l, r + s, 2s)
    SPDLOG_DEBUG("AuthArrayRef start with numel {}", x.numel());
    const int l = k + s;
    const int r = k;
    int l_ = std::max(l, r + static_cast<int>(s));
    l_ = std::max(l_, 2 * static_cast<int>(s));
    l_ = std::min(l_, static_cast<int>(SizeOf(field) * 8));
    SPU_ENFORCE(l_ >= static_cast<int>(SizeOf(field) * 8), "k = s");

    // 2. sample random masks
    int64_t t = x.numel();
    int64_t new_numel = t + 1;
    NdArrayRef x_hat(x.eltype(), {new_numel});
    auto x_mask = ring_rand(field, {1});

    NdArrayView<T> _x_hat(x_hat);
    NdArrayView<T> _x(x);
    NdArrayView<T> _x_mask(x_mask);
    for (int i = 0; i < t; ++i) {
      _x_hat[i] = _x[i];
    }
    _x_hat[t] = _x_mask[0];

    // 3. every pair calls vole && 4. receives vole output
    size_t WorldSize = comm_->getWorldSize();
    size_t rank = comm_->getRank();

    std::vector<NdArrayRef> a, b;
    auto alpha = ring_mul(ring_ones(field, {new_numel}), spdz_key_);
    for (size_t i = 0; i < WorldSize; ++i) {
      for (size_t j = 0; j < WorldSize; ++j) {
        if (i == j) {
          continue;
        }

        if (i == rank) {
          auto tmp = voleRecv(field, alpha);
          a.emplace_back(tmp);
        }
        if (j == rank) {
          auto tmp = voleSend(field, x_hat);
          b.emplace_back(tmp);
        }
      }
    }

    // 5. each party defines the MAC share
    auto a_b = ring_zeros(field, {new_numel});
    for (size_t i = 0; i < WorldSize - 1; ++i) {
      ring_add_(a_b, ring_sub(a[i], b[i]));
    }

    auto m = ring_add(ring_mul(x_hat, spdz_key_), a_b);
    NdArrayView<T> _m(m);

    // Consistency check
    // 6. get l public random values
    auto pub_r = prg_state_->genPubl(field, {new_numel});
    NdArrayView<T> _pub_r(pub_r);
    std::vector<int> rv;
    size_t numel = x.numel();
    for (size_t i = 0; i < numel; ++i) {
      rv.emplace_back(_pub_r[i]);
    }
    rv.emplace_back(1);

    // 7. caculate x_angle && 8. caculate m_angle
    T x_angle = 0;
    T m_angle = 0;
    for (int64_t i = 0; i < new_numel; ++i) {
      // x_hat, not x
      x_angle += rv[i] * _x_hat[i];
      m_angle += rv[i] * _m[i];
    }

    auto x_angle_sum =
        comm_->allReduce<T, std::plus>(std::vector{x_angle}, "allReduce x_ref");

    // 9. commmit and open
    auto z = m_angle - x_angle_sum[0] * spdz_key_;
    std::string z_str((char*)&z, sizeof(z));
    std::vector<std::string> recv_strs;
    SPU_ENFORCE(commit_and_open(comm_->lctx(), z_str, &recv_strs));
    SPU_ENFORCE(recv_strs.size() == WorldSize);

    // 10. check
    T plain_z = 0;
    for (const auto& str : recv_strs) {
      T t = *(reinterpret_cast<const T*>(str.data()));
      plain_z += t;
    }

    SPU_ENFORCE(plain_z == 0);

    // 11. output MAC share
    return m.slice({0}, {m.numel() - 1}, {1}).reshape(x.shape());
  });
}

BeaverTinyOt::Pair BeaverTinyOt::AuthCoinTossing(FieldType field,
                                                 const Shape& shape, size_t k,
                                                 size_t s) {
  auto rand = ring_rand(field, shape);
  auto mac = AuthArrayRef(rand, field, k, s);
  return {rand, mac};
}

// Refer to:
// New Primitives for Actively-Secure MPC over Rings with Applications to
// Private Machine Learning.
// Figure 2: TinyOT share to binary SPDZ2K share conversion.
// - https://eprint.iacr.org/2019/599.pdf
BeaverTinyOt::Triple_Pair BeaverTinyOt::AuthAnd(FieldType field,
                                                const Shape& shape, size_t s) {
  const size_t elsize = SizeOf(field);
  const int64_t tinyot_num = shape.numel();
  // extra sigma bits = 64
  const int64_t sigma = 64;

  auto [auth_a, auth_b, auth_c] =
      TinyMul(comm_, send_opts_, recv_opts_, tinyot_num, tinyot_key_);

  // we need extra sigma bits to check
  auto auth_r = RandomBits(comm_, send_opts_, recv_opts_, sigma, tinyot_key_);

  // For convenient, put a,b,c,r together
  // Then authorize them in SPDZ2k form
  // todo: maybe we can use uint64_t in FM64
  AuthBit auth_abcr{std::vector<bool>(3 * tinyot_num + sigma, false),
                    std::vector<uint128_t>(3 * tinyot_num + sigma, 0),
                    tinyot_key_};
  for (int64_t i = 0; i < tinyot_num; ++i) {
    auth_abcr.choices[i] = auth_a.choices[i];
    auth_abcr.choices[tinyot_num + i] = auth_b.choices[i];
    auth_abcr.choices[tinyot_num * 2 + i] = auth_c.choices[i];
  }
  for (int64_t i = 0; i < sigma; ++i) {
    auth_abcr.choices[tinyot_num * 3 + i] = auth_r.choices[i];
  }
  std::memcpy(&auth_abcr.mac[0], &auth_a.mac[0],
              tinyot_num * sizeof(uint128_t));
  std::memcpy(&auth_abcr.mac[tinyot_num], &auth_b.mac[0],
              tinyot_num * sizeof(uint128_t));
  std::memcpy(&auth_abcr.mac[tinyot_num * 2], &auth_c.mac[0],
              tinyot_num * sizeof(uint128_t));
  std::memcpy(&auth_abcr.mac[tinyot_num * 3], &auth_r.mac[0],
              sigma * sizeof(uint128_t));

  // Generate authorize bits in the form of B-Share
  NdArrayRef spdz_choices(makeType<RingTy>(field), {tinyot_num * 3 + sigma});

  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using U = std::make_unsigned<ring2k_t>::type;
    auto _size = auth_abcr.choices.size();
    NdArrayView<U> _spdz_choices(spdz_choices);
    // copy authbit choices
    yacl::parallel_for(0, _size, 4096, [&](int64_t beg, int64_t end) {
      for (int64_t idx = beg; idx < end; ++idx) {
        _spdz_choices[idx] = auth_abcr.choices[idx];
      }
    });
  });

  NdArrayRef spdz_mac(makeType<RingTy>(field), {tinyot_num * 3 + sigma});
  NdArrayRef mask0(makeType<RingTy>(field), {tinyot_num * 3 + sigma});
  NdArrayRef mask1(makeType<RingTy>(field), {tinyot_num * 3 + sigma});
  NdArrayRef t(makeType<RingTy>(field), {tinyot_num * 3 + sigma});
  auto ext_spdz_key =
      ring_mul(ring_ones(field, {tinyot_num * 3 + sigma}), spdz_key_);

  if (comm_->getRank() == 0) {
    rotRecv(field, spdz_choices, &t);
    auto recv = comm_->recv(comm_->nextRank(), makeType<RingTy>(field), "recv");

    rotSend(field, &mask0, &mask1);
    auto diff = ring_add(ring_sub(mask0, mask1), ext_spdz_key);
    comm_->sendAsync(comm_->nextRank(), diff, "send");
    spdz_mac = ring_add(t, ring_mul(spdz_choices, recv));
  } else {
    rotSend(field, &mask0, &mask1);
    auto diff = ring_add(ring_sub(mask0, mask1), ext_spdz_key);
    comm_->sendAsync(comm_->nextRank(), diff, "send");

    rotRecv(field, spdz_choices, &t);
    auto recv = comm_->recv(comm_->nextRank(), makeType<RingTy>(field), "recv");
    spdz_mac = ring_add(t, ring_mul(spdz_choices, recv));
  }
  spdz_mac = ring_sub(spdz_mac, mask0);
  spdz_mac = ring_add(spdz_mac, ring_mul(spdz_choices, ext_spdz_key));

  AuthBit check_tiny_bit = {std::vector<bool>(sigma, false),
                            std::vector<uint128_t>(sigma, 0), tinyot_key_};
  auto check_spdz_bit = ring_zeros(field, {sigma});
  auto check_spdz_mac = ring_zeros(field, {sigma});
  auto seed = GenSharedSeed(comm_);
  auto prg = yacl::crypto::Prg<uint64_t>(seed);

  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using U = std::make_unsigned<ring2k_t>::type;
    NdArrayView<U> _check_spdz_bit(check_spdz_bit);
    NdArrayView<U> _check_spdz_mac(check_spdz_mac);
    NdArrayView<U> _spdz_choices(spdz_choices);
    NdArrayView<U> _spdz_mac(spdz_mac);

    for (int64_t i = 0; i < sigma; ++i) {
      _check_spdz_bit[i] = _spdz_choices[3 * tinyot_num + i];
      _check_spdz_mac[i] = _spdz_mac[3 * tinyot_num + i];
      check_tiny_bit.mac[i] = auth_abcr.mac[tinyot_num * 3 + i];
    }
    for (int64_t j = 0; j < tinyot_num * 3; ++j) {
      // we can ignore check_tiny_bit.choices
      uint64_t ceof = prg();
      // sigma = 64
      for (size_t i = 0; i < sigma; ++i) {
        if (ceof & 1) {
          check_tiny_bit.mac[i] ^= auth_abcr.mac[j];
          _check_spdz_bit[i] += _spdz_choices[j];
          _check_spdz_mac[i] += _spdz_mac[j];
        }
        ceof >>= 1;
      }
    }
  });

  // Open sigma bits
  auto [open_bit, zero_mac] = BatchOpen(check_spdz_bit, check_spdz_mac, 1, s);
  check_tiny_bit.choices = ring_cast_vector_boolean(open_bit);

  // TINY Maccheck & SPDZ Maccheck!!
  size_t k = s;
  SPU_ENFORCE(TinyMacCheck(comm_, check_tiny_bit.choices, check_tiny_bit));
  SPU_ENFORCE(BatchMacCheck(open_bit, zero_mac, k, s));

  // Pack a,b,c and their mac
  auto compact_strides = makeCompactStrides(shape);
  auto a = NdArrayRef(spdz_choices.buf(), spdz_choices.eltype(), shape,
                      compact_strides, 0);
  auto b = NdArrayRef(spdz_choices.buf(), spdz_choices.eltype(), shape,
                      compact_strides, tinyot_num * elsize);
  auto c = NdArrayRef(spdz_choices.buf(), spdz_choices.eltype(), shape,
                      compact_strides, 2 * tinyot_num * elsize);

  auto a_mac =
      NdArrayRef(spdz_mac.buf(), spdz_mac.eltype(), shape, compact_strides, 0);
  auto b_mac = NdArrayRef(spdz_mac.buf(), spdz_mac.eltype(), shape,
                          compact_strides, tinyot_num * elsize);
  auto c_mac = NdArrayRef(spdz_mac.buf(), spdz_mac.eltype(), shape,
                          compact_strides, 2 * tinyot_num * elsize);

  return {{a, b, c}, {a_mac, b_mac, c_mac}};
}

BeaverTinyOt::Triple BeaverTinyOt::dot(FieldType field, int64_t M, int64_t N,
                                       int64_t K, size_t k, size_t /*s*/) {
  size_t WorldSize = comm_->getWorldSize();
  size_t rank = comm_->getRank();

  auto a = ring_rand(field, {M, K});
  auto b = ring_rand(field, {K, N});
  ring_bitmask_(a, 0, k);
  ring_bitmask_(b, 0, k);

  auto c = ring_mmul(a, b);

  // w = a * b + v
  std::vector<NdArrayRef> w;
  std::vector<NdArrayRef> v;
  // every pair calls voleDot
  for (size_t i = 0; i < WorldSize; ++i) {
    for (size_t j = 0; j < WorldSize; ++j) {
      if (i == j) {
        continue;
      }
      if (i == rank) {
        auto tmp = voleRecvDot(field, b, M, N, K);
        w.emplace_back(tmp);
      }
      if (j == rank) {
        auto tmp = voleSendDot(field, a, M, N, K);
        v.emplace_back(tmp);
      }
    }
  }

  for (size_t i = 0; i < WorldSize - 1; ++i) {
    ring_add_(c, ring_sub(w[i], v[i]));
  }
  return {a, b, c};
}

// Refer to:
// 6 PreProcessing: Creating Multiplication Triples,
// SPDZ2k: Efficient MPC mod 2k for Dishonest Majority
// - https://eprint.iacr.org/2018/482.pdf
BeaverTinyOt::Triple_Pair BeaverTinyOt::AuthDot(FieldType field, int64_t M,
                                                int64_t N, int64_t K, size_t k,
                                                size_t s) {
  // Dot
  auto [a_ext, b, c_ext] = dot(field, 2 * M, N, K, k, s);

  // Authenticate
  auto a_ext_mac = AuthArrayRef(a_ext, field, k, s);
  auto b_mac = AuthArrayRef(b, field, k, s);
  auto c_ext_mac = AuthArrayRef(c_ext, field, k, s);

  auto a = a_ext.slice({0, 0}, {M, K}, {1, 1});
  auto a_mac = a_ext_mac.slice({0, 0}, {M, K}, {1, 1});
  auto c = c_ext.slice({0, 0}, {M, N}, {1, 1});
  auto c_mac = c_ext_mac.slice({0, 0}, {M, N}, {1, 1});

  // Sacrifice
  auto a2 = a_ext.slice({M, 0}, {2 * M, K}, {1, 1});
  auto a2_mac = a_ext_mac.slice({M, 0}, {2 * M, K}, {1, 1});
  auto c2 = c_ext.slice({M, 0}, {2 * M, N}, {1, 1});
  auto c2_mac = c_ext_mac.slice({M, 0}, {2 * M, N}, {1, 1});

  auto t = prg_state_->genPubl(field, {M, M});
  auto rou = ring_sub(ring_mmul(t, a), a2);
  auto rou_mac = ring_sub(ring_mmul(t, a_mac), a2_mac);

  auto [pub_rou, check_rou_mac] = BatchOpen(rou, rou_mac, k, s);
  SPU_ENFORCE(BatchMacCheck(pub_rou, check_rou_mac, k, s));

  auto t_delta = ring_sub(ring_mmul(t, c), c2);
  auto delta = ring_sub(t_delta, ring_mmul(pub_rou, b));

  auto t_delta_mac = ring_sub(ring_mmul(t, c_mac), c2_mac);
  auto delta_mac = ring_sub(t_delta_mac, ring_mmul(pub_rou, b_mac));

  auto [pub_delta, check_delta_mac] = BatchOpen(delta, delta_mac, k, s);
  SPU_ENFORCE(BatchMacCheck(pub_delta, check_delta_mac, k, s));

  // Output
  return {{a, b, c}, {a_mac, b_mac, c_mac}};
}

BeaverTinyOt::Pair_Pair BeaverTinyOt::AuthTrunc(FieldType field,
                                                const Shape& shape, size_t bits,
                                                size_t k, size_t s) {
  size_t nbits = k;

  int64_t size = shape.numel();
  NdArrayRef b_val, b_mac;
  std::tie(b_val, b_mac) = AuthRandBit(field, {(int64_t)nbits * size}, k, s);

  // compose
  NdArrayRef r_val(b_val.eltype(), shape);
  NdArrayRef r_mac(b_val.eltype(), shape);
  NdArrayRef tr_val(b_val.eltype(), shape);
  NdArrayRef tr_mac(b_val.eltype(), shape);

  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using PShrT = ring2k_t;
    NdArrayView<PShrT> _val(b_val);
    NdArrayView<PShrT> _mac(b_mac);
    NdArrayView<PShrT> _r_val(r_val);
    NdArrayView<PShrT> _r_mac(r_mac);
    NdArrayView<PShrT> _tr_val(tr_val);
    NdArrayView<PShrT> _tr_mac(tr_mac);
    pforeach(0, size, [&](int64_t idx) {
      _r_val[idx] = 0;
      _r_mac[idx] = 0;
      _tr_val[idx] = 0;
      _tr_mac[idx] = 0;
      for (size_t bit = 0; bit < nbits; bit++) {
        size_t flat_idx = idx * nbits + bit;
        _r_val[idx] += _val[flat_idx] << bit;
        _r_mac[idx] += _mac[flat_idx] << bit;
      }
      for (size_t bit = 0; bit + bits < nbits; bit++) {
        size_t flat_idx = idx * nbits + bits + bit;
        _tr_val[idx] += _val[flat_idx] << bit;
        _tr_mac[idx] += _mac[flat_idx] << bit;
      }

      for (size_t bit = nbits - bits; bit < nbits; bit++) {
        size_t flat_idx = idx * nbits + nbits - 1;
        _tr_val[idx] += _val[flat_idx] << bit;
        _tr_mac[idx] += _mac[flat_idx] << bit;
      }
    });
  });

  return {{r_val, tr_val}, {r_mac, tr_mac}};
}

// Refer to:
// New Primitives for Actively-Secure MPC over Rings with Applications to
// Private Machine Learning.
// Figure 5: Protocol for obtaining authenticated shared bits
// - https://eprint.iacr.org/2019/599.pdf
BeaverTinyOt::Pair BeaverTinyOt::AuthRandBit(FieldType field,
                                             const Shape& shape, size_t k,
                                             size_t s) {
  auto u = ring_rand(field, shape);
  ring_bitmask_(u, 0, k + 2);
  auto u_mac = AuthArrayRef(u, field, k + 2, s);

  auto y = ring_mul(u, 2);
  auto y_mac = ring_mul(u_mac, 2);
  auto ones = ring_ones(field, shape);
  auto ones_mac = ring_mul(ones, spdz_key_);

  if (comm_->getRank() == 0) {
    ring_add_(y, ones);
  }
  ring_add_(y_mac, ones_mac);

  auto [beaver_vec, beaver_mac] = AuthMul(field, shape, k, s);
  auto& [a, b, c] = beaver_vec;
  auto& [a_mac, b_mac, c_mac] = beaver_mac;

  auto e = ring_sub(y, a);
  auto e_mac = ring_sub(y_mac, a_mac);
  auto f = ring_sub(y, b);
  auto f_mac = ring_sub(y_mac, b_mac);

  // Open the least significant bit and Check them
  auto [p_e, pe_mac] = BatchOpen(e, e_mac, k + 2, s);
  auto [p_f, pf_mac] = BatchOpen(f, f_mac, k + 2, s);

  SPU_ENFORCE(BatchMacCheck(p_e, pe_mac, k, s));
  SPU_ENFORCE(BatchMacCheck(p_f, pf_mac, k, s));

  // Reserve the least significant bit only
  ring_bitmask_(p_e, 0, k + 2);
  ring_bitmask_(p_f, 0, k + 2);
  auto p_ef = ring_mul(p_e, p_f);

  // z = p_e * b + p_f * a + c;
  auto z = ring_add(ring_mul(p_e, b), ring_mul(p_f, a));
  ring_add_(z, c);
  if (comm_->getRank() == 0) {
    // z += p_e * p_f;
    ring_add_(z, p_ef);
  }

  // z_mac = p_e * b_mac + p_f * a_mac + c_mac + p_e * p_f * key;
  auto z_mac = ring_add(ring_mul(p_e, b_mac), ring_mul(p_f, a_mac));
  ring_add_(z_mac, c_mac);
  ring_add_(z_mac, ring_mul(p_ef, spdz_key_));

  auto [square, zero_mac] = BatchOpen(z, z_mac, k + 2, s);
  SPU_ENFORCE(BatchMacCheck(square, zero_mac, k, s));
  SPU_ENFORCE(ring_all_equal(ring_bitmask(square, 0, 1), ones));
  auto root = ring_sqrt2k(square, k + 2);
  auto root_inv = ring_inv2k(root, k + 2);
  auto root_inv_div2 = ring_rshift(root_inv, 1);

  auto d = ring_mul(root_inv_div2, y);
  auto d_mac = ring_mul(root_inv_div2, y_mac);
  ring_add_(d, u);
  ring_add_(d_mac, u_mac);
  if (comm_->getRank() == 0) {
    ring_add_(d, ones);
  }
  ring_add_(d_mac, ones_mac);

  return {d, d_mac};
}

NdArrayRef BeaverTinyOt::genPublCoin(FieldType field, int64_t num) {
  NdArrayRef res(makeType<RingTy>(field), {num});

  // generate new seed
  uint128_t seed = yacl::crypto::SecureRandSeed();
  std::vector<std::string> all_strs;

  std::string seed_str(reinterpret_cast<char*>(&seed), sizeof(seed));
  SPU_ENFORCE(commit_and_open(comm_->lctx(), seed_str, &all_strs));

  uint128_t public_seed = 0;
  for (const auto& str : all_strs) {
    uint128_t seed = *(reinterpret_cast<const uint128_t*>(str.data()));
    public_seed += seed;
  }

  const auto kAesType = yacl::crypto::SymmetricCrypto::CryptoType::AES128_CTR;
  yacl::crypto::FillPRand(kAesType, public_seed, 0, 0,
                          absl::MakeSpan(res.data<char>(), res.buf()->size()));

  return res;
}

// Refer to:
// Procedure BatchCheck, 3.2 Batch MAC Checking with Random Linear
// Combinations, SPDZ2k: Efficient MPC mod 2k for Dishonest Majority
// - https://eprint.iacr.org/2018/482.pdf
//
// Check the opened value only
bool BeaverTinyOt::BatchMacCheck(const NdArrayRef& open_value,
                                 const NdArrayRef& mac, size_t k, size_t s) {
  SPDLOG_DEBUG("BatchMacCheck start...");
  SPU_ENFORCE(open_value.shape() == mac.shape());
  const auto field = open_value.eltype().as<Ring2k>()->field();
  const size_t mac_bits = k + s;
  const size_t key = spdz_key_;
  int64_t num = open_value.numel();

  // 1. Generate ceof
  auto coef = genPublCoin(field, num).reshape({1, num});
  ring_bitmask_(coef, 0, s);

  // 3. check_value = coef * open_value
  //    check_mac = coef * mac
  auto check_value = ring_mmul(coef, open_value.reshape({num, 1}));
  auto check_mac = ring_mmul(coef, mac.reshape({num, 1}));

  // 4. local_mac = check_mac - check_value * key
  auto local_mac = ring_sub(check_mac, ring_mul(check_value, key));
  // commit and reduce all macs
  std::string mac_str(local_mac.data<char>(),
                      local_mac.numel() * local_mac.elsize());
  std::vector<std::string> all_mac_strs;
  SPU_ENFORCE(commit_and_open(comm_->lctx(), mac_str, &all_mac_strs));
  SPU_ENFORCE(all_mac_strs.size() == comm_->getWorldSize());

  // 5. compute the sum of all macs
  auto zero_mac = ring_zeros(field, {1});
  for (size_t i = 0; i < comm_->getWorldSize(); ++i) {
    const auto& _mac_str = all_mac_strs[i];
    auto buf = std::make_shared<yacl::Buffer>(_mac_str.data(), _mac_str.size());
    NdArrayRef _mac(buf, zero_mac.eltype(),
                    {(int64_t)(_mac_str.size() / SizeOf(field))}, {1}, 0);
    ring_add_(zero_mac, _mac);
  }

  // 6. In B-share, the range of Mac is Z_2^{s+1}
  if (mac_bits != 0) {
    ring_bitmask_(zero_mac, 0, mac_bits);
  }

  // 7. verify whether the sum of all macs is zero
  auto res = ring_all_equal(zero_mac, ring_zeros(field, {1}));
  SPDLOG_DEBUG("BatchMacCheck end with ret {}.", res);
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
std::pair<NdArrayRef, NdArrayRef> BeaverTinyOt::BatchOpen(
    const NdArrayRef& value, const NdArrayRef& mac, size_t k, size_t s) {
  static constexpr char kBindName[] = "batch_open";
  SPU_ENFORCE(value.shape() == mac.shape());
  const auto field = value.eltype().as<Ring2k>()->field();
  size_t field_bits = std::min<size_t>(SizeOf(field) * 8, 64);
  auto [r_val, r_mac] = AuthCoinTossing(field, value.shape(), field_bits, s);
  // Open the low k_bits only
  // value = value + r * 2^k
  // mac = mac + r_mac * 2^k
  auto masked_val = ring_add(value, ring_lshift(r_val, k));
  auto masked_mac = ring_add(mac, ring_lshift(r_mac, k));

  // Because we would use Maccheck to comfirm the open value.
  // Thus, we don't need commit them.
  auto open_val = comm_->allReduce(ReduceOp::ADD, masked_val, kBindName);
  return {open_val, masked_mac};
}

void BeaverTinyOt::rotSend(FieldType field, NdArrayRef* q0, NdArrayRef* q1) {
  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using T = ring2k_t;

    SPDLOG_DEBUG("rotSend start with numel {}", q0->numel());
    SPU_ENFORCE(q0->numel() == q1->numel());
    size_t numel = q0->numel();
    auto* data0 = q0->data<T>();
    auto* data1 = q1->data<T>();

    SPU_ENFORCE(spdz2k_ot_primitives_ != nullptr);
    SPU_ENFORCE(spdz2k_ot_primitives_->GetSenderCOT() != nullptr);

    spdz2k_ot_primitives_->GetSenderCOT()->SendRMCC(
        absl::MakeSpan(data0, numel), absl::MakeSpan(data1, numel));
    spdz2k_ot_primitives_->GetSenderCOT()->Flush();

    SPDLOG_DEBUG("rotSend end");
  });
}

// todo: use dynamic_bitset instead of ArrayRef for `a` to improve performance
void BeaverTinyOt::rotRecv(FieldType field, const NdArrayRef& a,
                           NdArrayRef* s) {
  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using T = ring2k_t;

    SPDLOG_DEBUG("rotRecv start with numel {}", a.numel());
    size_t numel = a.numel();
    std::vector<uint8_t> b_v(numel);

    NdArrayView<T> _a(a);
    for (size_t i = 0; i < numel; ++i) {
      b_v[i] = _a[i];
    }

    SPU_ENFORCE(spdz2k_ot_primitives_ != nullptr);
    SPU_ENFORCE(spdz2k_ot_primitives_->GetSenderCOT() != nullptr);
    SPU_ENFORCE(spdz2k_ot_primitives_->GetReceiverCOT() != nullptr);

    auto* data = s->data<T>();
    spdz2k_ot_primitives_->GetReceiverCOT()->RecvRMCC(
        b_v, absl::MakeSpan(data, numel));
    spdz2k_ot_primitives_->GetReceiverCOT()->Flush();

    SPDLOG_DEBUG("rotRecv end");
  });
}

// Refer to:
// Appendix C. Implementing Vector-OLE mod 2^l, P35
// SPDZ2k: Efficient MPC mod 2k for Dishonest Majority
// - https://eprint.iacr.org/2018/482.pdf
NdArrayRef BeaverTinyOt::voleSend(FieldType field, const NdArrayRef& x) {
  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using T = ring2k_t;

    SPU_ENFORCE(spdz2k_ot_primitives_ != nullptr);
    SPU_ENFORCE(spdz2k_ot_primitives_->GetSenderCOT() != nullptr);

    NdArrayRef res(x.eltype(), x.shape());
    auto* data = res.data<T>();
    spdz2k_ot_primitives_->GetSenderCOT()->SendVole(
        absl::MakeConstSpan(x.data<const T>(), x.numel()),
        absl::MakeSpan(data, x.numel()));

    return res;
  });
}

NdArrayRef BeaverTinyOt::voleRecv(FieldType field, const NdArrayRef& alpha) {
  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using T = ring2k_t;

    SPU_ENFORCE(spdz2k_ot_primitives_ != nullptr);
    SPU_ENFORCE(spdz2k_ot_primitives_->GetReceiverCOT() != nullptr);

    NdArrayRef res(makeType<RingTy>(field), alpha.shape());
    auto* data = res.data<T>();
    spdz2k_ot_primitives_->GetReceiverCOT()->RecvVole(
        absl::MakeConstSpan(alpha.data<const T>(), alpha.numel()),
        absl::MakeSpan(data, alpha.numel()));

    return res;
  });
}

// Private Matrix Multiplication by VOLE
// W = V + A dot B
// Sender: input A, receive V
//
// Input: (M, K) matrix
// Output: (M, N) matrix
NdArrayRef BeaverTinyOt::voleSendDot(FieldType field, const NdArrayRef& x,
                                     int64_t M, int64_t N, int64_t K) {
  SPU_ENFORCE(x.shape() == (std::vector<int64_t>{M, K}));

  auto ret = ring_zeros(field, {M * N});
  for (int64_t i = 0; i < N; ++i) {
    // t: (M, K) matrix
    auto t = voleSend(field, x).reshape({M * K});

    // process the matrix
    auto ret_col = ret.slice({i}, {M * N}, {N});
    for (int64_t j = 0; j < K; ++j) {
      ring_add_(ret_col, t.slice({j}, {M * K}, {K}));
    }
  }

  return ret.reshape({M, N});
}

// Private Matrix Multiplication by VOLE
// W = V + A dot B
// Receiver: input B, receive W
//
// Input: (K, N) matrix
// Output: (M, N) matrix
NdArrayRef BeaverTinyOt::voleRecvDot(FieldType field, const NdArrayRef& alpha,
                                     int64_t M, int64_t N, int64_t K) {
  SPU_ENFORCE(alpha.shape() == (std::vector<int64_t>{K, N}));

  auto ret = ring_zeros(field, {M * N});
  auto f_alpha = alpha.reshape({alpha.numel()});
  for (int64_t i = 0; i < N; ++i) {
    auto alpha_col = f_alpha.slice({i}, {K * N}, {N});

    NdArrayRef alpha_ext(alpha.eltype(), {M * K});
    for (int64_t i = 0; i < M; ++i) {
      auto alpha_ext_row = alpha_ext.slice({i * K}, {(i + 1) * K}, {1});
      ring_assign(alpha_ext_row, alpha_col);
    }

    // t: (m, k) matrix
    auto t = voleRecv(field, alpha_ext);

    // process the matrix
    auto ret_col = ret.slice({i}, {M * N}, {N});
    for (int64_t j = 0; j < K; ++j) {
      ring_add_(ret_col, t.slice({j}, {M * K}, {K}));
    }
  }

  return ret.reshape({M, N});
}

// Refer to:
// 6 PreProcessing: Creating Multiplication Triples,
// SPDZ2k: Efficient MPC mod 2k for Dishonest Majority
// - https://eprint.iacr.org/2018/482.pdf
BeaverTinyOt::Triple_Pair BeaverTinyOt::AuthMul(FieldType field,
                                                const Shape& shape, size_t k,
                                                size_t s) {
  auto _size = shape.numel();

  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using T = ring2k_t;

    SPDLOG_DEBUG("AuthMul start...");
    int64_t tao = 4 * s + 2 * k;
    int64_t expand_tao = tao * _size;
    auto a = ring_randbit(field, {expand_tao});

    auto b = ring_rand(field, {_size});
    auto b_arr = ring_zeros(field, {expand_tao});

    NdArrayView<T> _b(b);
    NdArrayView<T> _b_arr(b_arr);
    for (int64_t i = 0; i < expand_tao; ++i) {
      _b_arr[i] = _b[i / tao];
    }

    // Every ordered pair does following
    size_t WorldSize = comm_->getWorldSize();
    size_t rank = comm_->getRank();
    NdArrayRef q0(makeType<RingTy>(field), {expand_tao});
    NdArrayRef q1(makeType<RingTy>(field), {expand_tao});
    NdArrayRef t_s(makeType<RingTy>(field), {expand_tao});

    std::vector<NdArrayRef> ci, cj;

    for (size_t i = 0; i < WorldSize; ++i) {
      for (size_t j = 0; j < WorldSize; ++j) {
        if (i == j) {
          continue;
        }

        if (i == rank) {
          rotRecv(field, a, &t_s);
          auto tmp = comm_->lctx()->Recv(j, "recv_d");
          NdArrayRef recv_d(std::make_shared<yacl::Buffer>(tmp), a.eltype(),
                            a.shape(), a.strides(), a.offset());
          auto t = ring_add(t_s, ring_mul(a, recv_d));
          ci.emplace_back(t);
        }

        if (j == rank) {
          rotSend(field, &q0, &q1);
          auto d = ring_add(ring_sub(q0, q1), b_arr);
          comm_->lctx()->SendAsync(i, *(d.buf().get()), "send_d");
          cj.emplace_back(ring_neg(q0));
        }
      }
    }

    auto cij = ring_zeros(field, {expand_tao});
    auto cji = ring_zeros(field, {expand_tao});
    for (size_t i = 0; i < WorldSize - 1; ++i) {
      ring_add_(cij, ci[i]);
      ring_add_(cji, cj[i]);
    }

    // Construct c
    auto c = ring_mul(a, b_arr);
    auto other_c = ring_add(cij, cji);
    ring_add_(c, other_c);

    // Combine
    auto r = prg_state_->genPubl(field, {expand_tao});
    auto r_hat = prg_state_->genPubl(field, {expand_tao});
    auto ra = ring_mul(r, a);
    auto ra_hat = ring_mul(r_hat, a);
    auto rc = ring_mul(r, c);
    auto rc_hat = ring_mul(r_hat, c);

    NdArrayRef cra = ring_zeros(field, {_size});
    NdArrayRef cra_hat = ring_zeros(field, {_size});
    NdArrayRef crc = ring_zeros(field, {_size});
    NdArrayRef crc_hat = ring_zeros(field, {_size});

    NdArrayView<T> _cra(cra);
    NdArrayView<T> _cra_hat(cra_hat);
    NdArrayView<T> _crc(crc);
    NdArrayView<T> _crc_hat(cra_hat);
    NdArrayView<T> _ra(ra);
    NdArrayView<T> _ra_hat(ra_hat);
    NdArrayView<T> _rc(rc);
    NdArrayView<T> _rc_hat(rc_hat);

    for (int64_t i = 0; i < expand_tao; ++i) {
      _cra[i / tao] += _ra[i];
      _cra_hat[i / tao] += _ra_hat[i];

      _crc[i / tao] += _rc[i];
      _crc_hat[i / tao] += _rc_hat[i];
    }

    // Authenticate
    auto a_mac = AuthArrayRef(cra, field, k, s);
    auto b_mac = AuthArrayRef(b, field, k, s);
    auto c_mac = AuthArrayRef(crc, field, k, s);

    auto a_hat_mac = AuthArrayRef(cra_hat, field, k, s);
    auto c_hat_mac = AuthArrayRef(crc_hat, field, k, s);

    // Sacrifice
    auto t = prg_state_->genPubl(field, {_size});
    auto rou = ring_sub(ring_mul(t, cra), cra_hat);
    auto rou_mac = ring_sub(ring_mul(t, a_mac), a_hat_mac);

    auto [pub_rou, check_rou_mac] = BatchOpen(rou, rou_mac, k, s);
    SPU_ENFORCE(BatchMacCheck(pub_rou, check_rou_mac, k, s));

    auto t_delta = ring_sub(ring_mul(t, crc), crc_hat);
    auto delta = ring_sub(t_delta, ring_mul(b, pub_rou));

    auto t_delta_mac = ring_sub(ring_mul(t, c_mac), c_hat_mac);
    auto delta_mac = ring_sub(t_delta_mac, ring_mul(b_mac, pub_rou));

    auto [pub_delta, check_delta_mac] = BatchOpen(delta, delta_mac, k, s);
    SPU_ENFORCE(BatchMacCheck(pub_delta, check_delta_mac, k, s));

    SPDLOG_DEBUG("AuthMul end");
    // Output
    return BeaverTinyOt::Triple_Pair{
        {cra.reshape(shape), b.reshape(shape), crc.reshape(shape)},
        {a_mac.reshape(shape), b_mac.reshape(shape), c_mac.reshape(shape)}};
  });
}

}  // namespace spu::mpc::spdz2k
