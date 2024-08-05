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
#include "tiny_ot.h"

#include "emp-tool/utils/f2k.h"
#include "yacl/crypto/tools/prg.h"
#include "yacl/crypto/utils/rand.h"

#include "libspu/mpc/spdz2k/commitment.h"
#include "libspu/mpc/spdz2k/ot/kos_ote.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::spdz2k {

namespace {
inline int GetBit(const std::vector<uint128_t>& choices, size_t idx) {
  uint128_t mask = uint128_t(1) << (idx & 127);
  return (choices[idx / 128] & mask) ? 1 : 0;
}

inline void SetBit(std::vector<uint128_t>& choices, size_t idx) {
  choices[idx / 128] |= (uint128_t(1) << (idx & 127));
}

inline emp::block U128ToBlock(uint128_t x) {
  auto [high, low] = yacl::DecomposeUInt128(x);
  return emp::makeBlock(high, low);
}

inline uint128_t BlockToU128(emp::block x) {
  auto high = static_cast<uint64_t>(_mm_extract_epi64(x, 1));
  auto low = static_cast<uint64_t>(_mm_extract_epi64(x, 0));
  return yacl::MakeInt128(high, low);
}

inline AuthBit AuthBitSender(
    const std::shared_ptr<Communicator>& comm,
    const std::shared_ptr<yacl::crypto::OtSendStore>& base_ot, size_t size,
    uint128_t tinyot_key) {
  std::vector<uint128_t> send_blocks(size);
  bool use_secure_rand = true;
  auto choices =
      yacl::crypto::RandBits<std::vector<bool>>(size, use_secure_rand);
  KosOtExtRecv(comm->lctx(), base_ot, choices, absl::MakeSpan(send_blocks));
  for (size_t k = 0; k < size; ++k) {
    if (choices[k]) {
      send_blocks[k] ^= tinyot_key;
    }
  }
  return AuthBit{std::move(choices), std::move(send_blocks), tinyot_key};
}

inline AuthBit AuthBitReceiver(
    const std::shared_ptr<Communicator>& comm,
    const std::shared_ptr<yacl::crypto::OtRecvStore>& base_ot, size_t size,
    uint128_t tinyot_key) {
  std::vector<uint128_t> recv_blocks(size);
  uint128_t delta = 0;
  KosOtExtSend(comm->lctx(), base_ot, absl::MakeSpan(recv_blocks), delta);
  // ENSURE KOS use tinyot_key as delta
  SPU_ENFORCE(delta == tinyot_key);

  return AuthBit{std::vector<bool>(size, false), std::move(recv_blocks),
                 tinyot_key};
}

inline void BatchSwitchKeySender(const std::shared_ptr<Communicator>& comm,
                                 const std::vector<uint128_t> new_tinyot_keys,
                                 AuthBit& bits) {
  auto sigmas = comm->recv<uint128_t>(comm->nextRank(), "TinyOT:DirtySwitch");
  // NOTICE!!!
  // we will not set bits.key although bits have already switch to some new keys
  for (size_t i = 0; i < new_tinyot_keys.size(); ++i) {
    if (bits.choices[i]) {
      bits.mac[i] ^= (sigmas[i] ^ new_tinyot_keys[i] ^ bits.key);
    }
  }
}

inline void BatchSwitchKeyReceiver(
    const std::shared_ptr<Communicator>& comm,
    const std::vector<uint128_t>& new_tinyot_keys, AuthBit& bits) {
  auto sigmas = new_tinyot_keys;
  for (auto& sigma : sigmas) {
    sigma ^= bits.key;
  }
  comm->sendAsync<uint128_t>(comm->nextRank(), sigmas, "TinyOT:DirtySwitch");
}

inline void SetSender(const std::shared_ptr<Communicator>& comm,
                      std::vector<bool> new_choices, AuthBit& bits) {
  SPU_ENFORCE(new_choices.size() <= bits.choices.size());
  std::vector<uint128_t> d((new_choices.size() + 127) / 128, 0);
  for (size_t i = 0; i < new_choices.size(); ++i) {
    if (new_choices[i] ^ bits.choices[i]) {
      SetBit(d, i);
      bits.mac[i] ^= bits.key;
    }
    bits.choices[i] = new_choices[i];
  }
  comm->sendAsync<uint128_t>(comm->nextRank(), d, "TinyOT:Set");
}

inline void SetReceiver(const std::shared_ptr<Communicator>& comm,
                        AuthBit& bits) {
  auto d = comm->recv<uint128_t>(comm->nextRank(), "TinyOT:Set");
  auto bound = std::min(bits.choices.size(), d.size() * 128);
  for (size_t i = 0; i < bound; ++i) {
    if (GetBit(d, i) == 1) {
      bits.mac[i] ^= bits.key;
    }
  }
}

// ShareOT protocol
// Reference: https://eprint.iacr.org/2014/101.pdf
// Page 11, protocol 8.
std::tuple<AuthBit, AuthBit, AuthBit, AuthBit> ShareOT(
    const std::shared_ptr<Communicator>& comm,
    const std::shared_ptr<yacl::crypto::OtSendStore>& send_opts,
    const std::shared_ptr<yacl::crypto::OtRecvStore>& recv_opts, size_t size,
    uint128_t tinyot_key) {
  AuthBit local_bits;
  AuthBit remote_bits;

  if (comm->getRank() == 0) {
    local_bits = AuthBitSender(comm, send_opts, 5 * size, tinyot_key);
    remote_bits = AuthBitReceiver(comm, recv_opts, 5 * size, tinyot_key);
  } else {
    remote_bits = AuthBitReceiver(comm, recv_opts, 5 * size, tinyot_key);
    local_bits = AuthBitSender(comm, send_opts, 5 * size, tinyot_key);
  }

  std::vector<bool> e_chi_z(4 * size);
  bool use_secure_rand = true;
  std::vector<bool> e =
      yacl::crypto::RandBits<std::vector<bool>>(size, use_secure_rand);

  //
  yacl::crypto::Prg<uint128_t> prg(yacl::crypto::SecureRandSeed());
  std::vector<uint128_t> eta(size, 0);
  prg.Fill(absl::MakeSpan(eta));

  AuthBit temp_local_bits{std::vector<bool>(size),
                          std::vector<uint128_t>(size, 0), tinyot_key};
  AuthBit temp_remote_bits{std::vector<bool>(size, false),
                           std::vector<uint128_t>(size, 0), tinyot_key};
  std::memcpy(temp_local_bits.mac.data(), local_bits.mac.data() + 4 * size,
              size * sizeof(uint128_t));
  std::memcpy(temp_remote_bits.mac.data(), remote_bits.mac.data() + 4 * size,
              size * sizeof(uint128_t));
  for (size_t i = 0; i < size; ++i) {
    temp_local_bits.choices[i] = local_bits.choices[4 * size + i];
  }

  if (comm->getRank() == 0) {
    SetSender(comm, e, temp_local_bits);
    SetReceiver(comm, temp_remote_bits);
    BatchSwitchKeySender(comm, eta, temp_local_bits);
    BatchSwitchKeyReceiver(comm, eta, temp_remote_bits);
  } else {
    SetReceiver(comm, temp_remote_bits);
    SetSender(comm, e, temp_local_bits);
    BatchSwitchKeyReceiver(comm, eta, temp_remote_bits);
    BatchSwitchKeySender(comm, eta, temp_local_bits);
  }
  for (size_t i = 0; i < size; ++i) {
    e_chi_z[i] = e[i];
    e_chi_z[size + i] = (1 & temp_remote_bits.mac[i]);
    e_chi_z[2 * size + i] = (1 & (eta[i] ^ temp_remote_bits.mac[i]));
    e_chi_z[3 * size + i] = (1 & temp_local_bits.mac[i]);
  }

  // Authorize choices [e||chi0||chi1||z]
  if (comm->getRank() == 0) {
    SetSender(comm, e_chi_z, local_bits);
    SetReceiver(comm, remote_bits);
  } else {
    SetReceiver(comm, remote_bits);
    SetSender(comm, e_chi_z, local_bits);
  }

  AuthBit auth_e{std::vector<bool>(size), std::vector<uint128_t>(size),
                 tinyot_key};
  AuthBit auth_chi0{std::vector<bool>(size), std::vector<uint128_t>(size),
                    tinyot_key};
  AuthBit auth_chi1{std::vector<bool>(size), std::vector<uint128_t>(size),
                    tinyot_key};
  AuthBit auth_z{std::vector<bool>(size), std::vector<uint128_t>(size),
                 tinyot_key};

  std::memcpy(auth_e.mac.data(), local_bits.mac.data(),
              size * sizeof(uint128_t));
  std::memcpy(auth_chi0.mac.data(), local_bits.mac.data() + size,
              size * sizeof(uint128_t));
  std::memcpy(auth_chi1.mac.data(), local_bits.mac.data() + 2 * size,
              size * sizeof(uint128_t));
  std::memcpy(auth_z.mac.data(), local_bits.mac.data() + 3 * size,
              size * sizeof(uint128_t));

  for (size_t k = 0; k < size; ++k) {
    auth_e.choices[k] = local_bits.choices[k];
    auth_e.mac[k] ^= remote_bits.mac[k];

    auth_chi0.choices[k] = local_bits.choices[size + k];
    auth_chi0.mac[k] ^= remote_bits.mac[size + k];

    auth_chi1.choices[k] = local_bits.choices[2 * size + k];
    auth_chi1.mac[k] ^= remote_bits.mac[2 * size + k];

    auth_z.choices[k] = local_bits.choices[3 * size + k];
    auth_z.mac[k] ^= remote_bits.mac[3 * size + k];
  }

  return {auth_e, auth_chi0, auth_chi1, auth_z};
}

// GaOT protocol (includes Authenticate OT and Sacrifice OT)
// Reference: https://eprint.iacr.org/2014/101.pdf
// Page 14, Protocol 9.
std::tuple<AuthBit, AuthBit, AuthBit, AuthBit> GaOT(
    const std::shared_ptr<Communicator>& comm,
    const std::shared_ptr<yacl::crypto::OtSendStore>& send_opts,
    const std::shared_ptr<yacl::crypto::OtRecvStore>& recv_opts, size_t size,
    uint128_t tinyot_key) {
  constexpr size_t kS = 64;

  // Theorem 3. T >= (k + log2(t)) / log2(t)
  size_t expand_factor = std::ceil((kS + log2(size)) / log2(size)) + 1;
  size_t total_num = expand_factor * size;
  auto [e, chi0, chi1, z] =
      ShareOT(comm, send_opts, recv_opts, total_num, tinyot_key);

  // Phase-I: cut and choose
  yacl::crypto::Prg<uint128_t> prg(GenSharedSeed(comm));
  auto swap_lambda = [](AuthBit& a, size_t i0, size_t i1) {
    std::swap(a.choices[i0], a.choices[i1]);
    std::swap(a.mac[i0], a.mac[i1]);
  };

  for (size_t i = 1; i <= total_num; ++i) {
    const size_t k = prg() % total_num;
    swap_lambda(e, total_num - i, k);
    swap_lambda(chi0, total_num - i, k);
    swap_lambda(chi1, total_num - i, k);
    swap_lambda(z, total_num - i, k);
  }
  // Open last "size" quadruples
  AuthBit s_e_chi_z{std::vector<bool>(4 * size, false),
                    std::vector<uint128_t>(4 * size, 0), tinyot_key};
  const size_t bias = total_num - size;

  // s_e_chi_z_vec = [ e || chi0 || chi1 || z ]
  std::vector<uint128_t> s_e_chi_z_vec((4 * size + 127) / 128, 0);

  std::memcpy(s_e_chi_z.mac.data(), e.mac.data() + bias,
              size * sizeof(uint128_t));
  std::memcpy(s_e_chi_z.mac.data() + size, chi0.mac.data() + bias,
              size * sizeof(uint128_t));
  std::memcpy(s_e_chi_z.mac.data() + 2 * size, chi1.mac.data() + bias,
              size * sizeof(uint128_t));
  std::memcpy(s_e_chi_z.mac.data() + 3 * size, z.mac.data() + bias,
              size * sizeof(uint128_t));

  for (size_t i = 0; i < size; ++i) {
    if (e.choices[bias + i]) {
      s_e_chi_z.choices[i] = true;
      SetBit(s_e_chi_z_vec, i);
    }
    if (chi0.choices[bias + i]) {
      s_e_chi_z.choices[size + i] = true;
      SetBit(s_e_chi_z_vec, size + i);
    }
    if (chi1.choices[bias + i]) {
      s_e_chi_z.choices[2 * size + i] = true;
      SetBit(s_e_chi_z_vec, 2 * size + i);
    }
    if (z.choices[bias + i]) {
      s_e_chi_z.choices[3 * size + i] = true;
      SetBit(s_e_chi_z_vec, 3 * size + i);
    }
  }

  s_e_chi_z_vec = comm->allReduce<uint128_t, std::bit_xor>(
      s_e_chi_z_vec, "GaOT:cut_choose_e_chi_z");
  std::vector<bool> s_e_chi_z_bool(4 * size, false);

  for (size_t i = 0; i < size; ++i) {
    s_e_chi_z_bool[i] = GetBit(s_e_chi_z_vec, i);
    s_e_chi_z_bool[size + i] = GetBit(s_e_chi_z_vec, size + i);
    s_e_chi_z_bool[2 * size + i] = GetBit(s_e_chi_z_vec, 2 * size + i);
    s_e_chi_z_bool[3 * size + i] = GetBit(s_e_chi_z_vec, 3 * size + i);
    if (s_e_chi_z_bool[i]) {
      SPU_ENFORCE(s_e_chi_z_bool[3 * size + i] == s_e_chi_z_bool[2 * size + i]);
    } else {
      SPU_ENFORCE(s_e_chi_z_bool[3 * size + i] == s_e_chi_z_bool[size + i]);
    }
  }

  // Phase-II: bucket sacrifice
  const size_t sacrifice_size = (expand_factor - 2) * size;
  AuthBit a{std::vector<bool>(sacrifice_size, false),
            std::vector<uint128_t>(sacrifice_size, 0), tinyot_key};
  AuthBit b{std::vector<bool>(sacrifice_size, false),
            std::vector<uint128_t>(sacrifice_size, 0), tinyot_key};
  AuthBit c{std::vector<bool>(sacrifice_size, false),
            std::vector<uint128_t>(sacrifice_size, 0), tinyot_key};

  for (size_t offset = 0; offset < sacrifice_size; offset += size) {
    for (size_t i = 0; i < size; ++i) {
      a.choices[offset + i] = e.choices[i] ^ e.choices[size + offset + i];
      a.mac[offset + i] = e.mac[i] ^ e.mac[size + offset + i];
      b.choices[offset + i] = chi0.choices[i] ^
                              chi0.choices[size + offset + i] ^
                              chi1.choices[i] ^ chi1.choices[size + offset + i];
      b.mac[offset + i] = chi0.mac[i] ^ chi0.mac[size + offset + i] ^
                          chi1.mac[i] ^ chi1.mac[size + offset + i];
    }
  }

  std::vector<uint128_t> a_vec((sacrifice_size + 127) / 128, 0);
  std::vector<uint128_t> b_vec((sacrifice_size + 127) / 128, 0);

  for (size_t i = 0; i < sacrifice_size; ++i) {
    if (a.choices[i]) {
      SetBit(a_vec, i);
    }
    if (b.choices[i]) {
      SetBit(b_vec, i);
    }
  }

  a_vec =
      comm->allReduce<uint128_t, std::bit_xor>(a_vec, "GaOT:Sacrifice_open_a");
  b_vec =
      comm->allReduce<uint128_t, std::bit_xor>(b_vec, "GaOT:Sacrifice_open_b");

  for (size_t offset = 0; offset < sacrifice_size; offset += size) {
    for (size_t i = 0; i < size; ++i) {
      const size_t si = offset + i;
      c.choices[si] = z.choices[i] ^ chi0.choices[i] ^ z.choices[size + si] ^
                      chi0.choices[size + si] ^
                      (GetBit(a_vec, si) *
                       (chi0.choices[size + si] ^ chi1.choices[size + si])) ^
                      (GetBit(b_vec, si) * e.choices[i]);
      c.mac[si] =
          z.mac[i] ^ chi0.mac[i] ^ z.mac[size + si] ^ chi0.mac[size + si] ^
          (GetBit(a_vec, si) * (chi0.mac[size + si] ^ chi1.mac[size + si])) ^
          (GetBit(b_vec, si) * e.mac[i]);
    }
  }

  std::vector<uint128_t> c_vec((sacrifice_size + 127) / 128, 0);
  for (size_t i = 0; i < sacrifice_size; ++i) {
    if (c.choices[i]) {
      SetBit(c_vec, i);
    }
  }

  c_vec =
      comm->allReduce<uint128_t, std::bit_xor>(c_vec, "GaOT:Sacrifice_open_c");
  for (const auto& c : c_vec) {
    SPU_ENFORCE(c == 0);
  }

  // Phase-III: Maccheck
  std::vector<bool> a_open_bool(sacrifice_size, false);
  std::vector<bool> b_open_bool(sacrifice_size, false);
  std::vector<bool> c_open_bool(sacrifice_size, false);
  // all elements in c are "false".
  for (size_t i = 0; i < sacrifice_size; ++i) {
    if (GetBit(a_vec, i)) {
      a_open_bool[i] = true;
    }
    if (GetBit(b_vec, i)) {
      b_open_bool[i] = true;
    }
  }

  TinyMacCheck(comm, s_e_chi_z_bool, s_e_chi_z);
  TinyMacCheck(comm, a_open_bool, a);
  TinyMacCheck(comm, b_open_bool, b);
  TinyMacCheck(comm, c_open_bool, c);
  return {e, chi0, chi1, z};
}

};  // namespace

uint128_t GenSharedSeed(const std::shared_ptr<Communicator>& comm) {
  uint128_t seed = yacl::crypto::SecureRandSeed();
  std::string seed_str(reinterpret_cast<char*>(&seed), sizeof(uint128_t));
  std::vector<std::string> all_seed_strs;
  SPU_ENFORCE(commit_and_open(comm->lctx(), seed_str, &all_seed_strs));
  SPU_ENFORCE(all_seed_strs.size() == comm->getWorldSize());
  uint128_t ret = 0;
  for (auto& str : all_seed_strs) {
    const uint128_t cur_seed = *reinterpret_cast<uint128_t*>(str.data());
    ret ^= cur_seed;
  }
  return ret;
}

// TinyMacCheck protocol
// Reference: https://eprint.iacr.org/2014/101.pdf
// Page 17, protocol 10.
bool TinyMacCheck(const std::shared_ptr<Communicator>& comm,
                  std::vector<bool> open_bits, const AuthBit& bits) {
  // WARNING: As of now, uint128_t does NOT support gfmul128
  const size_t size = open_bits.size();
  // generate the coefficient for "almost universal hash"
  const uint128_t seed = GenSharedSeed(comm);
  std::vector<emp::block> coeff(size);
  emp::uni_hash_coeff_gen(coeff.data(), U128ToBlock(seed), size);

  emp::block ret_val = emp::zero_block;
  std::vector<emp::block> mac(size);
  for (size_t i = 0; i < size; ++i) {
    // convert uint128_t to emp::block
    mac[i] = U128ToBlock(bits.mac[i]);
    if (open_bits[i]) {
      ret_val = ret_val ^ coeff[i];
    }
  }
  emp::block _sigma = emp::zero_block;
  // inner product over gf128
  emp::vector_inn_prdt_sum_red(&_sigma, coeff.data(), mac.data(), size);
  emp::block offset = emp::zero_block;
  emp::gfmul(ret_val, U128ToBlock(bits.key), &offset);
  uint128_t sigma = BlockToU128(_sigma ^ offset);

  std::string sigma_str(reinterpret_cast<char*>(&sigma), sizeof(sigma));
  std::vector<std::string> all_sigma_strs;
  SPU_ENFORCE(commit_and_open(comm->lctx(), sigma_str, &all_sigma_strs));
  SPU_ENFORCE(all_sigma_strs.size() == comm->getWorldSize());

  uint128_t simga_sum = 0;
  for (auto& str : all_sigma_strs) {
    const uint128_t _sigma = *reinterpret_cast<uint128_t*>(str.data());
    simga_sum ^= _sigma;
  }
  return simga_sum == 0;
}

AuthBit RandomBits(const std::shared_ptr<Communicator>& comm,
                   const std::shared_ptr<yacl::crypto::OtSendStore>& send_opts,
                   const std::shared_ptr<yacl::crypto::OtRecvStore>& recv_opts,
                   size_t size, uint128_t tinyot_key) {
  AuthBit local_bits;
  AuthBit remote_bits;
  if (comm->getRank() == 0) {
    local_bits = AuthBitSender(comm, send_opts, size, tinyot_key);
    remote_bits = AuthBitReceiver(comm, recv_opts, size, tinyot_key);
  } else {
    remote_bits = AuthBitReceiver(comm, recv_opts, size, tinyot_key);
    local_bits = AuthBitSender(comm, send_opts, size, tinyot_key);
  }
  for (size_t k = 0; k < size; ++k) {
    local_bits.choices[k] = local_bits.choices[k] ^ remote_bits.choices[k];
    local_bits.mac[k] ^= remote_bits.mac[k];
  }
  return local_bits;
}

// Reference: https://eprint.iacr.org/2014/101.pdf
// Page 10, Protocol 7.
std::tuple<AuthBit, AuthBit, AuthBit> TinyMul(
    const std::shared_ptr<Communicator>& comm,
    const std::shared_ptr<yacl::crypto::OtSendStore>& send_opts,
    const std::shared_ptr<yacl::crypto::OtRecvStore>& recv_opts, size_t size,
    uint128_t tinyot_key) {
  AuthBit a = RandomBits(comm, send_opts, recv_opts, size, tinyot_key);
  AuthBit b = RandomBits(comm, send_opts, recv_opts, size, tinyot_key);

  auto [e, chi0, chi1, z] = GaOT(comm, send_opts, recv_opts, size, tinyot_key);

  AuthBit f{std::vector<bool>(size), std::vector<uint128_t>(size), tinyot_key};
  AuthBit g{std::vector<bool>(size), std::vector<uint128_t>(size), tinyot_key};
  for (size_t k = 0; k < size; ++k) {
    f.choices[k] = b.choices[k] ^ e.choices[k];
    g.choices[k] = chi0.choices[k] ^ chi1.choices[k] ^ a.choices[k];
    f.mac[k] = b.mac[k] ^ e.mac[k];
    g.mac[k] = chi0.mac[k] ^ chi1.mac[k] ^ a.mac[k];
  }

  std::vector<uint128_t> f_buf((size + 127) / 128, 0);
  std::vector<uint128_t> g_buf((size + 127) / 128, 0);
  for (size_t k = 0; k < size; ++k) {
    if (f.choices[k]) {
      SetBit(f_buf, k);
    }
    if (g.choices[k]) {
      SetBit(g_buf, k);
    }
  }

  f_buf = comm->allReduce<uint128_t, std::bit_xor>(f_buf, "TinyOT:Mul_f");
  g_buf = comm->allReduce<uint128_t, std::bit_xor>(g_buf, "TinyOT:Mul_g");

  std::vector<bool> true_f(size);
  std::vector<bool> true_g(size);

  for (size_t k = 0; k < size; ++k) {
    if (GetBit(f_buf, k)) {
      true_f[k] = true;
    }
    if (GetBit(g_buf, k)) {
      true_g[k] = true;
    }
  }
  // Theoretical, we could try bits f and g together in a single TinyMacCheck
  TinyMacCheck(comm, true_f, f);
  TinyMacCheck(comm, true_g, g);

  AuthBit c{std::vector<bool>(size), std::vector<uint128_t>(size), tinyot_key};
  for (size_t k = 0; k < size; ++k) {
    c.choices[k] = chi0.choices[k] ^ (GetBit(f_buf, k) * a.choices[k]) ^
                   (GetBit(g_buf, k) * e.choices[k]) ^ z.choices[k];
    c.mac[k] = chi0.mac[k] ^ (GetBit(f_buf, k) * a.mac[k]) ^
               (GetBit(g_buf, k) * e.mac[k]) ^ z.mac[k];
  }
  return {a, b, c};
}

};  // namespace spu::mpc::spdz2k