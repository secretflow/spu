// Copyright 2023 zhangwfjh
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

#include "libspu/psi/core/fnp04_mp_psi/fnp04_mp_psi.h"

#include <algorithm>
#include <iterator>
#include <numeric>
#include <random>
#include <utility>

#include "yacl/utils/serialize.h"

#include "libspu/core/prelude.h"
#include "libspu/psi/utils/utils.h"

namespace {

static std::random_device rd{};
static std::independent_bits_engine<std::default_random_engine,
                                    sizeof(size_t) * 8, size_t>
    rng{rd()};

constexpr size_t BinCountHint(size_t size) {
  auto lnln_size = std::max(1., std::log(std::log(size)));
  return (size + lnln_size - 1) / lnln_size;
}

size_t ToUnsigned(const yacl::math::MPInt& x) {
  unsigned char bytes[sizeof(size_t)];
  x.ToBytes(bytes, sizeof(size_t));
  return *reinterpret_cast<size_t*>(bytes);
}

yacl::Buffer Serialize(size_t size) {
  yacl::Buffer buf(sizeof(size_t));
  std::memcpy(buf.data(), &size, sizeof(size_t));
  return buf;
}

size_t Deserialize(const yacl::Buffer& buf) {
  size_t size;
  std::memcpy(&size, buf.data(), sizeof(size_t));
  return size;
}

}  // namespace

namespace spu::psi {

FNP04Party::FNP04Party(const Options& options) : options_{options} {
  auto [ctx, wsize, me, leader] = CollectContext();
  encryptors_.resize(wsize);
  SecretKey sk;
  PublicKey pk;
  KeyGenerator::Generate(&sk, &pk);
  encryptors_[me] = std::make_shared<Encryptor>(pk);
  decryptor_ = std::make_shared<Decryptor>(pk, sk);
}

std::vector<std::string> FNP04Party::Run(
    const std::vector<std::string>& inputs) {
  auto [ctx, wsize, me, leader] = CollectContext();
  auto counts = AllGatherItemsSize(ctx, inputs.size());
  size_t count{};
  for (auto cnt : counts) {
    if (cnt == 0) {
      return {};
    }
    count = std::max(cnt, count);
  }
  // Step 0: Encode the inputs
  auto items = EncodeInputs(inputs, count);
  // Step 1: Broadcast the public key
  BroadcastPubKey();
  // Step 2: Zero sharing
  auto shares = ZeroSharing(count);
  if (leader == me) {
    for (size_t i{}; i != count; ++i) {
      shares[i][(leader + 1) % wsize] ^= items[i];
    }
  }
  if (leader == me) {
    // Step 3: Receive the encrypted set
    auto hashings = RecvEncryptedSet(count);
    // Step 4: Swap the encrypted shares
    SwapShares(shares, items, hashings);
  } else {
    // Step 3: Send the encrypted set
    SendEncryptedSet(items);
    // Step 4: Swap encrypted shares
    shares = SwapShares(shares);
  }
  // Step 5: Aggregate share
  auto aggregate = AggregateShare(shares);
  // Step 6: Get intersection
  auto intersection_items = GetIntersection(items, aggregate);
  std::vector<std::string> intersection;
  for (size_t i{}; i != inputs.size(); ++i) {
    if (std::find(intersection_items.begin(), intersection_items.end(),
                  items[i]) != intersection_items.end()) {
      intersection.emplace_back(inputs[i]);
    }
  }
  return intersection;
}

std::vector<size_t> FNP04Party::EncodeInputs(
    const std::vector<std::string>& inputs, size_t count) const {
  std::vector<size_t> items;
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(items),
                 [](std::string_view input) {
                   return std::hash<std::string_view>{}(input);
                 });
  //  Add random dummy elements
  std::generate_n(std::back_inserter(items), count - inputs.size(),
                  std::ref(rng));
  return items;
}

void FNP04Party::BroadcastPubKey() {
  auto [ctx, wsize, me, leader] = CollectContext();
  // Publish
  if (leader != me) {
    auto pk = encryptors_[me]->public_key().Serialize();
    for (size_t dst{}; dst != wsize; ++dst) {
      if (dst != me) {
        ctx->SendAsyncThrottled(
            dst, pk,
            fmt::format("Party {} sends the public key to {}", me, dst));
      }
    }
  }
  // Collect
  for (size_t src{}; src != wsize; ++src) {
    if (src != me && src != leader) {
      auto buf = ctx->Recv(
          src,
          fmt::format("Party {} receives the public key from {}", me, src));
      PublicKey pk;
      pk.Deserialize(buf);
      encryptors_[src] = std::make_shared<Encryptor>(pk);
    }
  }
}

void FNP04Party::SendEncryptedSet(const std::vector<size_t>& items) const {
  auto [ctx, wsize, me, leader] = CollectContext();
  assert(me != leader);
  // Encode
  size_t B{BinCountHint(items.size())};
  std::vector<std::vector<size_t>> hashing(B);
  std::for_each(items.begin(), items.end(),
                [&](auto item) { hashing[item % B].emplace_back(item); });
  // Hashing
  SecretPolynomial bins(B);
  std::vector<yacl::Buffer> buffers;
  for (auto& roots : hashing) {
    SPU_ENFORCE(roots.size() <= BinSize, "Severe hash collisions");
    roots.resize(BinSize);
    std::array<size_t, BinSize> coeffs{
        -(roots[0] * roots[1] * roots[2] * roots[3] * roots[4]),
        +(roots[0] * roots[1] * roots[2] * roots[3] +
          roots[0] * roots[1] * roots[2] * roots[4] +
          roots[0] * roots[1] * roots[3] * roots[4] +
          roots[0] * roots[2] * roots[3] * roots[4] +
          roots[1] * roots[2] * roots[3] * roots[4]),
        -(roots[0] * roots[1] * roots[2] + roots[0] * roots[1] * roots[3] +
          roots[0] * roots[1] * roots[4] + roots[0] * roots[2] * roots[3] +
          roots[0] * roots[2] * roots[4] + roots[0] * roots[3] * roots[4] +
          roots[1] * roots[2] * roots[3] + roots[1] * roots[2] * roots[4] +
          roots[1] * roots[3] * roots[4] + roots[2] * roots[3] * roots[4]),
        +(roots[0] * roots[1] + roots[0] * roots[2] + roots[0] * roots[3] +
          roots[0] * roots[4] + roots[1] * roots[2] + roots[1] * roots[3] +
          roots[1] * roots[4] + roots[2] * roots[3] + roots[2] * roots[4] +
          roots[3] * roots[4]),
        -(roots[0] + roots[1] + roots[2] + roots[3] + roots[4])};
    for (const auto& coeff : coeffs) {
      buffers.emplace_back(
          encryptors_[me]->Encrypt(Plaintext(coeff)).Serialize());
    }
  }
  ctx->SendAsyncThrottled(
      leader, yacl::SerializeArrayOfBuffers({buffers.begin(), buffers.end()}),
      fmt::format("Party {} sends the encrypted set", me));
}

auto FNP04Party::RecvEncryptedSet(size_t count) const
    -> std::vector<SecretPolynomial> {
  auto [ctx, wsize, me, leader] = CollectContext();
  assert(me == leader);
  size_t B{BinCountHint(count)};
  std::vector<SecretPolynomial> hashings(wsize, SecretPolynomial(B));
  for (size_t src{}; src != wsize; ++src) {
    if (src != me) {
      auto buf = ctx->Recv(
          src, fmt::format(
                   "The leader receives the encrypted set from party {}", src));
      auto buffers = yacl::DeserializeArrayOfBuffers(buf);
      size_t i{};
      for (auto& bin : hashings[src]) {
        for (auto& coeff : bin) {
          coeff.Deserialize(buffers[i++]);
        }
      }
    }
  }
  return hashings;
}

auto FNP04Party::ZeroSharing(size_t count) const -> std::vector<Share> {
  auto [ctx, wsize, me, leader] = CollectContext();
  std::vector<Share> shares(count, Share(wsize));
  for (auto& share : shares) {
    std::generate_n(share.begin(), wsize, std::ref(rng));
    auto sum = std::reduce(share.begin(), share.end(), share[leader],
                           [](const auto& x, const auto& y) { return x ^ y; });
    share[(leader + 1) % wsize] ^= sum;
    share[leader] = 0;
  }
  return shares;
}

auto FNP04Party::SwapShares(const std::vector<Share>& shares) const
    -> std::vector<Share> {
  auto [ctx, wsize, me, leader] = CollectContext();
  auto count = shares.size();
  // Send
  for (size_t dst{}; dst != wsize; ++dst) {
    if (dst != me && dst != leader) {
      std::vector<yacl::Buffer> buffers;
      for (auto& share : shares) {
        auto cipher = encryptors_[dst]->Encrypt(Plaintext(share[dst]));
        buffers.emplace_back(cipher.Serialize());
      }
      ctx->SendAsyncThrottled(
          dst, yacl::SerializeArrayOfBuffers({buffers.begin(), buffers.end()}),
          fmt::format("Party {} sends secret shares to {}", me, dst));
    }
  }
  // Receive
  std::vector<Share> recv_shares(count, Share(wsize));
  for (size_t src{}; src != wsize; ++src) {
    if (src != me) {
      auto buf = ctx->Recv(
          src, fmt::format("Party {} receives secret shares from {}", me, src));
      auto buffers = yacl::DeserializeArrayOfBuffers(buf);
      size_t i{};
      for (auto& share : recv_shares) {
        Ciphertext cipher;
        cipher.Deserialize(buffers[i++]);
        share[src] = ToUnsigned(decryptor_->Decrypt(cipher));
      }
    } else {
      for (size_t i{}; i != count; ++i) {
        recv_shares[i][me] = shares[i][me];
      }
    }
  }
  return recv_shares;
}

void FNP04Party::SwapShares(
    const std::vector<Share>& shares, const std::vector<size_t>& items,
    const std::vector<SecretPolynomial>& hashings) const {
  auto [ctx, wsize, me, leader] = CollectContext();
  auto count = items.size();
  size_t B{BinCountHint(count)};
  for (size_t dst{}; dst != wsize; ++dst) {
    if (dst != me) {
      Evaluator evaluator{encryptors_[dst]->public_key()};
      std::vector<Ciphertext> share(count);
      for (size_t i{}; i != count; ++i) {
        auto coeffs = hashings[dst][items[i] % B];
        Plaintext scale{rng()}, bias{shares[i][dst]};
        Plaintext x{items[i]};
        // share[i] = (((((x+c4).x+c3).x+c2).x+c1).x+c0).s+b
        share[i] = coeffs[4];
        evaluator.AddInplace(&share[i], x);
        evaluator.MulInplace(&share[i], x);
        evaluator.AddInplace(&share[i], coeffs[3]);
        evaluator.MulInplace(&share[i], x);
        evaluator.AddInplace(&share[i], coeffs[2]);
        evaluator.MulInplace(&share[i], x);
        evaluator.AddInplace(&share[i], coeffs[1]);
        evaluator.MulInplace(&share[i], x);
        evaluator.AddInplace(&share[i], coeffs[0]);
        evaluator.MulInplace(&share[i], scale);
        evaluator.AddInplace(&share[i], bias);
      }
      std::vector<yacl::Buffer> buffers(count);
      std::transform(share.begin(), share.end(), buffers.begin(),
                     [&](const auto& s) { return s.Serialize(); });
      ctx->SendAsyncThrottled(
          dst, yacl::SerializeArrayOfBuffers({buffers.begin(), buffers.end()}),
          fmt::format("The leader sends secret shares to {}", dst));
    }
  }
}

auto FNP04Party::AggregateShare(const std::vector<Share>& shares) const
    -> Share {
  auto [ctx, wsize, me, leader] = CollectContext();
  Share share(shares.size());
  if (me != leader) {
    for (size_t i{}; i != shares.size(); ++i) {
      for (size_t src{}; src != wsize; ++src) {
        share[i] ^= shares[i][src];
      }
    }
  }
  return share;
}

std::vector<size_t> FNP04Party::GetIntersection(
    const std::vector<size_t>& items, const Share& share) const {
  auto [ctx, wsize, me, leader] = CollectContext();
  if (me != leader) {
    std::vector<yacl::Buffer> buffers(share.size());
    std::transform(share.begin(), share.end(), buffers.begin(),
                   [&](size_t s) { return Serialize(s); });
    ctx->SendAsyncThrottled(
        leader, yacl::SerializeArrayOfBuffers({buffers.begin(), buffers.end()}),
        fmt::format("Party {} sends aggregated shares to the leader", me));
    return {};
  }
  std::vector<size_t> universe = share;
  for (size_t src{}; src != wsize; ++src) {
    if (src != me) {
      auto buf = ctx->Recv(
          src,
          fmt::format("The leader receives aggregated shares from {}", src));
      auto buffers = yacl::DeserializeArrayOfBuffers(buf);
      size_t i{};
      for (auto& item : universe) {
        item ^= Deserialize(buffers[i++]);
      }
    }
  }
  std::vector<size_t> intersection;
  std::copy_if(items.begin(), items.end(), std::back_inserter(intersection),
               [&](size_t item) {
                 return std::find(universe.begin(), universe.end(), item) !=
                        universe.end();
               });
  return intersection;
}

}  // namespace spu::psi
