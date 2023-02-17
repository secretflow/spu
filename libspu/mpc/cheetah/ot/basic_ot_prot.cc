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

#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"

#include "libspu/mpc/cheetah/ot/util.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/semi2k/type.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {

using BShrTy = semi2k::BShrTy;

BasicOTProtocols::BasicOTProtocols(std::shared_ptr<Communicator> conn)
    : conn_(std::move(conn)) {
  SPU_ENFORCE(conn_ != nullptr);
  if (conn_->getRank() == 0) {
    ferret_sender_ = std::make_shared<FerretOT>(conn_, true);
    ferret_receiver_ = std::make_shared<FerretOT>(conn_, false);
  } else {
    ferret_receiver_ = std::make_shared<FerretOT>(conn_, false);
    ferret_sender_ = std::make_shared<FerretOT>(conn_, true);
  }
}

std::unique_ptr<BasicOTProtocols> BasicOTProtocols::Fork() {
  // TODO(juhou) we can take from cached ROTs from the caller
  auto conn = std::make_shared<Communicator>(conn_->lctx()->Spawn());
  return std::make_unique<BasicOTProtocols>(conn);
}

BasicOTProtocols::~BasicOTProtocols() { Flush(); }

void BasicOTProtocols::Flush() {
  if (ferret_sender_) {
    ferret_sender_->Flush();
  }
}

ArrayRef BasicOTProtocols::B2A(const ArrayRef &inp) {
  const auto *share_t = inp.eltype().as<BShrTy>();
  if (share_t->nbits() == 1) {
    return SingleB2A(inp);
  }
  return PackedB2A(inp);
}

// Random bit r \in {0, 1} and return as AShr
ArrayRef BasicOTProtocols::RandBits(FieldType filed, size_t numel) {
  // TODO(juhou): profile ring_randbit performance
  auto r = ring_randbit(filed, numel).as(makeType<BShrTy>(filed, 1));
  return SingleB2A(r);
}

ArrayRef BasicOTProtocols::PackedB2A(const ArrayRef &inp) {
  const auto *share_t = inp.eltype().as<BShrTy>();
  auto field = inp.eltype().as<Ring2k>()->field();
  const size_t nbits = share_t->nbits();
  SPU_ENFORCE(nbits > 0 && nbits <= 8 * SizeOf(field));

  auto convert_from_bits_form = [&](ArrayRef bform) {
    const size_t n = bform.numel() / nbits;
    // init as all 0s.
    auto iform = ring_zeros(field, n);
    DISPATCH_ALL_FIELDS(field, "", [&]() {
      auto xb = ArrayView<const ring2k_t>(bform);
      auto xi = ArrayView<ring2k_t>(iform);
      SPU_ENFORCE(xb.isCompact());
      for (size_t i = 0; i < n; ++i) {
        // LSB is bits[0]; MSB is bits[nbits - 1]
        // We use reverse_iterator to iterate the bits.
        auto bits = xb.data() + i * nbits;
        std::reverse_iterator<const ring2k_t *> rbits(bits + nbits);
        xi[i] = std::accumulate(rbits, rbits + nbits,
                                /*init*/ static_cast<ring2k_t>(0),
                                [](ring2k_t init, ring2k_t next) {
                                  return (init << 1) | (next & 1);
                                });
      }
    });
    return iform;
  };

  const size_t n = inp.numel();
  auto rand_bits = RandBits(field, n * nbits);
  auto rand = convert_from_bits_form(rand_bits);

  // open c = x ^ r
  // FIXME(juhou): Actually, we only want to exchange the low-end bits.
  auto opened =
      conn_->allReduce(ReduceOp::XOR, ring_xor(inp, rand), "B2AFull_open");

  // compute c + (1 - 2*c)*<r>
  ArrayRef oup = ring_zeros(field, n);
  DISPATCH_ALL_FIELDS(field, "", [&]() {
    using u2k = std::make_unsigned<ring2k_t>::type;
    int rank = Rank();
    auto xr = ArrayView<const u2k>(rand_bits);
    auto xc = ArrayView<const u2k>(opened);
    auto xo = ArrayView<ring2k_t>(oup);

    for (size_t i = 0; i < n; ++i) {
      auto rbits = xr.data() + i * nbits;
      u2k this_elt = xc[i];
      for (size_t j = 0; j < nbits; ++j, this_elt >>= 1) {
        u2k c_ij = this_elt & 1;
        ring2k_t one_bit = (1 - c_ij * 2) * rbits[j];
        if (rank == 0) {
          one_bit += c_ij;
        }
        xo[i] += (one_bit << j);
      }
    }
  });
  return oup;
}

ArrayRef BasicOTProtocols::SingleB2A(const ArrayRef &inp) {
  // Math:
  //   b0^b1 = b0 + b1 - 2*b0*b1
  // Sender set corr = -2*b0
  // Recv set choice b1
  // Sender gets x0
  // Recv gets x1 = x0 + corr*b1 = x0 - 2*b0*b1
  //
  //   b0 - x0 + b1 + x1
  // = b0 - x0 + b1 + x0 - 2*b0*b1
  const auto *share_t = inp.eltype().as<BShrTy>();
  SPU_ENFORCE_EQ(share_t->nbits(), 1UL);
  auto field = inp.eltype().as<Ring2k>()->field();
  const size_t n = inp.numel();

  ArrayRef oup = ring_zeros(field, n);
  DISPATCH_ALL_FIELDS(field, "", [&]() {
    using u2k = std::make_unsigned<ring2k_t>::type;
    auto xinp = ArrayView<const u2k>(inp);
    auto xoup = ArrayView<u2k>(oup);
    SPU_ENFORCE(xoup.isCompact());

    if (Rank() == 0) {
      std::vector<u2k> corr_data(n);
      // NOTE(juhou): Masking to make sure there is only single bit.
      for (size_t i = 0; i < n; ++i) {
        // corr=-2*xi
        corr_data[i] = -((xinp[i] & 1) << 1);
      }
      ferret_sender_->SendCAMCC(absl::MakeSpan(corr_data), {xoup.data(), n});
      ferret_sender_->Flush();

      for (size_t i = 0; i < n; ++i) {
        xoup[i] = (xinp[i] & 1) - xoup[i];
      }
    } else {
      std::vector<uint8_t> choices(n);
      for (size_t i = 0; i < n; ++i) {
        choices[i] = static_cast<uint8_t>(xinp[i] & 1);
      }
      ferret_receiver_->RecvCAMCC(absl::MakeSpan(choices), {xoup.data(), n});

      for (size_t i = 0; i < n; ++i) {
        xoup[i] = (xinp[i] & 1) + xoup[i];
      }
    }
  });

  return oup;
}

ArrayRef BasicOTProtocols::BitwiseAnd(const ArrayRef &lhs,
                                      const ArrayRef &rhs) {
  SPU_ENFORCE_EQ(lhs.numel(), rhs.numel());

  auto field = lhs.eltype().as<Ring2k>()->field();
  const auto *shareType = lhs.eltype().as<semi2k::BShrTy>();
  size_t size = lhs.numel();
  auto [a, b, c] = AndTriple(field, size, shareType->nbits());

  // open x^a, y^b
  auto x_a = conn_->allReduce(ReduceOp::XOR, ring_xor(lhs, a), "BitwiseAnd");
  auto y_b = conn_->allReduce(ReduceOp::XOR, ring_xor(rhs, b), "BitwiseAnd");
  // Zi = Ci ^ ((X ^ A) & Bi) ^ ((Y ^ B) & Ai) ^ <(X ^ A) & (Y ^ B)>
  auto z = ring_xor(ring_xor(ring_and(x_a, b), ring_and(y_b, a)), c);
  if (conn_->getRank() == 0) {
    ring_xor_(z, ring_and(x_a, y_b));
  }

  return z.as(lhs.eltype());
}

std::array<ArrayRef, 3> BasicOTProtocols::AndTriple(FieldType field,
                                                    size_t numel,
                                                    size_t nbits_each) {
  SPU_ENFORCE(numel > 0);
  SPU_ENFORCE(nbits_each >= 1 && nbits_each <= SizeOf(field) * 8,
              "invalid packing load {} for one AND", nbits_each);

  // NOTE(juhou): we use uint8_t to store 1-bit ROT
  constexpr size_t ot_msg_width = 1;
  std::vector<uint8_t> a(numel * nbits_each);
  std::vector<uint8_t> b(numel * nbits_each);
  std::vector<uint8_t> v(numel * nbits_each);
  std::vector<uint8_t> u(numel * nbits_each);
  if (0 == Rank()) {
    ferret_receiver_->RecvRMRC(absl::MakeSpan(a), absl::MakeSpan(u),
                               ot_msg_width);
    ferret_sender_->SendRMRC(absl::MakeSpan(v), absl::MakeSpan(b),
                             ot_msg_width);
    ferret_sender_->Flush();
  } else {
    ferret_sender_->SendRMRC(absl::MakeSpan(v), absl::MakeSpan(b),
                             ot_msg_width);
    ferret_sender_->Flush();
    ferret_receiver_->RecvRMRC(absl::MakeSpan(a), absl::MakeSpan(u),
                               ot_msg_width);
  }

  std::vector<uint8_t> c(numel * nbits_each);
  pforeach(0, c.size(), [&](int64_t i) {
    b[i] = b[i] ^ v[i];
    c[i] = (a[i] & b[i]) ^ u[i] ^ v[i];
  });

  // init as zero
  ArrayRef AND_a = ring_zeros(field, numel);
  ArrayRef AND_b = ring_zeros(field, numel);
  ArrayRef AND_c = ring_zeros(field, numel);
  DISPATCH_ALL_FIELDS(field, "AndTriple", [&]() {
    auto AND_xa = ArrayView<ring2k_t>(AND_a);
    auto AND_xb = ArrayView<ring2k_t>(AND_b);
    auto AND_xc = ArrayView<ring2k_t>(AND_c);
    pforeach(0, numel, [&](int64_t i) {
      int64_t bgn = i * nbits_each;
      int64_t end = bgn + nbits_each;
      for (int64_t j = bgn; j < end; ++j) {
        AND_xa[i] = (AND_xa[i] << 1) | (a[j] & 1);
        AND_xb[i] = (AND_xb[i] << 1) | (b[j] & 1);
        AND_xc[i] = (AND_xc[i] << 1) | (c[j] & 1);
      }
    });
  });

  return {AND_a, AND_b, AND_c};
}

int BasicOTProtocols::Rank() const { return ferret_sender_->Rank(); }

ArrayRef BasicOTProtocols::Multiplexer(const ArrayRef &msg,
                                       const ArrayRef &select) {
  SPU_ENFORCE_EQ(msg.numel(), select.numel());
  const auto *shareType = select.eltype().as<semi2k::BShrTy>();
  SPU_ENFORCE_EQ(shareType->nbits(), 1UL);

  const auto field = msg.eltype().as<Ring2k>()->field();
  const size_t size = msg.numel();

  ArrayRef _corr_data = ring_zeros(field, size);
  ArrayRef _sent = ring_zeros(field, size);
  ArrayRef _recv = ring_zeros(field, size);
  std::vector<uint8_t> sel(size);
  // Compute (x0 + x1) * (b0 ^ b1)
  // Also b0 ^ b1 = 1 - 2*b0*b1
  return DISPATCH_ALL_FIELDS(field, "Multiplexer", [&]() {
    ArrayView<const ring2k_t> _msg(msg);
    ArrayView<const ring2k_t> _sel(select);
    ArrayView<ring2k_t> corr_data(_corr_data);
    ArrayView<ring2k_t> sent(_sent);
    ArrayView<ring2k_t> recv(_recv);

    pforeach(0, size, [&](int64_t i) {
      sel[i] = static_cast<uint8_t>(_sel[i] & 1);
      corr_data[i] = _msg[i] * (1 - 2 * sel[i]);
    });

    if (Rank() == 0) {
      ferret_sender_->SendCAMCC({corr_data.data(), size}, {sent.data(), size});
      ferret_sender_->Flush();
      ferret_receiver_->RecvCAMCC(absl::MakeSpan(sel), {recv.data(), size});
    } else {
      ferret_receiver_->RecvCAMCC(absl::MakeSpan(sel), {recv.data(), size});
      ferret_sender_->SendCAMCC({corr_data.data(), size}, {sent.data(), size});
      ferret_sender_->Flush();
    }

    pforeach(0, size, [&](int64_t i) {
      recv[i] = _msg[i] * static_cast<ring2k_t>(sel[i]) - sent[i] + recv[i];
    });

    return _recv;
  });
}
}  // namespace spu::mpc::cheetah
