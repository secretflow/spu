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

#include "libspu/mpc/cheetah/yacl_ot/basic_ot_prot.h"

#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/cheetah/yacl_ot/util.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {

BasicOTProtocols::BasicOTProtocols(std::shared_ptr<Communicator> conn)
    : conn_(std::move(conn)) {
  SPU_ENFORCE(conn_ != nullptr);
  if (conn_->getRank() == 0) {
    ferret_sender_ = std::make_shared<YaclFerretOT>(conn_, true);
    ferret_receiver_ = std::make_shared<YaclFerretOT>(conn_, false);
  } else {
    ferret_receiver_ = std::make_shared<YaclFerretOT>(conn_, false);
    ferret_sender_ = std::make_shared<YaclFerretOT>(conn_, true);
  }
}

BasicOTProtocols::~BasicOTProtocols() { Flush(); }

void BasicOTProtocols::Flush() {
  if (ferret_sender_) {
    ferret_sender_->Flush();
  }
}

NdArrayRef BasicOTProtocols::B2A(const NdArrayRef &inp) {
  const auto *share_t = inp.eltype().as<BShrTy>();
  if (share_t->nbits() == 1) {
    return SingleB2A(inp);
  }
  return PackedB2A(inp);
}

NdArrayRef BasicOTProtocols::PackedB2A(const NdArrayRef &inp) {
  const auto *share_t = inp.eltype().as<BShrTy>();
  auto field = inp.eltype().as<Ring2k>()->field();
  const size_t nbits = share_t->nbits();
  SPU_ENFORCE(nbits > 0 && nbits <= 8 * SizeOf(field));

  auto convert_from_bits_form = [&](NdArrayRef _bits) {
    SPU_ENFORCE(_bits.isCompact(), "need compact input");
    const int64_t n = _bits.numel() / nbits;
    // init as all 0s.
    auto iform = ring_zeros(field, inp.shape());
    DISPATCH_ALL_FIELDS(field, "conv_to_bits", [&]() {
      auto bits = NdArrayView<const ring2k_t>(_bits);
      auto digit = NdArrayView<ring2k_t>(iform);
      for (int64_t i = 0; i < n; ++i) {
        // LSB is bits[0]; MSB is bits[nbits - 1]
        // We iterate the bits in reversed order
        const size_t offset = i * nbits;
        digit[i] = 0;
        for (size_t j = nbits; j > 0; --j) {
          digit[i] = (digit[i] << 1) | (bits[offset + j - 1] & 1);
        }
      }
    });
    return iform;
  };

  const int64_t n = inp.numel();
  auto rand_bits = RandBits(field, Shape{n * static_cast<int>(nbits)});
  auto rand = convert_from_bits_form(rand_bits);

  // open c = x ^ r
  // FIXME(juhou): Actually, we only want to exchange the low-end bits.
  auto opened =
      conn_->allReduce(ReduceOp::XOR, ring_xor(inp, rand), "B2AFull_open");

  // compute c + (1 - 2*c)*<r>
  NdArrayRef oup = ring_zeros(field, inp.shape());
  DISPATCH_ALL_FIELDS(field, "packed_b2a", [&]() {
    using u2k = std::make_unsigned<ring2k_t>::type;
    int rank = Rank();
    auto xr = NdArrayView<const u2k>(rand_bits);
    auto xc = NdArrayView<const u2k>(opened);
    auto xo = NdArrayView<ring2k_t>(oup);

    for (int64_t i = 0; i < n; ++i) {
      const size_t offset = i * nbits;
      u2k this_elt = xc[i];
      for (size_t j = 0; j < nbits; ++j, this_elt >>= 1) {
        u2k c_ij = this_elt & 1;
        ring2k_t one_bit = (1 - c_ij * 2) * xr[offset + j];
        if (rank == 0) {
          one_bit += c_ij;
        }
        xo[i] += (one_bit << j);
      }
    }
  });
  return oup;
}

// Math:
//   b0^b1 = b0 + b1 - 2*b0*b1
// Sender set corr = -2*b0
// Recv set choice b1
// Sender gets x0
// Recv gets x1 = x0 + corr*b1 = x0 - 2*b0*b1
//
//   b0 - x0 + b1 + x1
// = b0 - x0 + b1 + x0 - 2*b0*b1
NdArrayRef BasicOTProtocols::SingleB2A(const NdArrayRef &inp, int bit_width) {
  const auto *share_t = inp.eltype().as<BShrTy>();
  SPU_ENFORCE_EQ(share_t->nbits(), 1UL);
  auto field = inp.eltype().as<Ring2k>()->field();
  if (bit_width == 0) {
    bit_width = SizeOf(field) * 8;
  }
  const int64_t n = inp.numel();

  NdArrayRef oup = ring_zeros(field, inp.shape());
  DISPATCH_ALL_FIELDS(field, "single_b2a", [&]() {
    using u2k = std::make_unsigned<ring2k_t>::type;
    auto input = NdArrayView<const u2k>(inp);
    // NOTE(lwj): oup is compact, so we just use Span
    auto output = absl::MakeSpan(&oup.at<u2k>(0), n);

    SPU_ENFORCE(oup.isCompact());

    if (Rank() == 0) {
      std::vector<u2k> corr_data(n);
      // NOTE(lwj): Masking to make sure there is only single bit.
      for (int64_t i = 0; i < n; ++i) {
        // corr=-2*xi
        corr_data[i] = -((input[i] & 1) << 1);
      }
      ferret_sender_->SendCAMCC(absl::MakeSpan(corr_data), output, bit_width);
      ferret_sender_->Flush();

      for (int64_t i = 0; i < n; ++i) {
        output[i] = (input[i] & 1) - output[i];
      }
    } else {
      std::vector<uint8_t> choices(n);
      for (int64_t i = 0; i < n; ++i) {
        choices[i] = static_cast<uint8_t>(input[i] & 1);
      }
      ferret_receiver_->RecvCAMCC(absl::MakeSpan(choices), output, bit_width);

      for (int64_t i = 0; i < n; ++i) {
        output[i] = (input[i] & 1) + output[i];
      }
    }
  });
  return oup;
}

// Random bit r \in {0, 1} and return as AShr
NdArrayRef BasicOTProtocols::RandBits(FieldType filed, const Shape &shape) {
  // TODO(juhou): profile ring_randbit performance
  auto r = ring_randbit(filed, shape).as(makeType<BShrTy>(filed, 1));
  return SingleB2A(r);
}

NdArrayRef BasicOTProtocols::B2ASingleBitWithSize(const NdArrayRef &inp,
                                                  int bit_width) {
  const auto *share_t = inp.eltype().as<BShrTy>();
  SPU_ENFORCE(share_t->nbits() == 1, "Support for 1bit boolean only");
  auto field = inp.eltype().as<Ring2k>()->field();
  SPU_ENFORCE(bit_width > 1 && bit_width < (int)(8 * SizeOf(field)),
              "bit_width={} is invalid", bit_width);
  return SingleB2A(inp, bit_width);
}

NdArrayRef BasicOTProtocols::BitwiseAnd(const NdArrayRef &lhs,
                                        const NdArrayRef &rhs) {
  SPU_ENFORCE_EQ(lhs.shape(), rhs.shape());

  auto field = lhs.eltype().as<Ring2k>()->field();
  const auto *shareType = lhs.eltype().as<BShrTy>();
  size_t numel = lhs.numel();
  auto [a, b, c] = AndTriple(field, lhs.shape(), shareType->nbits());

  NdArrayRef x_a = ring_xor(lhs, a);
  NdArrayRef y_b = ring_xor(rhs, b);
  size_t pack_load = 8 * SizeOf(field) / shareType->nbits();

  if (pack_load == 1) {
    // Open x^a, y^b
    auto res = vmap({x_a, y_b}, [&](const NdArrayRef &s) {
      return conn_->allReduce(ReduceOp::XOR, s, "BitwiseAnd");
    });
    x_a = std::move(res[0]);
    y_b = std::move(res[1]);
  } else {
    // Open x^a, y^b
    // pack multiple nbits() into single field element before sending through
    // network
    SPU_ENFORCE(x_a.isCompact() && y_b.isCompact());
    int64_t packed_sze = CeilDiv(numel, pack_load);

    NdArrayRef packed_xa(x_a.eltype(), {packed_sze});
    NdArrayRef packed_yb(y_b.eltype(), {packed_sze});

    DISPATCH_ALL_FIELDS(field, "_", [&]() {
      auto xa_wrap = absl::MakeSpan(&x_a.at<ring2k_t>(0), numel);
      auto yb_wrap = absl::MakeSpan(&y_b.at<ring2k_t>(0), numel);
      auto packed_xa_wrap =
          absl::MakeSpan(&packed_xa.at<ring2k_t>(0), packed_sze);
      auto packed_yb_wrap =
          absl::MakeSpan(&packed_yb.at<ring2k_t>(0), packed_sze);

      int64_t used =
          ZipArray<ring2k_t>(xa_wrap, shareType->nbits(), packed_xa_wrap);
      (void)ZipArray<ring2k_t>(yb_wrap, shareType->nbits(), packed_yb_wrap);
      SPU_ENFORCE_EQ(used, packed_sze);

      // open x^a, y^b
      auto res = vmap({packed_xa, packed_yb}, [&](const NdArrayRef &s) {
        return conn_->allReduce(ReduceOp::XOR, s, "BitwiseAnd");
      });

      packed_xa = std::move(res[0]);
      packed_yb = std::move(res[1]);
      packed_xa_wrap = absl::MakeSpan(&packed_xa.at<ring2k_t>(0), packed_sze);
      packed_yb_wrap = absl::MakeSpan(&packed_yb.at<ring2k_t>(0), packed_sze);
      UnzipArray<ring2k_t>(packed_xa_wrap, shareType->nbits(), xa_wrap);
      UnzipArray<ring2k_t>(packed_yb_wrap, shareType->nbits(), yb_wrap);
    });
  }

  // Zi = Ci ^ ((X ^ A) & Bi) ^ ((Y ^ B) & Ai) ^ <(X ^ A) & (Y ^ B)>
  auto z = ring_xor(ring_xor(ring_and(x_a, b), ring_and(y_b, a)), c);
  if (conn_->getRank() == 0) {
    ring_xor_(z, ring_and(x_a, y_b));
  }

  return z.as(lhs.eltype());
}

std::array<NdArrayRef, 2> BasicOTProtocols::CorrelatedBitwiseAnd(
    const NdArrayRef &lhs, const NdArrayRef &rhs0, const NdArrayRef &rhs1) {
  SPU_ENFORCE_EQ(lhs.shape(), rhs0.shape());
  SPU_ENFORCE(lhs.eltype() == rhs0.eltype());
  SPU_ENFORCE_EQ(lhs.shape(), rhs1.shape());
  SPU_ENFORCE(lhs.eltype() == rhs1.eltype());

  auto field = lhs.eltype().as<Ring2k>()->field();
  const auto *shareType = lhs.eltype().as<BShrTy>();
  SPU_ENFORCE_EQ(shareType->nbits(), 1UL);
  auto [a, b0, c0, b1, c1] = CorrelatedAndTriple(field, lhs.shape());

  // open x^a, y^b0, y1^b1
  auto res =
      vmap({ring_xor(lhs, a), ring_xor(rhs0, b0), ring_xor(rhs1, b1)},
           [&](const NdArrayRef &s) {
             return conn_->allReduce(ReduceOp::XOR, s, "CorrelatedBitwiseAnd");
           });
  auto xa = std::move(res[0]);
  auto y0b0 = std::move(res[1]);
  auto y1b1 = std::move(res[2]);

  // Zi = Ci ^ ((X ^ A) & Bi) ^ ((Y ^ B) & Ai) ^ <(X ^ A) & (Y ^ B)>
  auto z0 = ring_xor(ring_xor(ring_and(xa, b0), ring_and(y0b0, a)), c0);
  auto z1 = ring_xor(ring_xor(ring_and(xa, b1), ring_and(y1b1, a)), c1);
  if (conn_->getRank() == 0) {
    ring_xor_(z0, ring_and(xa, y0b0));
    ring_xor_(z1, ring_and(xa, y1b1));
  }

  return {z0.as(lhs.eltype()), z1.as(lhs.eltype())};
}

// Ref: https://eprint.iacr.org/2013/552.pdf
// Algorithm 1. AND triple using 1-of-2 ROT.
// Math
// ROT sender obtains x_0, x_1
// ROT recevier obtains x_a, a for a \in {0, 1}
//
// Sender set (b = x0 ^ x1, v = x0)
// Recevier set (a, u = x_a)
// a & b = a & (x0 ^ x1)
//       = a & (x0 ^ x1) ^ (x0 ^ x0) <- zero m0 ^ m0
//       = (a & (x0 ^ x1) ^ x0) ^ x0
//       = (x_a) ^ x0
//       = u ^ v
//
// P0 acts as S to obtain (a0, u0)
// P1 acts as R to obtain (b1, v1)
// such that a0 & b1 = u0 ^ v1
//
// Flip the role
// P1 obtains (a1, u1)
// P0 obtains (b0, v0)
// such that a1 & b0 = u1 ^ v0
//
// Pi sets ci = ai & bi ^ ui ^ vi
// such that (a0 ^ a1) & (b0 ^ b1) = (c0 ^ c1)
std::array<NdArrayRef, 3> BasicOTProtocols::AndTriple(FieldType field,
                                                      const Shape &shape,
                                                      size_t nbits_each) {
  int64_t numel = shape.numel();
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
  auto AND_a = ring_zeros(field, shape);
  auto AND_b = ring_zeros(field, shape);
  auto AND_c = ring_zeros(field, shape);

  DISPATCH_ALL_FIELDS(field, "AndTriple", [&]() {
    auto AND_xa = NdArrayView<ring2k_t>(AND_a);
    auto AND_xb = NdArrayView<ring2k_t>(AND_b);
    auto AND_xc = NdArrayView<ring2k_t>(AND_c);
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

std::array<NdArrayRef, 5> BasicOTProtocols::CorrelatedAndTriple(
    FieldType field, const Shape &shape) {
  int64_t numel = shape.numel();
  SPU_ENFORCE(numel > 0);
  // NOTE(juhou): we use uint8_t to store 2-bit ROT
  constexpr size_t ot_msg_width = 2;
  std::vector<uint8_t> a(numel);
  std::vector<uint8_t> b(numel);
  std::vector<uint8_t> v(numel);
  std::vector<uint8_t> u(numel);
  // random choice a is 1-bit
  // random messages b, v and u are 2-bit
  if (0 == Rank()) {
    ferret_receiver_->RecvRMRC(/*choice*/ absl::MakeSpan(a), absl::MakeSpan(u),
                               ot_msg_width);
    ferret_sender_->SendRMRC(absl::MakeSpan(v), absl::MakeSpan(b),
                             ot_msg_width);
    ferret_sender_->Flush();
  } else {
    ferret_sender_->SendRMRC(absl::MakeSpan(v), absl::MakeSpan(b),
                             ot_msg_width);
    ferret_sender_->Flush();
    ferret_receiver_->RecvRMRC(/*choice*/ absl::MakeSpan(a), absl::MakeSpan(u),
                               ot_msg_width);
  }

  std::vector<uint8_t> c(numel);
  pforeach(0, c.size(), [&](int64_t i) {
    b[i] = b[i] ^ v[i];
    // broadcast to 2-bit AND
    c[i] = (((a[i] << 1) | a[i]) & b[i]) ^ u[i] ^ v[i];
  });

  auto AND_a = ring_zeros(field, shape);
  auto AND_b0 = ring_zeros(field, shape);
  auto AND_c0 = ring_zeros(field, shape);
  auto AND_b1 = ring_zeros(field, shape);
  auto AND_c1 = ring_zeros(field, shape);

  DISPATCH_ALL_FIELDS(field, "AndTriple", [&]() {
    auto AND_xa = NdArrayView<ring2k_t>(AND_a);
    auto AND_xb0 = NdArrayView<ring2k_t>(AND_b0);
    auto AND_xc0 = NdArrayView<ring2k_t>(AND_c0);
    auto AND_xb1 = NdArrayView<ring2k_t>(AND_b1);
    auto AND_xc1 = NdArrayView<ring2k_t>(AND_c1);
    pforeach(0, numel, [&](int64_t i) {
      AND_xa[i] = a[i] & 1;
      AND_xb0[i] = b[i] & 1;
      AND_xc0[i] = c[i] & 1;
      AND_xb1[i] = (b[i] >> 1) & 1;
      AND_xc1[i] = (c[i] >> 1) & 1;
    });
  });

  return {AND_a, AND_b0, AND_c0, AND_b1, AND_c1};
}

int BasicOTProtocols::Rank() const { return ferret_sender_->Rank(); }

NdArrayRef BasicOTProtocols::Multiplexer(const NdArrayRef &msg,
                                         const NdArrayRef &select) {
  SPU_ENFORCE_EQ(msg.shape(), select.shape());
  const auto *shareType = select.eltype().as<BShrTy>();
  SPU_ENFORCE_EQ(shareType->nbits(), 1UL);

  const auto field = msg.eltype().as<Ring2k>()->field();
  const int64_t size = msg.numel();

  auto _corr_data = ring_zeros(field, msg.shape());
  auto _sent = ring_zeros(field, msg.shape());
  auto _recv = ring_zeros(field, msg.shape());
  std::vector<uint8_t> sel(size);
  // Compute (x0 + x1) * (b0 ^ b1)
  // Also b0 ^ b1 = 1 - 2*b0*b1
  return DISPATCH_ALL_FIELDS(field, "Multiplexer", [&]() {
    NdArrayView<const ring2k_t> _msg(msg);
    NdArrayView<const ring2k_t> _sel(select);
    auto corr_data = absl::MakeSpan(&_corr_data.at<ring2k_t>(0), size);
    auto sent = absl::MakeSpan(&_sent.at<ring2k_t>(0), size);
    auto recv = absl::MakeSpan(&_recv.at<ring2k_t>(0), size);

    pforeach(0, size, [&](int64_t i) {
      sel[i] = static_cast<uint8_t>(_sel[i] & 1);
      corr_data[i] = _msg[i] * (1 - 2 * sel[i]);
    });

    if (Rank() == 0) {
      ferret_sender_->SendCAMCC(corr_data, sent);
      ferret_sender_->Flush();
      ferret_receiver_->RecvCAMCC(absl::MakeSpan(sel), recv);
    } else {
      ferret_receiver_->RecvCAMCC(absl::MakeSpan(sel), recv);
      ferret_sender_->SendCAMCC(corr_data, sent);
      ferret_sender_->Flush();
    }

    pforeach(0, size, [&](int64_t i) {
      recv[i] = _msg[i] * static_cast<ring2k_t>(sel[i]) - sent[i] + recv[i];
    });

    return _recv;
  });
}

}  // namespace spu::mpc::cheetah
