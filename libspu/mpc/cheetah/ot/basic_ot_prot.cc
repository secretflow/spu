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

#include "ot_util.h"

#include "libspu/mpc/cheetah/ot/emp/ferret.h"
#include "libspu/mpc/cheetah/ot/ot_util.h"
#include "libspu/mpc/cheetah/ot/yacl/ferret.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {

BasicOTProtocols::BasicOTProtocols(std::shared_ptr<Communicator> conn,
                                   CheetahOtKind kind)
    : conn_(std::move(conn)) {
  SPU_ENFORCE(conn_ != nullptr);

  if (kind == CheetahOtKind::EMP_Ferret) {
    using Ot = EmpFerretOt;
    if (conn_->getRank() == 0) {
      ferret_sender_ = std::make_shared<Ot>(conn_, true);
      ferret_receiver_ = std::make_shared<Ot>(conn_, false);
    } else {
      ferret_receiver_ = std::make_shared<Ot>(conn_, false);
      ferret_sender_ = std::make_shared<Ot>(conn_, true);
    }
  } else {
    using Ot = YaclFerretOt;
    bool use_ss = (kind == CheetahOtKind::YACL_Softspoken);
    if (conn_->getRank() == 0) {
      ferret_sender_ = std::make_shared<Ot>(conn_, true, use_ss);
      ferret_receiver_ = std::make_shared<Ot>(conn_, false, use_ss);
    } else {
      ferret_receiver_ = std::make_shared<Ot>(conn_, false, use_ss);
      ferret_sender_ = std::make_shared<Ot>(conn_, true, use_ss);
    }
  }
}

BasicOTProtocols::~BasicOTProtocols() { Flush(); }

void BasicOTProtocols::Flush() {
  if (ferret_sender_) {
    ferret_sender_->Flush();
  }
}

MemRef BasicOTProtocols::B2A(const MemRef &inp) {
  const auto *share_t = inp.eltype().as<BoolShareTy>();
  if (share_t->valid_bits() == 1) {
    return SingleB2A(inp);
  }
  return PackedB2A(inp);
}

// Convert the packed boolean shares to arithmetic share
// Input x in Z2k is the packed of b-bits for 1 <= b <= k.
//       That is x0, x1, ..., x{b-1}
// Output y in Z2k such that y = \sum_i x{i}*2^i mod 2^k
//
// Ref: The ABY paper https://encrypto.de/papers/DSZ15.pdf Section E
MemRef BasicOTProtocols::PackedB2A(const MemRef &inp) {
  const auto *share_t = inp.eltype().as<BoolShareTy>();
  const auto field = SizeOf(inp.eltype().storage_type()) * 8;
  const int64_t ring_width = SizeOf(field) * 8;

  const int64_t n = inp.numel();
  int64_t nbits = share_t->valid_bits() == 0 ? 1 : share_t->valid_bits();
  const int64_t numel = n * nbits;

  MemRef cot_oup(makeType<RingTy>(SE_INVALID, field), {numel});
  MemRef arith_oup(makeType<RingTy>(SE_INVALID, field), inp.shape());

  ring_zeros(cot_oup);
  ring_zeros(arith_oup);

  DISPATCH_ALL_STORAGE_TYPES(inp.eltype().storage_type(), [&]() {
    using u2k = std::make_unsigned<ScalarT>::type;
    auto input = MemRefView<const u2k>(inp);
    auto cot_output = absl::MakeSpan(&cot_oup.at<u2k>(0), cot_oup.numel());

    if (Rank() == 0) {
      std::vector<u2k> corr_data(numel);

      for (int64_t k = 0; k < nbits; ++k) {
        int64_t i = k * n;
        auto msk = makeBitsMask<u2k>(ring_width - k);
        for (int64_t j = 0; j < n; ++j) {
          // corr[k] = -2*x0_k
          corr_data[i + j] = -2 * ((input[j] >> k) & 1);
          corr_data[i + j] &= msk;
        }
      }
      // Run the multiple COT in the collapse mode.
      // That is, the k-th COT returns output of `ring_width - k` bits.
      //
      // The k-th COT gives the arithmetic share of the k-th bit of the input
      // according to x_0 ^ x_1 = x_0 + x_1 - 2 * x_0 * x_1
      ferret_sender_->SendCAMCC_Collapse(absl::MakeSpan(corr_data), cot_output,
                                         /*bw*/ ring_width,
                                         /*num_level*/ nbits);

      ferret_sender_->Flush();
      for (int64_t k = 0; k < nbits; ++k) {
        int64_t i = k * n;
        for (int64_t j = 0; j < n; ++j) {
          cot_output[i + j] = ((input[j] >> k) & 1) - cot_output[i + j];
        }
      }
    } else {
      // choice[k] is the k-th bit x1_k
      std::vector<uint8_t> choices(numel);
      for (int64_t k = 0; k < nbits; ++k) {
        int64_t i = k * n;
        for (int64_t j = 0; j < n; ++j) {
          choices[i + j] = (input[j] >> k) & 1;
        }
      }

      ferret_receiver_->RecvCAMCC_Collapse(absl::MakeSpan(choices), cot_output,
                                           ring_width, nbits);

      for (int64_t k = 0; k < nbits; ++k) {
        int64_t i = k * n;
        for (int64_t j = 0; j < n; ++j) {
          cot_output[i + j] = ((input[j] >> k) & 1) + cot_output[i + j];
        }
      }
    }

    // <x> = \sum_k 2^k * <x_k>
    // where <x_k> is the arithmetic share of the k-th bit
    MemRefView<u2k> arith(arith_oup);
    for (int64_t k = 0; k < nbits; ++k) {
      int64_t i = k * n;
      for (int64_t j = 0; j < n; ++j) {
        arith[j] += (cot_output[i + j] << k);
      }
    }
  });

  return arith_oup;
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
MemRef BasicOTProtocols::SingleB2A(const MemRef &inp, int bit_width) {
  const auto *share_t = inp.eltype().as<BoolShareTy>();
  SPU_ENFORCE_EQ(share_t->valid_bits(), 1);
  const auto field = SizeOf(inp.eltype().storage_type()) * 8;
  if (bit_width == 0) {
    bit_width = SizeOf(field) * 8;
  }
  const int64_t n = inp.numel();

  MemRef oup(makeType<RingTy>(SE_INVALID, field), inp.shape());
  ring_zeros(oup);
  DISPATCH_ALL_STORAGE_TYPES(inp.eltype().storage_type(), [&]() {
    using u2k = std::make_unsigned<ScalarT>::type;
    const u2k msk = makeBitsMask<u2k>(bit_width);
    auto input = MemRefView<const u2k>(inp);
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
        output[i] = ((input[i] & 1) - output[i]) & msk;
      }
    } else {
      std::vector<uint8_t> choices(n);
      for (int64_t i = 0; i < n; ++i) {
        choices[i] = static_cast<uint8_t>(input[i] & 1);
      }
      ferret_receiver_->RecvCAMCC(absl::MakeSpan(choices), output, bit_width);

      for (int64_t i = 0; i < n; ++i) {
        output[i] = ((input[i] & 1) + output[i]) & msk;
      }
    }
  });
  return oup;
}

// Random bit r \in {0, 1} and return as AShr
MemRef BasicOTProtocols::RandBits(size_t field, const Shape &shape) {
  MemRef r(makeType<BoolShareTy>(SE_INVALID, GetStorageType(field), 1), shape);
  ring_randbit(r);
  return SingleB2A(r);
}

MemRef BasicOTProtocols::B2ASingleBitWithSize(const MemRef &inp,
                                              int bit_width) {
  const auto *share_t = inp.eltype().as<BoolShareTy>();
  SPU_ENFORCE(share_t->valid_bits() == 1, "Support for 1bit boolean only");
  auto field = SizeOf(inp.eltype().storage_type()) * 8;
  SPU_ENFORCE(bit_width > 1 && bit_width < (int)(8 * SizeOf(field)),
              "bit_width={} is invalid", bit_width);
  return SingleB2A(inp, bit_width);
}

MemRef BasicOTProtocols::BitwiseAnd(const MemRef &lhs, const MemRef &rhs) {
  SPU_ENFORCE_EQ(lhs.shape(), rhs.shape());
  auto field = SizeOf(lhs.eltype().storage_type()) * 8;
  const auto *shareType = lhs.eltype().as<BoolShareTy>();
  auto [a, b, c] = AndTriple(field, lhs.shape(), shareType->valid_bits());

  // open x^a, y^b
  int nbits = shareType->valid_bits();
  auto xa = OpenShare(ring_xor(lhs, a), ReduceOp::XOR, nbits, conn_);
  auto yb = OpenShare(ring_xor(rhs, b), ReduceOp::XOR, nbits, conn_);

  // Zi = Ci ^ ((X ^ A) & Bi) ^ ((Y ^ B) & Ai) ^ <(X ^ A) & (Y ^ B)>
  auto z = ring_xor(ring_xor(ring_and(xa, b), ring_and(yb, a)), c);
  if (conn_->getRank() == 0) {
    ring_xor_(z, ring_and(xa, yb));
  }

  return z.as(lhs.eltype());
}

std::array<MemRef, 2> BasicOTProtocols::CorrelatedBitwiseAnd(
    const MemRef &lhs, const MemRef &rhs0, const MemRef &rhs1) {
  SPU_ENFORCE_EQ(lhs.shape(), rhs0.shape());
  SPU_ENFORCE(lhs.eltype() == rhs0.eltype());
  SPU_ENFORCE_EQ(lhs.shape(), rhs1.shape());
  SPU_ENFORCE(lhs.eltype() == rhs1.eltype());

  auto field = SizeOf(lhs.eltype().storage_type()) * 8;

  const auto *shareType = lhs.eltype().as<BoolShareTy>();
  SPU_ENFORCE_EQ(shareType->valid_bits(), 1);
  auto [a, b0, c0, b1, c1] = CorrelatedAndTriple(field, lhs.shape());

  // open x^a, y^b0, y1^b1
  int nbits = shareType->valid_bits();
  auto xa = OpenShare(ring_xor(lhs, a), ReduceOp::XOR, nbits, conn_);
  auto y0b0 = OpenShare(ring_xor(rhs0, b0), ReduceOp::XOR, nbits, conn_);
  auto y1b1 = OpenShare(ring_xor(rhs1, b1), ReduceOp::XOR, nbits, conn_);

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
// ROT receiver obtains x_a, a for a \in {0, 1}
//
// Sender set (b = x0 ^ x1, v = x0)
// Receiver set (a, u = x_a)
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
std::array<MemRef, 3> BasicOTProtocols::AndTriple(size_t field,
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
  MemRef AND_a(makeType<RingTy>(SE_INVALID, field), shape);
  MemRef AND_b(makeType<RingTy>(SE_INVALID, field), shape);
  MemRef AND_c(makeType<RingTy>(SE_INVALID, field), shape);
  ring_zeros(AND_a);
  ring_zeros(AND_b);
  ring_zeros(AND_c);

  DISPATCH_ALL_STORAGE_TYPES(GetStorageType(field), [&]() {
    auto AND_xa = MemRefView<ScalarT>(AND_a);
    auto AND_xb = MemRefView<ScalarT>(AND_b);
    auto AND_xc = MemRefView<ScalarT>(AND_c);
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

std::array<MemRef, 5> BasicOTProtocols::CorrelatedAndTriple(
    size_t field, const Shape &shape) {
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

  MemRef AND_a(makeType<RingTy>(SE_INVALID, field), shape);
  MemRef AND_b0(makeType<RingTy>(SE_INVALID, field), shape);
  MemRef AND_c0(makeType<RingTy>(SE_INVALID, field), shape);
  MemRef AND_b1(makeType<RingTy>(SE_INVALID, field), shape);
  MemRef AND_c1(makeType<RingTy>(SE_INVALID, field), shape);

  ring_zeros(AND_a);
  ring_zeros(AND_b0);
  ring_zeros(AND_c0);
  ring_zeros(AND_b1);
  ring_zeros(AND_c1);

  DISPATCH_ALL_STORAGE_TYPES(GetStorageType(field), [&]() {
    auto AND_xa = MemRefView<ScalarT>(AND_a);
    auto AND_xb0 = MemRefView<ScalarT>(AND_b0);
    auto AND_xc0 = MemRefView<ScalarT>(AND_c0);
    auto AND_xb1 = MemRefView<ScalarT>(AND_b1);
    auto AND_xc1 = MemRefView<ScalarT>(AND_c1);
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

MemRef BasicOTProtocols::Multiplexer(const MemRef &msg, const MemRef &select) {
  SPU_ENFORCE_EQ(msg.shape(), select.shape());
  const auto *shareType = select.eltype().as<BoolShareTy>();
  SPU_ENFORCE_EQ(shareType->valid_bits(), 1);

  const auto field = SizeOf(msg.eltype().storage_type()) * 8;
  const int64_t size = msg.numel();

  MemRef _corr_data(makeType<RingTy>(SE_INVALID, field), msg.shape());
  MemRef _sent(makeType<RingTy>(SE_INVALID, field), msg.shape());
  MemRef _recv(makeType<RingTy>(SE_INVALID, field), msg.shape());
  ring_zeros(_corr_data);
  ring_zeros(_sent);
  ring_zeros(_recv);

  std::vector<uint8_t> sel(size);
  // Compute (x0 + x1) * (b0 ^ b1)
  // Also b0 ^ b1 = 1 - 2*b0*b1
  return DISPATCH_ALL_STORAGE_TYPES(GetStorageType(field), [&]() {
    using ring2k_t = ScalarT;
    MemRefView<const ring2k_t> _msg(msg);
    MemRefView<const ring2k_t> _sel(select);
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

MemRef BasicOTProtocols::PrivateMulxRecv(const MemRef &msg,
                                         const MemRef &select) {
  SPU_ENFORCE_EQ(msg.shape(), select.shape());
  SPU_ENFORCE(msg.eltype().isa<BaseRingType>());
  SPU_ENFORCE(select.eltype().isa<BaseRingType>());

  const auto field = SizeOf(msg.eltype().storage_type()) * 8;
  const int64_t size = msg.numel();

  MemRef recv(makeType<RingTy>(SE_INVALID, field), msg.shape());
  ring_zeros(recv);
  std::vector<uint8_t> sel(size);
  DISPATCH_ALL_STORAGE_TYPES(select.eltype().storage_type(), [&]() {
    MemRefView<const ScalarT> _sel(select);
    pforeach(0, size,
             [&](int64_t i) { sel[i] = static_cast<uint8_t>(_sel[i] & 1); });
  });
  return PrivateMulxRecv(msg, absl::MakeConstSpan(sel));
}

MemRef BasicOTProtocols::PrivateMulxRecv(const MemRef &msg,
                                         absl::Span<const uint8_t> select) {
  SPU_ENFORCE_EQ(msg.numel(), (int64_t)select.size());
  SPU_ENFORCE(msg.eltype().isa<BaseRingType>());

  const auto field = SizeOf(msg.eltype().storage_type()) * 8;
  const int64_t size = msg.numel();

  MemRef recv(makeType<RingTy>(SE_INVALID, field), msg.shape());
  ring_zeros(recv);
  // Compute (x0 + x1) * b
  // x0 * b + x1 * b
  // COT compute <x1*b>
  DISPATCH_ALL_STORAGE_TYPES(msg.eltype().storage_type(), [&]() {
    MemRefView<const ScalarT> _msg(msg);
    auto _recv = absl::MakeSpan(&recv.at<ScalarT>(0), size);

    ferret_receiver_->RecvCAMCC(select, _recv);

    pforeach(0, size, [&](int64_t i) {
      _recv[i] = _msg[i] * static_cast<ScalarT>(select[i] & 1) + _recv[i];
    });
  });

  return recv.as(msg.eltype());
}

MemRef BasicOTProtocols::PrivateMulxSend(const MemRef &msg) {
  SPU_ENFORCE(msg.isCompact());

  const auto field = SizeOf(msg.eltype().storage_type()) * 8;
  const int64_t size = msg.numel();

  MemRef recv(makeType<RingTy>(SE_INVALID, field), msg.shape());
  ring_zeros(recv);
  // Compute (x0 + x1) * b
  // x0 * b + x1 * b
  DISPATCH_ALL_STORAGE_TYPES(GetStorageType(field), [&]() {
    auto _msg = absl::MakeConstSpan(&msg.at<ScalarT>(0), size);
    auto _recv = absl::MakeSpan(&recv.at<ScalarT>(0), size);

    ferret_sender_->SendCAMCC(_msg, _recv);
    ferret_sender_->Flush();

    pforeach(0, size, [&](int64_t i) { _recv[i] = -_recv[i]; });
  });

  return recv.as(msg.eltype());
}

}  // namespace spu::mpc::cheetah
