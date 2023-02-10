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

#include "yacl/link/link.h"

#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/semi2k/type.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {

using BShrTy = semi2k::BShrTy;

template <typename T>
inline T CeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

inline uint8_t BoolToU8(const uint8_t *bits, size_t len = 0) {
  if (0 == len) {
    len = 8;
  } else {
    len = std::min(8UL, len);
  }

  return std::accumulate(
      bits, bits + len,
      /*init*/ static_cast<uint8_t>(0),
      [](uint8_t init, uint8_t next) { return (init << 1) | (next & 1); });
}

BasicOTProtocols::BasicOTProtocols(std::shared_ptr<yacl::link::Context> conn)
    : conn_(conn) {
  SPU_ENFORCE(conn != nullptr);
  // NOTE(juhou): we create a seperate link for OT
  // std::shared_ptr<yacl::link::Context> ot_conn{conn->Spawn()};
  if (conn->Rank() == 0) {
    ferret_sender_ = std::make_shared<FerretOT>(conn, true);
    ferret_receiver_ = std::make_shared<FerretOT>(conn, false);
  } else {
    ferret_receiver_ = std::make_shared<FerretOT>(conn, false);
    ferret_sender_ = std::make_shared<FerretOT>(conn, true);
  }
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
      YACL_ENFORCE(xb.isCompact());
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
  Communicator communicator(conn_);
  auto opened = communicator.allReduce(ReduceOp::XOR, ring_xor(inp, rand),
                                       "B2AFull_open");

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
    YACL_ENFORCE(xoup.isCompact());

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
  auto [a, b, c] = AndTriple(field, size * shareType->nbits(),
                             /*packed*/ shareType->nbits() > 1);

  Communicator comm(conn_);
  // open x^a, y^b
  auto x_a = comm.allReduce(ReduceOp::XOR, ring_xor(lhs, a), "BitwiseAnd");
  auto y_b = comm.allReduce(ReduceOp::XOR, ring_xor(rhs, b), "BitwiseAnd");
  // Zi = Ci ^ ((X ^ A) & Bi) ^ ((Y ^ B) & Ai) ^ <(X ^ A) & (Y ^ B)>
  auto z = ring_xor(ring_xor(ring_and(x_a, b), ring_and(y_b, a)), c);
  if (conn_->Rank() == 0) {
    ring_xor_(z, ring_and(x_a, y_b));
  }

  return z.as(lhs.eltype());
}

std::array<ArrayRef, 3> BasicOTProtocols::AndTriple(FieldType field,
                                                    size_t numel, bool packed) {
  SPU_ENFORCE(numel > 0);

  std::vector<uint8_t> a(numel);
  std::vector<uint8_t> b(numel);
  std::vector<uint8_t> c(numel);

  std::vector<uint8_t> u(numel);
  std::vector<uint8_t> v(numel);

  constexpr size_t bit_width = 1;
  if (0 == Rank()) {
    ferret_receiver_->RecvRMRC(/*choice*/ absl::MakeSpan(a),
                               /*msg*/ absl::MakeSpan(u), bit_width);
    ferret_sender_->SendRMRC(absl::MakeSpan(v), absl::MakeSpan(b), bit_width);
  } else {
    ferret_sender_->SendRMRC(absl::MakeSpan(v), absl::MakeSpan(b), bit_width);
    ferret_receiver_->RecvRMRC(/*choice*/ absl::MakeSpan(a),
                               /*msg*/ absl::MakeSpan(u), bit_width);
  }
  ferret_sender_->Flush();

  for (size_t i = 0; i < numel; i++) {
    b[i] = b[i] ^ v[i];
    c[i] = (a[i] & b[i]) ^ u[i] ^ v[i];
  }

  const size_t n = CeilDiv(numel, packed ? 8UL * SizeOf(field) : 1UL);

  ArrayRef _a = ring_zeros(field, n);
  ArrayRef _b = ring_zeros(field, n);
  ArrayRef _c = ring_zeros(field, n);

  DISPATCH_ALL_FIELDS(field, "", [&]() {
    if (not packed) {
      auto xa = ArrayView<ring2k_t>(_a);
      auto xb = ArrayView<ring2k_t>(_b);
      auto xc = ArrayView<ring2k_t>(_c);

      SPU_ENFORCE_EQ(n, numel);
      for (size_t i = 0; i < n; ++i) {
        xa[i] = static_cast<ring2k_t>(a[i]);
        xb[i] = static_cast<ring2k_t>(b[i]);
        xc[i] = static_cast<ring2k_t>(c[i]);
      }
    } else {
      auto xa = ArrayView<uint8_t>(_a);
      auto xb = ArrayView<uint8_t>(_b);
      auto xc = ArrayView<uint8_t>(_c);
      for (size_t i = 0; i < numel; i += 8) {
        size_t sze = std::min(i + 8, numel) - i;
        xa[i >> 3] = BoolToU8(a.data() + i, sze);
        xb[i >> 3] = BoolToU8(b.data() + i, sze);
        xc[i >> 3] = BoolToU8(c.data() + i, sze);
      }
    }
  });

  return {_a, _b, _c};
}

int BasicOTProtocols::Rank() const { return ferret_sender_->Rank(); }

}  // namespace spu::mpc::cheetah
