// Copyright 2024 Ant Group Co., Ltd.
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

#include "libspu/mpc/cheetah/nonlinear/prime_ring_ext_prot.h"

#include <memory>

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/parallel_utils.h"
#include "libspu/core/prelude.h"
#include "libspu/core/type.h"
#include "libspu/core/type_util.h"
#include "libspu/mpc/cheetah/nonlinear/compare_prot.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {

namespace {
template <typename T>
auto makeMask(size_t bw) {
  using U = typename std::make_unsigned<T>::type;
  if (bw == sizeof(U) * 8) {
    return static_cast<U>(-1);
  }
  return (static_cast<U>(1) << bw) - 1;
}

NdArrayRef CastTo(const NdArrayRef &x, FieldType from, FieldType to) {
  SPU_ENFORCE_EQ(x.eltype().as<RingTy>()->field(), from);
  // cast from src type to dst type
  auto out = ring_zeros(to, x.shape());
  DISPATCH_ALL_FIELDS(from, "cast_to", [&]() {
    using U0 = ring2k_t;
    NdArrayView<U0> src(x);

    DISPATCH_ALL_FIELDS(to, "cast_to", [&]() {
      using U1 = ring2k_t;
      NdArrayView<U1> dst(out);
      pforeach(0, src.numel(),
               [&](int64_t i) { dst[i] = static_cast<U1>(src[i]); });
    });
  });
  return out;
}

}  // namespace

PrimeRingExtendProtocol::PrimeRingExtendProtocol(
    const std::shared_ptr<BasicOTProtocols> &base)
    : basic_ot_prot_(base) {}

// We assume the input is ``positive''
// Given h0 + h1 = h mod p and h < p / 2
// Define b0 = 1{h0 >= p/2}
//        b1 = 1{h1 >= p/2}
// Compute w = 1{h0 + h1 >= p}
NdArrayRef PrimeRingExtendProtocol::MSB0ToWrap(const NdArrayRef &inp,
                                               const Meta &meta) {
  const auto src_ring = inp.eltype().as<Ring2k>()->field();
  const int64_t numel = inp.numel();
  const int rank = basic_ot_prot_->Rank();

  constexpr size_t N = 2;  // 1-of-2 OT
  constexpr size_t nbits = 1;

  NdArrayRef wrap_bool;
  if (0 == rank) {
    wrap_bool = ring_randbit(src_ring, inp.shape());
    std::vector<uint8_t> send(numel * N);

    DISPATCH_ALL_FIELDS(src_ring, "MSB0_adjust", [&]() {
      using U0 = std::make_unsigned<ring2k_t>::type;
      NdArrayView<const U0> xinp(inp);
      NdArrayView<const U0> xrnd(wrap_bool);
      U0 phalf = static_cast<U0>(meta.prime) >> 1;
      for (int64_t i = 0; i < numel; ++i) {
        send[2 * i + 0] = xrnd[i] ^ (static_cast<uint8_t>(xinp[i] >= phalf));
        send[2 * i + 1] = xrnd[i] ^ 1;
      }
    });

    auto sender = basic_ot_prot_->GetSenderCOT();
    sender->SendCMCC(absl::MakeSpan(send), N, nbits);
    sender->Flush();
  } else {
    std::vector<uint8_t> choices(numel, 0);
    DISPATCH_ALL_FIELDS(src_ring, "MSB0_adjust", [&]() {
      using U0 = std::make_unsigned<ring2k_t>::type;
      U0 phalf = static_cast<U0>(meta.prime) >> 1;
      NdArrayView<const U0> xinp(inp);
      for (int64_t i = 0; i < numel; ++i) {
        choices[i] = static_cast<uint8_t>(xinp[i] >= phalf);
      }

      std::vector<uint8_t> recv(numel);
      basic_ot_prot_->GetReceiverCOT()->RecvCMCC(absl::MakeSpan(choices), N,
                                                 absl::MakeSpan(recv), nbits);

      wrap_bool = ring_zeros(src_ring, inp.shape());
      NdArrayView<ring2k_t> xoup(wrap_bool);
      pforeach(0, numel, [&](int64_t i) {
        xoup[i] = static_cast<ring2k_t>(recv[i] & 1);
      });
    });
  }

  // cast from src type to dst type
  NdArrayRef out = CastTo(wrap_bool, src_ring, meta.dst_ring);

  return basic_ot_prot_->B2ASingleBitWithSize(
      out.as(makeType<BShrTy>(meta.dst_ring, 1)), meta.dst_width);
}

NdArrayRef PrimeRingExtendProtocol::Compute(const NdArrayRef &inp,
                                            const Meta &meta) {
  const auto src_ring = inp.eltype().as<Ring2k>()->field();
  size_t prime_width = absl::bit_width(meta.prime);
  SPU_ENFORCE(SizeOf(src_ring) * 8 >= prime_width);
  SPU_ENFORCE(meta.dst_width > (int64_t)prime_width);
  auto truncate_nbits = meta.truncate_nbits ? *meta.truncate_nbits : 0;
  SPU_ENFORCE(truncate_nbits >= 0, "invalid truncate_nbits={}", truncate_nbits);

  DISPATCH_ALL_FIELDS(src_ring, "check_range", [&]() {
    NdArrayView<const ring2k_t> input(inp);
    ring2k_t prime = meta.prime;
    for (int64_t i = 0; i < input.numel(); ++i) {
      SPU_ENFORCE(input[i] < prime, "prime share out-of-range");
    }
  });

  const int rank = basic_ot_prot_->Rank();
  const int shft = truncate_nbits;
  NdArrayRef outp = ring_zeros(meta.dst_ring, inp.shape());
  if (rank == 0) {
    auto wrap_arith = MSB0ToWrap(inp, meta);
    DISPATCH_ALL_FIELDS(src_ring, "finalize", [&]() {
      using U0 = ring2k_t;
      U0 prime = static_cast<U0>(meta.prime);
      NdArrayView<const U0> input(inp);
      DISPATCH_ALL_FIELDS(meta.dst_ring, "finalize", [&]() {
        using U1 = ring2k_t;
        auto msk = makeMask<U1>(meta.dst_width);
        NdArrayView<U1> output(outp);
        NdArrayView<const U1> wrap(wrap_arith);
        pforeach(0, inp.numel(), [&](int64_t i) {
          output[i] = static_cast<U1>(input[i] >> shft) -
                      static_cast<U1>(prime >> shft) * wrap[i];
          output[i] &= msk;
        });
      });
    });

    return outp.as(makeType<AShrTy>(meta.dst_ring));
  }

  /// rank = 1
  auto adjusted = inp.clone();
  DISPATCH_ALL_FIELDS(src_ring, "wrap.adj", [&]() {
    using U0 = ring2k_t;
    U0 prime = static_cast<U0>(meta.prime);
    NdArrayView<U0> xadj(adjusted);
    U0 big_val = static_cast<U0>(1) << (prime_width - kHeuristicBound);
    // add a big value (then modulo prime) so that the prime share
    // shoud be positive now.
    pforeach(0, xadj.numel(), [&](int64_t i) {
      xadj[i] = big_val + xadj[i];
      xadj[i] -= (xadj[i] >= prime ? prime : 0);
    });
  });

  // Wrap w = 1{h0 + h1 >= p}
  auto wrap_arith = MSB0ToWrap(adjusted, meta);

  DISPATCH_ALL_FIELDS(src_ring, "finalize", [&]() {
    using U0 = ring2k_t;
    U0 prime = static_cast<U0>(meta.prime);
    NdArrayView<const U0> input(adjusted);
    DISPATCH_ALL_FIELDS(meta.dst_ring, "finalize", [&]() {
      using U1 = ring2k_t;
      const auto msk = makeMask<U1>(meta.dst_width);
      const auto big_val = static_cast<U0>(1)
                           << (prime_width - kHeuristicBound - shft);

      NdArrayView<U1> output(outp);
      NdArrayView<const U1> wrap(wrap_arith);
      pforeach(0, inp.numel(), [&](int64_t i) {
        // Result is (h1/2^d) - (p/2^d) * w1 mod 2^k.
        output[i] = static_cast<U1>(input[i] >> shft) -
                    static_cast<U1>(prime >> shft) * wrap[i];
        // Remove the (shifted) big val from the mod-2^k share.
        output[i] = (output[i] - big_val) & msk;
      });
    });
  });

  return outp.as(makeType<AShrTy>(meta.dst_ring));
}

}  // namespace spu::mpc::cheetah
