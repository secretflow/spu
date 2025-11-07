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

#include "libspu/mpc/cheetah/nonlinear/mix_mul_prot.h"

#include <future>

#include "libspu/mpc/cheetah/nonlinear/compare_prot.h"
#include "libspu/mpc/cheetah/ot/matrix_transpose.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/utils/ring_ops.h"
namespace spu::mpc::cheetah {

namespace {
template <typename T>
typename std::make_unsigned<T>::type makeMask(size_t bw) {
  using U = typename std::make_unsigned<T>::type;
  if (bw == sizeof(U) * 8) {
    return static_cast<U>(-1);
  }
  return (static_cast<U>(1) << bw) - 1;
}

NdArrayRef CastTo(const NdArrayRef& x, FieldType from, FieldType to) {
  SPU_ENFORCE_EQ(x.eltype().as<RingTy>()->field(), from);

  if (from == to) {
    return x;
  }

  // cast from src type to dst type
  auto out = ring_zeros(to, x.shape());
  DISPATCH_ALL_FIELDS(from, [&]() {
    using sT = ring2k_t;
    NdArrayView<sT> src(x);

    DISPATCH_ALL_FIELDS(to, [&]() {
      using dT = ring2k_t;
      NdArrayView<dT> dst(out);
      pforeach(0, src.numel(),
               [&](int64_t i) { dst[i] = static_cast<dT>(src[i]); });
    });
  });
  return out;
}
}  // namespace

// w = msbA | msbB
NdArrayRef MixMulProtocol::MSB0ToWrap(const NdArrayRef& inp,
                                      const WrapMeta& meta) {
  const auto field = inp.eltype().as<Ring2k>()->field();

  const int64_t numel = inp.numel();
  const int rank = basic_ot_prot_->Rank();

  constexpr size_t N = 2;  // 1-of-2 OT
  constexpr size_t nbits = 1;

  NdArrayRef wrap_bool;
  if (0 == rank) {
    wrap_bool = ring_randbit(meta.src_ring, inp.shape());
    std::vector<uint8_t> send(numel * N);

    DISPATCH_ALL_FIELDS(field, [&]() {
      using u2k = std::make_unsigned<ring2k_t>::type;
      NdArrayView<const u2k> xinp(inp);
      NdArrayView<const u2k> xrnd(wrap_bool);
      // when msb(xA) = 0, set (r, 1^r)
      //  ow. msb(xA) = 1, set (1^r, 1^r)
      // Equals to (r^msb(xA), r^1)
      for (int64_t i = 0; i < numel; ++i) {
        send[2 * i + 0] = xrnd[i] ^ ((xinp[i] >> (meta.src_width - 1)) & 1);
        send[2 * i + 1] = xrnd[i] ^ 1;
      }
    });

    auto sender = basic_ot_prot_->GetSenderCOT();
    sender->SendCMCC(absl::MakeSpan(send), N, nbits);
    sender->Flush();
  } else {
    std::vector<uint8_t> choices(numel, 0);
    DISPATCH_ALL_FIELDS(meta.src_ring, [&]() {
      using sT = std::make_unsigned<ring2k_t>::type;
      NdArrayView<const sT> xinp(inp);
      for (int64_t i = 0; i < numel; ++i) {
        choices[i] = (xinp[i] >> (meta.src_width - 1)) & 1;
      }

      std::vector<uint8_t> recv(numel);
      basic_ot_prot_->GetReceiverCOT()->RecvCMCC(absl::MakeSpan(choices), N,
                                                 absl::MakeSpan(recv), nbits);

      wrap_bool = ring_zeros(meta.src_ring, inp.shape());
      NdArrayView<ring2k_t> xoup(wrap_bool);
      pforeach(0, numel, [&](int64_t i) {
        xoup[i] = static_cast<ring2k_t>(recv[i] & 1);
      });
    });
  }

  // cast from src type to dst type
  NdArrayRef out = CastTo(wrap_bool, meta.src_ring, meta.dst_ring);

  return out.as(makeType<BShrTy>(meta.dst_ring, 1));
}

// w = msbA & msbB
// w = ~((~msbA) | (~msbB))
NdArrayRef MixMulProtocol::MSB1ToWrap(const NdArrayRef& inp,
                                      const WrapMeta& meta) {
  const auto field = inp.eltype().as<Ring2k>()->field();

  const int64_t numel = inp.numel();
  const int rank = basic_ot_prot_->Rank();

  constexpr size_t N = 2;  // 1-of-2 OT
  constexpr size_t nbits = 1;

  NdArrayRef wrap_bool;
  if (0 == rank) {
    wrap_bool = ring_randbit(meta.src_ring, inp.shape());
    std::vector<uint8_t> send(numel * N);

    DISPATCH_ALL_FIELDS(field, [&]() {
      using u2k = std::make_unsigned<ring2k_t>::type;
      NdArrayView<const u2k> xinp(inp);
      NdArrayView<const u2k> xrnd(wrap_bool);
      // when ~msb(xA) = 0, set (r, 1^r)
      //  ow. ~msb(xA) = 1, set (1^r, 1^r)
      // Equals to (r ^ ~msb(xA), r^1)
      for (int64_t i = 0; i < numel; ++i) {
        send[2 * i + 0] =
            xrnd[i] ^ (((xinp[i] >> (meta.src_width - 1)) & 1) ^ 1);
        send[2 * i + 1] = xrnd[i] ^ 1;
      }
    });

    auto sender = basic_ot_prot_->GetSenderCOT();
    sender->SendCMCC(absl::MakeSpan(send), N, nbits);
    sender->Flush();
  } else {
    std::vector<uint8_t> choices(numel, 0);
    DISPATCH_ALL_FIELDS(meta.src_ring, [&]() {
      using sT = std::make_unsigned<ring2k_t>::type;
      NdArrayView<const sT> xinp(inp);
      for (int64_t i = 0; i < numel; ++i) {
        choices[i] = ((xinp[i] >> (meta.src_width - 1)) & 1) ^ 1;
      }

      std::vector<uint8_t> recv(numel);
      basic_ot_prot_->GetReceiverCOT()->RecvCMCC(absl::MakeSpan(choices), N,
                                                 absl::MakeSpan(recv), nbits);

      // take negate of recv to get w = ~((~msbA) | (~msbB))
      wrap_bool = ring_zeros(meta.src_ring, inp.shape());
      NdArrayView<ring2k_t> xoup(wrap_bool);
      pforeach(0, numel, [&](int64_t i) {
        xoup[i] = static_cast<ring2k_t>((recv[i] & 1) ^ 1);
      });
    });
  }

  // cast from src type to dst type
  NdArrayRef out = CastTo(wrap_bool, meta.src_ring, meta.dst_ring);

  return out.as(makeType<BShrTy>(meta.dst_ring, 1));
}

NdArrayRef MixMulProtocol::ComputeWrap(const NdArrayRef& inp,
                                       const WrapMeta& meta) {
  const auto field = inp.eltype().as<Ring2k>()->field();
  SPU_ENFORCE(field == meta.src_ring);

  // NdArrayRef input = inp.clone();
  // if (SizeOf(field) * 8 != static_cast<size_t>(meta.src_width)) {
  //   ring_reduce_(input, meta.src_width);
  // }

  switch (meta.sign) {
    case SignType::Positive:
      return MSB0ToWrap(inp, meta);
      break;
    case SignType::Negative:
      return MSB1ToWrap(inp, meta);
      break;
    default:
      break;
  }

  CompareProtocol cmp_protocol(basic_ot_prot_);
  NdArrayRef wrap_bool;

  if (basic_ot_prot_->Rank() == 0) {
    wrap_bool = cmp_protocol.Compute(inp, /*gt*/ true, meta.src_width);
  } else {
    //     w = 1{x0 + x1 >= 2^k}
    // <=> w = 1{x0 > 2^k - 1 - x1 mod 2^k}
    // <=> w = 1{x0 > -x1 - 1 mod 2^k}
    auto adjusted = ring_neg(inp);  // -x1 mod 2^k
    DISPATCH_ALL_FIELDS(meta.src_ring, [&]() {
      NdArrayView<ring2k_t> xadj(adjusted);
      const auto mask = makeMask<ring2k_t>(meta.src_width);
      // -1 mod 2^k
      pforeach(0, xadj.numel(), [&](int64_t i) {
        xadj[i] -= 1;
        xadj[i] &= mask;
      });
    });

    wrap_bool = cmp_protocol.Compute(adjusted, /*gt*/ true, meta.src_width);
  }

  auto wrap = CastTo(wrap_bool, meta.src_ring, meta.dst_ring);

  return wrap.as(makeType<BShrTy>(meta.dst_ring, 1));
}

// unsigned cross mul
// P0 has x, P1 has y, compute out = a0 + a1 = x * y
// x, y, out can be in different fields
NdArrayRef MixMulProtocol::CrossMul(const NdArrayRef& inp, const Meta& meta) {
  // Let N be the number of data points
  // For x \in Z_M, y \in Z_N, where M = 2^m, N = 2^n, m<=n
  // We should run (N * m) COTs
  const auto N = inp.numel();
  int receiver_rank = (meta.bw_x <= meta.bw_y) ? 0 : 1;
  int64_t m;
  int64_t n;
  // make sure m<=n
  if (meta.bw_x <= meta.bw_y) {
    m = meta.bw_x;
    n = meta.bw_y;
  } else {
    m = meta.bw_y;
    n = meta.bw_x;
  }

  const auto out_bw = meta.bw_out;
  // nums of COT per data
  const auto level = m;
  const int rank = basic_ot_prot_->Rank();
  const auto field = inp.eltype().as<Ring2k>()->field();

  const auto num_cots = N * level;

  // REF: SIRNN Algorithm 4
  // Run COT to get <t_i>
  NdArrayRef oup = ring_zeros(meta.field_out, {static_cast<int64_t>(num_cots)});
  if (rank == receiver_rank) {
    //
    DISPATCH_ALL_FIELDS(field, [&]() {
      using u2k = std::make_unsigned<ring2k_t>::type;
      const auto msk = makeMask<ring2k_t>(m);
      NdArrayView<u2k> inp_(inp);

      // collapse cots
      // Sample-major order
      //   N   ||     N     ||      N    || .... ||    N
      // k=bw  ||   k=bw-1  ||  k=bw - 2 || ....
      std::vector<uint8_t> choices(num_cots);
      for (int64_t i = 0; i < level; ++i) {
        const auto offset = i * N;
        pforeach(0, N, [&](int64_t idx) {
          const auto x = inp_[idx] & msk;
          choices[offset + idx] = (x >> i) & 1;
        });
      }

      DISPATCH_ALL_FIELDS(meta.field_out, [&]() {
        using out_u2k = std::make_unsigned<ring2k_t>::type;
        auto out_span = absl::MakeSpan(&oup.at<out_u2k>(0), num_cots);

        basic_ot_prot_->GetReceiverCOT()->RecvCAMCC_Collapse(
            absl::MakeSpan(choices), out_span, out_bw, level);
      });
    });
  } else {
    // sender
    DISPATCH_ALL_FIELDS(field, [&]() {
      using u2k = std::make_unsigned<ring2k_t>::type;
      const auto msk = makeMask<ring2k_t>(n);
      NdArrayView<u2k> inp_(inp);

      DISPATCH_ALL_FIELDS(meta.field_out, [&]() {
        using out_u2k = std::make_unsigned<ring2k_t>::type;

        // collapse cots
        // Sample-major order
        //   N   ||     N     ||      N    || .... ||    N
        // k=bw  ||   k=bw-1  ||  k=bw - 2 || ....
        std::vector<out_u2k> corr_data(num_cots);
        for (int64_t i = 0; i < level; ++i) {
          const auto offset = i * N;
          pforeach(0, N, [&](int64_t idx) {
            const auto y = static_cast<out_u2k>(inp_[idx] & msk);
            corr_data[offset + idx] = y;
          });
        }

        auto out_span = absl::MakeSpan(&oup.at<out_u2k>(0), num_cots);
        basic_ot_prot_->GetSenderCOT()->SendCAMCC_Collapse(
            absl::MakeSpan(corr_data), out_span, out_bw, level);
        basic_ot_prot_->GetSenderCOT()->Flush();
      });
    });
  }

  // local computation
  // <z>_l = \sum_i=0^{m-1} 2^i <t_i>
  NdArrayRef ret = ring_zeros(meta.field_out, inp.shape());
  DISPATCH_ALL_FIELDS(meta.field_out, [&]() {
    using u2k = std::make_unsigned<ring2k_t>::type;
    const auto msk = makeMask<ring2k_t>(out_bw);

    NdArrayView<u2k> ret_(ret);
    NdArrayView<u2k> oup_(oup);

    // cot get (x, x + corr*bit), so sender need to negate
    if (rank != receiver_rank) {
      pforeach(0, oup.numel(), [&](int64_t idx) {  //
        oup_[idx] = -oup_[idx];
      });
    }

    // naive
    for (int64_t i = 0; i < level; ++i) {
      const auto offset = i * N;
      const auto cur_bw = meta.bw_out - i;
      const auto cur_mask = makeMask<ring2k_t>(cur_bw);

      pforeach(0, N, [&](int64_t idx) {  //
        ret_[idx] += ((oup_[offset + idx] & cur_mask) << i);
        ret_[idx] &= msk;
      });
    }

    // transpose first for cache friendly (some bugs for uint8 and uint16..)
    // std::vector<u2k> transposed(num_cots);
    // sse_transpose(oup.data<u2k>(), transposed.data(), level, N);
    // pforeach(0, N, [&](int64_t idx) {
    //   const auto offset = idx * level;
    //   for (int64_t i = 0; i < level; ++i) {
    //     const auto cur_bw = meta.bw_out - i;
    //     const auto cur_mask = makeMask<ring2k_t>(cur_bw);
    //     ret_[idx] += ((transposed[offset + i] & cur_mask) << i);
    //   }
    //   ret_[idx] &= msk;
    // });
    //
  });

  ret.set_fxp_bits(meta.bw_out);
  return ret;
}

// unsigned mix mul, the x and y are unsigned
// the bits of x, y, out can be different
// x \in Z_M, y \in Z_N, out \in Z_{L}
// where M = 2^l, N = 2^n, L = 2^l
// return: x*y, wrap_x, wrap_y
std::tuple<NdArrayRef, NdArrayRef, NdArrayRef> MixMulProtocol::UnsignedMixMul(
    const NdArrayRef& x, const NdArrayRef& y, const Meta& meta) {
  // only deal with unsigned mix mul
  SPU_ENFORCE(meta.signed_arith == false);

  // SIRNN Algorithm 3
  const auto rank = basic_ot_prot_->Rank();
  const auto m = meta.bw_x;
  const auto n = meta.bw_y;
  const auto l = meta.bw_out;

  // 1. Compute `l` bits cross mul Ashr
  NdArrayRef x0y1;
  NdArrayRef x1y0;
  std::future<NdArrayRef> task;

  // TODO: use async may block?
  if (rank == 0) {
    auto _swap_meta = meta;
    _swap_meta.bw_x = meta.bw_y;
    _swap_meta.bw_y = meta.bw_x;
    _swap_meta.field_x = meta.field_y;
    _swap_meta.field_y = meta.field_x;

    // task = std::async(
    //     std::launch::async,
    //     [&](const NdArrayRef& inp, const Meta& meta) {
    //       return CrossMul(inp, meta);
    //     },
    //     y, _swap_meta);

    x0y1 = CrossMul(x, meta);
    x1y0 = CrossMul(y, _swap_meta);
  } else {
    auto _swap_meta = meta;
    _swap_meta.bw_x = meta.bw_y;
    _swap_meta.bw_y = meta.bw_x;
    _swap_meta.field_x = meta.field_y;
    _swap_meta.field_y = meta.field_x;

    // task = std::async(
    //     std::launch::async,
    //     [&](const NdArrayRef& inp, const Meta& meta) {
    //       return CrossMul(inp, meta);
    //     },
    //     x, _swap_meta);

    x0y1 = CrossMul(y, meta);
    x1y0 = CrossMul(x, _swap_meta);
  }
  // x1y0 = task.get();

  // 2. Compute wrap according to bit-width
  //   a. Compute wrap (BshrTy)
  //   b. Use mux to get Ashare of wrap * x/y (only m / n bits are enough)

  // wrap is single bit Bshr, so FM8 is enough
  NdArrayRef wrap_x = ring_zeros(FM8, x.shape());
  // w_x * y \in Z_{N}
  NdArrayRef wrapx_yA = ring_zeros(meta.field_y, x.shape());
  // wrap is single bit Bshr, so FM8 is enough
  NdArrayRef wrap_y = ring_zeros(FM8, x.shape());
  // w_y * x \in Z_{M}
  NdArrayRef wrapy_xA = ring_zeros(meta.field_x, x.shape());

  // should compute wrap i.f.f l > m or n
  if (l > m) {
    WrapMeta _wrap_meta;
    _wrap_meta.sign = meta.sign_x;
    _wrap_meta.src_ring = meta.field_x;
    _wrap_meta.src_width = meta.bw_x;
    wrap_x = ComputeWrap(x, _wrap_meta);

    BasicOTProtocols::MultiplexMeta _mux_meta;
    _mux_meta.src_ring = meta.field_y;
    _mux_meta.src_width = meta.bw_y;
    _mux_meta.dst_ring = meta.field_y;
    _mux_meta.dst_width = meta.bw_y;
    wrapx_yA = basic_ot_prot_->Multiplexer(y, wrap_x, _mux_meta);
  }

  if (l > n) {
    WrapMeta _wrap_meta;
    _wrap_meta.sign = meta.sign_y;
    _wrap_meta.src_ring = meta.field_y;
    _wrap_meta.src_width = meta.bw_y;
    wrap_y = ComputeWrap(y, _wrap_meta);

    BasicOTProtocols::MultiplexMeta _mux_meta;
    _mux_meta.src_ring = meta.field_x;
    _mux_meta.src_width = meta.bw_x;
    _mux_meta.dst_ring = meta.field_x;
    _mux_meta.dst_width = meta.bw_x;
    wrapy_xA = basic_ot_prot_->Multiplexer(x, wrap_y, _mux_meta);
  }

  // 3. Doing final local computation
  NdArrayRef ret(makeType<RingTy>(meta.field_out), x.shape());

  // NdArrayView<uint8_t> _wrapx(wrap_x);
  // NdArrayView<uint8_t> _wrapy(wrap_y);
  DISPATCH_ALL_FIELDS(meta.field_out, [&]() {
    using out_el_t = std::make_unsigned_t<ring2k_t>;
    const auto out_msk = makeMask<out_el_t>(l);

    NdArrayView<out_el_t> _ret(ret);
    NdArrayView<out_el_t> _x0y1(x0y1);
    NdArrayView<out_el_t> _x1y0(x1y0);

    DISPATCH_ALL_FIELDS(meta.field_x, [&]() {
      using x_el_t = std::make_unsigned_t<ring2k_t>;
      const auto x_msk = makeMask<x_el_t>(m);

      NdArrayView<x_el_t> _x(x);
      NdArrayView<x_el_t> _wrapy_xA(wrapy_xA);

      DISPATCH_ALL_FIELDS(meta.field_y, [&]() {
        using y_el_t = std::make_unsigned_t<ring2k_t>;
        const auto y_msk = makeMask<y_el_t>(n);

        NdArrayView<y_el_t> _y(y);
        NdArrayView<y_el_t> _wrapx_yA(wrapx_yA);

        pforeach(0, ret.numel(), [&](int64_t idx) {
          // x_i * y_i
          out_el_t local_term = static_cast<out_el_t>(_x[idx] & x_msk) *
                                static_cast<out_el_t>(_y[idx] & y_msk);
          // cross term
          out_el_t cross_term = _x0y1[idx] + _x1y0[idx];
          _ret[idx] = local_term + cross_term -
                      (static_cast<out_el_t>(_wrapx_yA[idx]) << m) -
                      (static_cast<out_el_t>(_wrapy_xA[idx]) << n);
          _ret[idx] &= out_msk;
        });
      });
    });
  });

  ret.set_fxp_bits(l);
  return std::make_tuple(ret, wrap_x, wrap_y);
}

namespace {
// x + 2^{bw-1}
NdArrayRef FlipMsb(const NdArrayRef& x, size_t bw, size_t rank) {
  if (rank == 1) {
    auto ret = x;
    ret.set_fxp_bits(bw);
    return ret;
  }

  const auto field = x.eltype().as<Ring2k>()->field();

  auto ret = ring_zeros(field, x.shape());

  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _x(x);
    NdArrayView<ring2k_t> _ret(ret);
    const auto power = static_cast<ring2k_t>(1) << (bw - 1);
    const auto msk = makeMask<ring2k_t>(bw);

    pforeach(0, x.numel(), [&](int64_t idx) {  //
      _ret[idx] = (_x[idx] + power) & msk;
    });
  });

  ret.set_fxp_bits(bw);

  return ret;
}
}  // namespace

// general mix mul
NdArrayRef MixMulProtocol::Compute(const NdArrayRef& x, const NdArrayRef& y,
                                   const Meta& meta) {
  const auto rank = basic_ot_prot_->Rank();
  const auto m = meta.bw_x;
  const auto n = meta.bw_y;
  const auto l = meta.bw_out;

  // sanity check part
  {
    if (meta.use_heuristic) {
      // not think about the recovery trick of heuristic
      SPU_THROW("Not implemented yet...");
      SPU_ENFORCE(meta.signed_arith,
                  "use_heuristic=true need signed arith=true");
    }

    // if uniform ring, should use 2pc mul
    if ((l == m) && (l == n)) {
      SPU_THROW("uniform ring should use mul_aa kernel.");
    }

    SPU_ENFORCE((m <= l) && (n <= l));
    SPU_ENFORCE(l <= m + n);

    SPU_ENFORCE((meta.field_x <= meta.field_out) &&
                (meta.field_y <= meta.field_out));

    SPU_ENFORCE(m <= (int64_t)SizeOf(meta.field_x) * 8);
    SPU_ENFORCE(n <= (int64_t)SizeOf(meta.field_y) * 8);
    SPU_ENFORCE(l <= (int64_t)SizeOf(meta.field_out) * 8);

    SPU_ENFORCE(x.eltype().as<Ring2k>()->field() == meta.field_x);
    SPU_ENFORCE(y.eltype().as<Ring2k>()->field() == meta.field_y);

    SPU_ENFORCE_EQ(x.fxp_bits(), m);
    SPU_ENFORCE_EQ(y.fxp_bits(), n);
  }

  // TODO: how to fix the heuristic?
  // if (meta.signed_arith && meta.use_heuristic) {
  //   // Use heuristic via adding a large positive value to make sure the input
  //   // is also positive.

  // }

  NdArrayRef wrap_x;
  NdArrayRef wrap_y;
  NdArrayRef unsigned_mul;
  if (!meta.signed_arith) {
    std::tie(unsigned_mul, wrap_x, wrap_y) = UnsignedMixMul(x, y, meta);

    return unsigned_mul;
  }

  // Some maths for signed arithmetic:
  // x \in Z_{M}, y \in Z_{N}, out \in Z_{L}, where max{m,n} <= l <= m+n
  // x^' = x + 2^{m-1}, y^' = y + 2^{n-1}
  // x * y = (x^' - 2^{m-1}) * (y^' - 2^{n-1})
  //       = (x^' * y^')
  //         - 2^{m-1}(y_0^' + y_1^' - 2^n w_{y^'})
  //         - 2^{n-1}(x_0^' + x_1^' - 2^m w_{x^'})
  //         + 2^{m+n-2}
  // Notes:
  // a. (x^' * y^') can be computed by unsigned mix mul
  // b. 2^{m+n-1} * (w_{x^'} + w_{y^'}) can be computed by re-using the wrap
  //    - i.e. 2^{m+n-1} * w = 2^{m+n-1} * (w0 + w1 - 2*w0*w1) mod L
  //                         = 2^{m+n-1} * (w0 + w1) mod L
  // c. The rest items can be computed locally
  Meta _meta = meta;
  // flip sign when doing signed arithmetic
  if (meta.sign_x != SignType::Unknown) {
    _meta.sign_x = (meta.sign_x == SignType::Positive ? SignType::Negative
                                                      : SignType::Positive);
  }
  if (meta.sign_y != SignType::Unknown) {
    _meta.sign_y = (meta.sign_y == SignType::Positive ? SignType::Negative
                                                      : SignType::Positive);
  }
  _meta.signed_arith = false;

  auto x_prime = FlipMsb(x, m, rank);
  auto y_prime = FlipMsb(y, n, rank);

  std::tie(unsigned_mul, wrap_x, wrap_y) =
      UnsignedMixMul(x_prime, y_prime, _meta);

  // ring_print(x_prime, std::to_string(rank) + ":x_prime");
  // ring_print(y_prime, std::to_string(rank) + ":y_prime");
  // ring_print(unsigned_mul, std::to_string(rank) + ":unsigned_mul");
  // ring_print(wrap_x, std::to_string(rank) + ":wrap_x");
  // ring_print(wrap_y, std::to_string(rank) + ":wrap_y");

  // wrap is always FM8
  NdArrayView<uint8_t> _wrapx(wrap_x);
  NdArrayView<uint8_t> _wrapy(wrap_y);

  DISPATCH_ALL_FIELDS(meta.field_out, [&]() {
    using out_el_t = ring2k_t;
    const auto out_msk = makeMask<out_el_t>(l);
    // we re-use this buffer
    NdArrayView<out_el_t> _unsigned_mul(unsigned_mul);

    DISPATCH_ALL_FIELDS(meta.field_x, [&]() {
      using x_el_t = ring2k_t;
      NdArrayView<x_el_t> _x_prime(x_prime);
      // 2^{m-1}
      const auto pow_x = static_cast<out_el_t>(1) << (m - 1);

      DISPATCH_ALL_FIELDS(meta.field_y, [&]() {
        using y_el_t = ring2k_t;
        NdArrayView<y_el_t> _y_prime(y_prime);
        // 2^{n-1}
        const auto pow_y = static_cast<out_el_t>(1) << (n - 1);

        // SPDLOG_INFO("mask {:0b}, pow: {:0b}, {:0b}, {:0b},", out_msk, pow_x,
        //             pow_y, pow_x << (n - 1));

        if (rank == 0) {
          pforeach(0, x.numel(), [&](int64_t idx) {
            _unsigned_mul[idx] +=
                (-pow_x *
                     (static_cast<out_el_t>(_y_prime[idx]) -
                      (pow_y << 1) * static_cast<out_el_t>(_wrapy[idx] & 1)) -
                 pow_y *
                     (static_cast<out_el_t>(_x_prime[idx]) -
                      (pow_x << 1) * static_cast<out_el_t>((_wrapx[idx] & 1))));
            // add constant 2^{m+n-2}
            _unsigned_mul[idx] += (pow_x << (n - 1));
            _unsigned_mul[idx] &= out_msk;
          });
        } else {
          pforeach(0, x.numel(), [&](int64_t idx) {
            _unsigned_mul[idx] +=
                (-pow_x *
                     (static_cast<out_el_t>(_y_prime[idx]) -
                      (pow_y << 1) * static_cast<out_el_t>((_wrapy[idx] & 1))) -
                 pow_y *
                     (static_cast<out_el_t>(_x_prime[idx]) -
                      (pow_x << 1) * static_cast<out_el_t>((_wrapx[idx] & 1))));
            _unsigned_mul[idx] &= out_msk;
          });
        }
      });
    });
  });

  unsigned_mul.set_fxp_bits(l);
  return unsigned_mul;
}
}  // namespace spu::mpc::cheetah