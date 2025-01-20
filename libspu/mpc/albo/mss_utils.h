#include "libspu/core/ndarray_ref.h"
#include "libspu/mpc/kernel.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/albo/value.h"

// #define EQ_USE_OFFLINE
// #define EQ_USE_PRG_STATE

#define P0_COUT if (comm->getRank() == 0) std::cout

namespace spu::mpc::albo {

NdArrayRef AssXor2(KernelEvalContext* ctx, const NdArrayRef& lhs,
                    const NdArrayRef& rhs);
NdArrayRef RssXor2(KernelEvalContext* ctx, const NdArrayRef& lhs,
                    const NdArrayRef& rhs);
NdArrayRef MssXor2(KernelEvalContext* ctx, const NdArrayRef& lhs,
                        const NdArrayRef& rhs);

NdArrayRef RssAnd2NoComm(KernelEvalContext* ctx, const NdArrayRef& lhs,
                    const NdArrayRef& rhs);
NdArrayRef MssAnd2NoComm(KernelEvalContext* ctx, const NdArrayRef& lhs,
                    const NdArrayRef& rhs);
NdArrayRef MssAnd3NoComm(KernelEvalContext* ctx, const NdArrayRef& op1,
                    const NdArrayRef& op2, const NdArrayRef& op3);
NdArrayRef MssAnd4NoComm(KernelEvalContext* ctx, const NdArrayRef& op1,
                    const NdArrayRef& op2, const NdArrayRef& op3, const NdArrayRef& op4);

NdArrayRef ResharingAss2Rss(KernelEvalContext* ctx, const NdArrayRef& in);
NdArrayRef ResharingAss2Mss(KernelEvalContext* ctx, const NdArrayRef& in);
NdArrayRef ResharingRss2Mss(KernelEvalContext* ctx, const NdArrayRef& in);
NdArrayRef ResharingMss2Rss(KernelEvalContext* ctx, const NdArrayRef& in);
NdArrayRef ResharingMss2RssAri(KernelEvalContext* ctx, const NdArrayRef& in);
NdArrayRef ResharingRss2Ass(KernelEvalContext* ctx, const NdArrayRef& in);

/**
 * bit deinterleave. Return std::pair(hi, lo).
 * i.e. AaBbCcDd -> ABCD, abcd
 */
template<typename ShareT, size_t num_shares>
std::pair<NdArrayRef, NdArrayRef> bit_split(const NdArrayRef& in) {
  constexpr std::array<uint128_t, 6> kSwapMasks = {{
      yacl::MakeUint128(0x2222222222222222, 0x2222222222222222),  // 4bit
      yacl::MakeUint128(0x0C0C0C0C0C0C0C0C, 0x0C0C0C0C0C0C0C0C),  // 8bit
      yacl::MakeUint128(0x00F000F000F000F0, 0x00F000F000F000F0),  // 16bit
      yacl::MakeUint128(0x0000FF000000FF00, 0x0000FF000000FF00),  // 32bit
      yacl::MakeUint128(0x00000000FFFF0000, 0x00000000FFFF0000),  // 64bit
      yacl::MakeUint128(0x0000000000000000, 0xFFFFFFFF00000000),  // 128bit
  }};
  constexpr std::array<uint128_t, 6> kKeepMasks = {{
      yacl::MakeUint128(0x9999999999999999, 0x9999999999999999),  // 4bit
      yacl::MakeUint128(0xC3C3C3C3C3C3C3C3, 0xC3C3C3C3C3C3C3C3),  // 8bit
      yacl::MakeUint128(0xF00FF00FF00FF00F, 0xF00FF00FF00FF00F),  // 16bit
      yacl::MakeUint128(0xFF0000FFFF0000FF, 0xFF0000FFFF0000FF),  // 32bit
      yacl::MakeUint128(0xFFFF00000000FFFF, 0xFFFF00000000FFFF),  // 64bit
      yacl::MakeUint128(0xFFFFFFFF00000000, 0x00000000FFFFFFFF),  // 128bit
  }};

  const auto* in_ty = in.eltype().as<ShareT>();
  const size_t in_nbits = in_ty->nbits();
  SPU_ENFORCE(in_nbits != 0, "in_nbits={}", in_nbits);
  const size_t out_nbits = in_nbits / 2 + in_nbits % 2;
  const auto out_backtype = calcBShareBacktype(out_nbits);
  const auto out_type = makeType<ShareT>(out_backtype, out_nbits);

  NdArrayRef lo(out_type, in.shape());
  NdArrayRef hi(out_type, in.shape());

  DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), [&]() {
    using in_el_t = ScalarT;
    using in_shr_t = std::array<in_el_t, num_shares>;
    NdArrayView<in_shr_t> _in(in);

    DISPATCH_UINT_PT_TYPES(out_backtype, [&]() {
      using out_el_t = ScalarT;
      using out_shr_t = std::array<out_el_t, num_shares>;

      NdArrayView<out_shr_t> _lo(lo);
      NdArrayView<out_shr_t> _hi(hi);

      pforeach(0, in.numel(), [&](int64_t idx) {
          auto r = _in[idx];
          // algorithm:
          //      0101010101010101
          // swap  ^^  ^^  ^^  ^^
          //      0011001100110011
          // swap   ^^^^    ^^^^
          //      0000111100001111
          // swap     ^^^^^^^^
          //      0000000011111111
          for (int k = 0; k + 1 < Log2Ceil(in_nbits); k++) {
          auto keep = static_cast<in_el_t>(kKeepMasks[k]);
          auto move = static_cast<in_el_t>(kSwapMasks[k]);
          int shift = 1ull << k;

          for (size_t i = 0; i < num_shares; i++) {
              r[i] = (r[i] & keep) ^ ((r[i] >> shift) & move) ^
                  ((r[i] & move) << shift);
          }
          }
          in_el_t mask = (in_el_t(1) << (in_nbits / 2)) - 1;
          for (size_t i = 1; i < num_shares; i++) {
          _lo[idx][i] = static_cast<out_el_t>(r[i]) & mask;
          _hi[idx][i] = static_cast<out_el_t>(r[i] >> (in_nbits / 2)) & mask;
          }
      });
    });
  });

  return std::make_pair(hi, lo);
}

/**
 * bit deinterleave. Return NdArrayRef.
 * i.e. AaBbCcDd -> ABCD, abcd
 */
template<typename ShareT, size_t num_shares>
NdArrayRef bit_split_2(const NdArrayRef& in) {
  constexpr std::array<uint128_t, 6> kSwapMasks = {{
      yacl::MakeUint128(0x2222222222222222, 0x2222222222222222),  // 4bit
      yacl::MakeUint128(0x0C0C0C0C0C0C0C0C, 0x0C0C0C0C0C0C0C0C),  // 8bit
      yacl::MakeUint128(0x00F000F000F000F0, 0x00F000F000F000F0),  // 16bit
      yacl::MakeUint128(0x0000FF000000FF00, 0x0000FF000000FF00),  // 32bit
      yacl::MakeUint128(0x00000000FFFF0000, 0x00000000FFFF0000),  // 64bit
      yacl::MakeUint128(0x0000000000000000, 0xFFFFFFFF00000000),  // 128bit
  }};
  constexpr std::array<uint128_t, 6> kKeepMasks = {{
      yacl::MakeUint128(0x9999999999999999, 0x9999999999999999),  // 4bit
      yacl::MakeUint128(0xC3C3C3C3C3C3C3C3, 0xC3C3C3C3C3C3C3C3),  // 8bit
      yacl::MakeUint128(0xF00FF00FF00FF00F, 0xF00FF00FF00FF00F),  // 16bit
      yacl::MakeUint128(0xFF0000FFFF0000FF, 0xFF0000FFFF0000FF),  // 32bit
      yacl::MakeUint128(0xFFFF00000000FFFF, 0xFFFF00000000FFFF),  // 64bit
      yacl::MakeUint128(0xFFFFFFFF00000000, 0x00000000FFFFFFFF),  // 128bit
  }};

  const auto* in_ty = in.eltype().as<ShareT>();
  const size_t in_nbits = in_ty->nbits();
  SPU_ENFORCE(in_nbits != 0, "in_nbits={}", in_nbits);
  NdArrayRef out(in.eltype(), in.shape());

  DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), [&]() {
    using in_el_t = ScalarT;
    using in_shr_t = std::array<in_el_t, num_shares>;
    NdArrayView<in_shr_t> _in(in);
    NdArrayView<in_shr_t> _out(out);

    pforeach(0, in.numel(), [&](int64_t idx) {
        auto r = _in[idx];
        // algorithm:
        //      0101010101010101
        // swap  ^^  ^^  ^^  ^^
        //      0011001100110011
        // swap   ^^^^    ^^^^
        //      0000111100001111
        // swap     ^^^^^^^^
        //      0000000011111111
        for (int k = 0; k + 1 < Log2Ceil(in_nbits); k++) {
        auto keep = static_cast<in_el_t>(kKeepMasks[k]);
        auto move = static_cast<in_el_t>(kSwapMasks[k]);
        int shift = 1ull << k;

        for (size_t i = 0; i < num_shares; i++) {
            r[i] = (r[i] & keep) ^ ((r[i] >> shift) & move) ^
                ((r[i] & move) << shift);
        }
        }
        for (size_t i = 1; i < num_shares; i++) _out[idx][i] = r[i];
    });
  });

  return out;
}

template<typename ShareT, size_t num_shares>
NdArrayRef bit_interleave(
    const NdArrayRef& in) {
  constexpr std::array<uint128_t, 6> kSwapMasks = {{
      yacl::MakeUint128(0x2222222222222222, 0x2222222222222222),  // 4bit
      yacl::MakeUint128(0x0C0C0C0C0C0C0C0C, 0x0C0C0C0C0C0C0C0C),  // 8bit
      yacl::MakeUint128(0x00F000F000F000F0, 0x00F000F000F000F0),  // 16bit
      yacl::MakeUint128(0x0000FF000000FF00, 0x0000FF000000FF00),  // 32bit
      yacl::MakeUint128(0x00000000FFFF0000, 0x00000000FFFF0000),  // 64bit
      yacl::MakeUint128(0x0000000000000000, 0xFFFFFFFF00000000),  // 128bit
  }};
  constexpr std::array<uint128_t, 6> kKeepMasks = {{
      yacl::MakeUint128(0x9999999999999999, 0x9999999999999999),  // 4bit
      yacl::MakeUint128(0xC3C3C3C3C3C3C3C3, 0xC3C3C3C3C3C3C3C3),  // 8bit
      yacl::MakeUint128(0xF00FF00FF00FF00F, 0xF00FF00FF00FF00F),  // 16bit
      yacl::MakeUint128(0xFF0000FFFF0000FF, 0xFF0000FFFF0000FF),  // 32bit
      yacl::MakeUint128(0xFFFF00000000FFFF, 0xFFFF00000000FFFF),  // 64bit
      yacl::MakeUint128(0xFFFFFFFF00000000, 0x00000000FFFFFFFF),  // 128bit
  }};

  const auto* in_ty = in.eltype().as<ShareT>();
  const size_t in_nbits = in_ty->nbits();
  SPU_ENFORCE(in_nbits != 0, "in_nbits={}", in_nbits);
  const size_t out_nbits = in_nbits;
  const auto out_backtype = calcBShareBacktype(out_nbits);
  const auto out_type = makeType<ShareT>(out_backtype, out_nbits);

  NdArrayRef out(out_type, in.shape());

  DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), [&]() {
    using in_el_t = ScalarT;
    using in_shr_t = std::array<in_el_t, num_shares>;
    NdArrayView<in_shr_t> _in(in);
    NdArrayView<in_shr_t> _out(out);

    pforeach(0, in.numel(), [&](int64_t idx) {
    for (size_t i = 1; i < num_shares; i++) _out[idx][i] = _in[idx][i];

    // algorithm:
    //      0101010101010101
    // swap  ^^  ^^  ^^  ^^
    //      0011001100110011
    // swap   ^^^^    ^^^^
    //      0000111100001111
    // swap     ^^^^^^^^
    //      0000000011111111
    for (int k = Log2Ceil(in_nbits) - 2; k >= 0; k--) {
        auto keep = static_cast<in_el_t>(kKeepMasks[k]);
        auto move = static_cast<in_el_t>(kSwapMasks[k]);
        int shift = 1ull << k;
        for (size_t i = 1; i < num_shares; i++) _out[idx][i] =
            (_out[idx][i] & keep) ^ ((_out[idx][i] >> shift) & move) ^ ((_out[idx][i] & move) << shift);
    }
    });
  });
  return out;
}

template<typename ShareT, size_t num_shares>
NdArrayRef lshift_fixed_bitwidth(const NdArrayRef& in, size_t shift)
{
    const auto* in_ty = in.eltype().as<ShareT>();
    const size_t in_nbits = in_ty->nbits();
    const size_t out_nbits = std::min(in_nbits+shift, SizeOf(in_ty->getBacktype()));
    NdArrayRef out(makeType<ShareT>(in_ty->getBacktype(), out_nbits), in.shape());

    return DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), [&]() {
        using in_el_t = ScalarT;
        using in_shr_t = std::array<in_el_t, num_shares>;
        NdArrayView<in_shr_t> _in(in);
        NdArrayView<in_shr_t> _out(out);

        pforeach(0, in.numel(), [&](int64_t idx) {
            for (size_t i = 0; i < num_shares; i++)
                _out[idx][i] = _in[idx][i] << shift;
        });

        return out;
    });
}

template<typename ShareT, size_t num_shares>
NdArrayRef lshift(const NdArrayRef& in, size_t shift)
{
    const auto* in_ty = in.eltype().as<ShareT>();
    const size_t in_nbits = in_ty->nbits();
    const size_t out_nbits = in_nbits + shift;
    const PtType out_btype = calcBShareBacktype(out_nbits);
    NdArrayRef out(makeType<ShareT>(out_btype, out_nbits), in.shape());

    return DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), [&]() {
        using in_el_t = ScalarT;
        using in_shr_t = std::array<in_el_t, num_shares>;
        NdArrayView<in_shr_t> _in(in);

        return DISPATCH_UINT_PT_TYPES(out_btype, [&]() {
            using out_el_t = ScalarT;
            using out_shr_t = std::array<out_el_t, num_shares>;
            NdArrayView<out_shr_t> _out(out);

            pforeach(0, in.numel(), [&](int64_t idx) {
                for (size_t i = 0; i < num_shares; i++)
                    _out[idx][i] = static_cast<out_el_t>(_in[idx][i]) << shift;
            });

            return out;
        });
    });
}

template<typename ShareT, size_t num_shares>
NdArrayRef rshift(const NdArrayRef& in, size_t shift)
{
    const auto* in_ty = in.eltype().as<ShareT>();
    const size_t in_nbits = in_ty->nbits();
    const size_t out_nbits = in_nbits - shift;
    const PtType out_btype = calcBShareBacktype(out_nbits);
    NdArrayRef out(makeType<ShareT>(out_btype, out_nbits), in.shape());

    return DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), [&]() {
        using in_el_t = ScalarT;
        using in_shr_t = std::array<in_el_t, num_shares>;
        NdArrayView<in_shr_t> _in(in);

        return DISPATCH_UINT_PT_TYPES(out_btype, [&]() {
            using out_el_t = ScalarT;
            using out_shr_t = std::array<out_el_t, num_shares>;
            NdArrayView<out_shr_t> _out(out);

            pforeach(0, in.numel(), [&](int64_t idx) {
                for (size_t i = 0; i < num_shares; i++)
                    _out[idx][i] = _in[idx][i] >> shift;
            });

            return out;
        });
    });
}

// given lo and hi, output hi | lo.
template<typename ShareT, size_t num_shares>
NdArrayRef pack_2_bitvec(const NdArrayRef& lo, const NdArrayRef& hi) {
    assert(lo.shape() == hi.shape());
    const auto* lo_ty = lo.eltype().as<ShareT>();
    const auto* hi_ty = hi.eltype().as<ShareT>();
    const size_t out_nbits = lo_ty->nbits() + hi_ty->nbits();
    const PtType out_btype = calcBShareBacktype(out_nbits);
    NdArrayRef out(makeType<ShareT>(out_btype, out_nbits), lo.shape());

    return DISPATCH_UINT_PT_TYPES(hi_ty->getBacktype(), [&]() {
        using hi_el_t = ScalarT;
        using hi_shr_t = std::array<hi_el_t, num_shares>;
        NdArrayView<hi_shr_t> _hi(hi);

        return DISPATCH_UINT_PT_TYPES(lo_ty->getBacktype(), [&]() {
            using lo_el_t = ScalarT;
            using lo_shr_t = std::array<lo_el_t, num_shares>;
            NdArrayView<lo_shr_t> _lo(lo);

            return DISPATCH_UINT_PT_TYPES(out_btype, [&]() {
                using out_el_t = ScalarT;
                using out_shr_t = std::array<out_el_t, num_shares>;
                NdArrayView<out_shr_t> _out(out);

                pforeach(0, lo.numel(), [&](int64_t idx) {
                const lo_shr_t& l = _lo[idx];
                const hi_shr_t& h = _hi[idx];
                out_shr_t& o = _out[idx];
                for (size_t i=0; i<num_shares; i++) {
                    o[i] = l[i] | (static_cast<out_el_t>(h[i]) << lo_ty->nbits());
                }
                });
                return out;
            });
        });
    });
}

template<typename ShareT, size_t num_shares>
std::pair<NdArrayRef, NdArrayRef> unpack_2_bitvec(const NdArrayRef& in, size_t lo_nbits=0)
{
    const auto* in_ty = in.eltype().as<ShareT>();
    if (lo_nbits==0) lo_nbits = in_ty->nbits() / 2;
    const size_t hi_nbits = in_ty->nbits() - lo_nbits;
    const PtType lo_btype = calcBShareBacktype(lo_nbits);
    const PtType hi_btype = calcBShareBacktype(hi_nbits);
    NdArrayRef lo(makeType<ShareT>(lo_btype, lo_nbits), in.shape());
    NdArrayRef hi(makeType<ShareT>(hi_btype, hi_nbits), in.shape());

    return DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), [&]() {
        using in_el_t = ScalarT;
        using in_shr_t = std::array<in_el_t, num_shares>;
        NdArrayView<in_shr_t> _in(in);

        return DISPATCH_UINT_PT_TYPES(lo_btype, [&]() {
        using lo_el_t = ScalarT;
        using lo_shr_t = std::array<lo_el_t, num_shares>;
        NdArrayView<lo_shr_t> _lo(lo);

        return DISPATCH_UINT_PT_TYPES(hi_btype, [&]() {
            using hi_el_t = ScalarT;
            using hi_shr_t = std::array<hi_el_t, num_shares>;
            NdArrayView<hi_shr_t> _hi(hi);

            pforeach(0, in.numel(), [&](int64_t idx) {
            const in_shr_t& i = _in[idx];
            lo_shr_t& l = _lo[idx];
            hi_shr_t& h = _hi[idx];
            for (size_t j=0; j<num_shares; j++) {
                l[j] = i[j] & ((1 << lo_nbits) - 1);
                h[j] = (i[j] >> lo_nbits) & ((1 << hi_nbits) - 1);
            }
            });
            return std::make_pair(lo, hi);
        });
        });
    });
}

/**
 * Pack the input into a specific container, apply the function, and unpack the result.
 * This is used to compress multiple items with non-standard bitwidth, i.e. 48 bits, into one target_nbits element.
 */
template<typename ShareT, size_t num_shares, typename OutShareT, size_t out_num_shares, typename UnaryOp>
NdArrayRef bitwise_vmap(const NdArrayRef& in, size_t target_nbits, UnaryOp func, size_t in_nbits=0)
{
  const auto* in_ty = in.eltype().as<ShareT>();
  in_nbits = in_nbits == 0? in_ty->nbits(): in_nbits;
  const size_t compress_ratio =  target_nbits / in_nbits;
  const size_t compress_nbits = compress_ratio * in_nbits;
  const PtType compress_btype = calcBShareBacktype(compress_nbits);
  size_t numel = in.numel();
  size_t compress_numel = numel / compress_ratio + ((numel % compress_ratio) > 0);
  Shape compress_shape = {1, static_cast<int64_t>(compress_numel)};
  NdArrayRef compress(makeType<ShareT>(compress_btype, compress_nbits), compress_shape);
  NdArrayRef out(makeType<OutShareT>(in_ty->getBacktype(), in_nbits), in.shape());
  const size_t in_mask = (1 << in_nbits) - 1;

  return DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), [&]() {
      using in_el_t = ScalarT;
      using in_shr_t = std::array<in_el_t, num_shares>;
      using out_shr_t = std::array<in_el_t, out_num_shares>;
      NdArrayView<in_shr_t> _in(in);
      NdArrayView<out_shr_t> _out(out);

      return DISPATCH_UINT_PT_TYPES(compress_btype, [&]() {
          using compress_el_t = ScalarT;
          using compress_shr_t = std::array<compress_el_t, num_shares>;
          using out_compress_shr_t = std::array<compress_el_t, out_num_shares>;
          NdArrayView<compress_shr_t> _compress(compress);

          pforeach(0, compress_numel, [&](int64_t idx) {
            size_t loc = idx * compress_ratio;
            for (size_t s = 0; s < num_shares; s++)
            {
              compress_el_t& _cc = _compress[idx][s];
              _cc = 0;                      // NdArrayRef's buffer is not empty. We should clear it manually.
              compress_el_t _c;
              for (size_t cr = 0; cr < compress_ratio; cr++)
              {
                if (loc + cr < static_cast<size_t>(numel)) _c = _in[loc + cr][s] & in_mask;
                else _c = 0;
                _cc ^= _c << (compress_ratio - 1 - cr);
              }
            }
          });

          auto ret = func(compress);

          NdArrayView<out_compress_shr_t> _ret(ret);
          pforeach(0, compress_numel, [&](int64_t idx) {
            size_t loc = idx * compress_ratio;
            for (size_t s = 0; s < out_num_shares; s++)
            {
              compress_el_t _cc = _ret[idx][s];
              for (size_t cr = 0; cr < compress_ratio; cr++)
              {
                if (loc + cr >= static_cast<size_t>(numel)) break;
                _out[loc + cr][s] = (_cc >> (compress_ratio - 1 - cr)) & in_mask;
              }
            }
          });
          
          return out;
      });
  });
}

/**
 * Serialize the input into a byte array, apply the function, and deserialize the result.
 * This is used to compress single item with non-standard bitwidth, i.e. an element with 48 bits.
 */
template<typename ShareT, size_t num_shares, typename OutShareT, size_t out_num_shares, typename UnaryOp>
NdArrayRef bitwise_vmap_by_byte(const NdArrayRef& in, UnaryOp func, size_t in_nbits=0)
{
  const auto* in_ty = in.eltype().as<ShareT>();
  in_nbits = in_nbits == 0? in_ty->nbits(): in_nbits;
  const size_t compress_ratio =  in_nbits / 8 + ((in_nbits % 8) > 0);
  const PtType compress_btype = calcBShareBacktype(8);
  size_t numel = in.numel();
  size_t compress_numel = numel * compress_ratio;
  Shape compress_shape = {1, static_cast<int64_t>(compress_numel)};
  NdArrayRef compress(makeType<ShareT>(compress_btype, 8), compress_shape);
  NdArrayRef out(makeType<OutShareT>(in_ty->getBacktype(), in_nbits), in.shape());

  return DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), [&]() {
      using in_el_t = ScalarT;
      using in_shr_t = std::array<in_el_t, num_shares>;
      using out_shr_t = std::array<in_el_t, out_num_shares>;
      NdArrayView<in_shr_t> _in(in);
      NdArrayView<out_shr_t> _out(out);

      return DISPATCH_UINT_PT_TYPES(compress_btype, [&]() {
          using compress_el_t = ScalarT;
          using compress_shr_t = std::array<compress_el_t, num_shares>;
          using out_compress_shr_t = std::array<compress_el_t, out_num_shares>;
          NdArrayView<compress_shr_t> _compress(compress);

          pforeach(0, compress_numel, [&](int64_t idx) {
            size_t loc = idx / compress_ratio;
            size_t cr = idx % compress_ratio;
            for (size_t s = 0; s < num_shares; s++)
            {
              compress_el_t& _cc = _compress[idx][s];
              compress_el_t _c = _in[loc][s];
              _cc = (_c >> (8 * cr)) & 0xFF;
            }
          });

          auto ret = func(compress);

          NdArrayView<out_compress_shr_t> _ret(ret);
          pforeach(0, numel, [&](int64_t idx) {
            size_t loc = idx * compress_ratio;
            for (size_t s = 0; s < out_num_shares; s++)
            {
              _out[idx][s] = 0;
              for (size_t cr = 0; cr < compress_ratio; cr++)
              {
                _out[idx][s] |= static_cast<in_el_t>(_ret[loc + cr][s]) << (8 * cr);
              }
            }
          });
          
          return out;
      });
  });
}

void AddRounds(KernelEvalContext* ctx, size_t rounds, bool reduce_spu=true, bool reduce_yacl=true);
void SubRounds(KernelEvalContext* ctx, size_t rounds, bool reduce_spu=true, bool reduce_yacl=true);

} // namespace spu::mpc::albo