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

#include "libspu/mpc/utils/ring_ops.h"

#include <cstring>
#include <random>

#include "absl/types/span.h"
#include "yacl/crypto/rand/rand.h"
#include "yacl/crypto/tools/prg.h"

#include "libspu/mpc/utils/linalg.h"

// TODO: ArrayRef is simple enough, consider using other SIMD libraries.
namespace spu::mpc {
namespace {

#define SPU_ENFORCE_RING(x)                                           \
  SPU_ENFORCE((x).eltype().isa<RingTy>(), "expect ring type, got={}", \
              (x).eltype());

#define ENFORCE_EQ_SHAPE(lhs, rhs)            \
  SPU_ENFORCE((lhs).shape() == (rhs).shape(), \
              "shape mismatch, lhs={}, rhs={}", lhs, rhs);

#define ENFORCE_EQ_ELSIZE_AND_SHAPE(lhs, rhs)                                  \
  SPU_ENFORCE((lhs).eltype().storage_type() == (rhs).eltype().storage_type(),  \
              "type mismatch lhs={}, rhs={}", (lhs).eltype(), (rhs).eltype()); \
  ENFORCE_EQ_SHAPE(lhs, rhs);

#define DEF_UNARY_RING_OP(NAME, OP)                                     \
  void NAME##_impl(MemRef& ret, const MemRef& x) {                      \
    ENFORCE_EQ_ELSIZE_AND_SHAPE(ret, x);                                \
    const auto back_type = x.eltype().storage_type();                   \
    const int64_t numel = ret.numel();                                  \
    return DISPATCH_ALL_STORAGE_TYPES(back_type, [&]() {                \
      using T = std::make_signed_t<ScalarT>;                            \
      MemRefView<T> _x(x);                                              \
      MemRefView<T> _ret(ret);                                          \
      pforeach(0, numel, [&](int64_t idx) { _ret[idx] = OP _x[idx]; }); \
    });                                                                 \
  }

DEF_UNARY_RING_OP(ring_not, ~);
DEF_UNARY_RING_OP(ring_neg, -);

#undef DEF_UNARY_RING_OP

#define DEF_BINARY_RING_OP(NAME, OP)                                  \
  void NAME##_impl(MemRef& ret, const MemRef& x, const MemRef& y) {   \
    ENFORCE_EQ_ELSIZE_AND_SHAPE(ret, x);                              \
    ENFORCE_EQ_ELSIZE_AND_SHAPE(ret, y);                              \
    const auto back_type = x.eltype().storage_type();                 \
    const int64_t numel = ret.numel();                                \
    return DISPATCH_ALL_STORAGE_TYPES(back_type, [&]() {              \
      MemRefView<ScalarT> _x(x);                                      \
      MemRefView<ScalarT> _y(y);                                      \
      MemRefView<ScalarT> _ret(ret);                                  \
      pforeach(0, numel,                                              \
               [&](int64_t idx) { _ret[idx] = _x[idx] OP _y[idx]; }); \
    });                                                               \
  }

DEF_BINARY_RING_OP(ring_add, +)
DEF_BINARY_RING_OP(ring_sub, -)

DEF_BINARY_RING_OP(ring_and, &)
DEF_BINARY_RING_OP(ring_xor, ^)

#undef DEF_BINARY_RING_OP

void ring_mul_impl(MemRef& ret, const MemRef& x, const MemRef& y) {
  ENFORCE_EQ_SHAPE(ret, x);
  ENFORCE_EQ_SHAPE(ret, y);
  const int64_t numel = ret.numel();

  DISPATCH_ALL_STORAGE_TYPES(x.eltype().storage_type(), [&]() {
    MemRefView<ScalarT> _x(x);
    DISPATCH_ALL_STORAGE_TYPES(y.eltype().storage_type(), [&]() {
      MemRefView<ScalarT> _y(y);
      DISPATCH_ALL_STORAGE_TYPES(ret.eltype().storage_type(), [&]() {
        MemRefView<ScalarT> _ret(ret);
        pforeach(0, numel, [&](int64_t idx) { _ret[idx] = _x[idx] * _y[idx]; });
      });
    });
  });
}

void ring_arshift_impl(MemRef& ret, const MemRef& x, const Sizes& bits) {
  ENFORCE_EQ_ELSIZE_AND_SHAPE(ret, x);
  bool is_splat = bits.size() == 1;
  SPU_ENFORCE(static_cast<int64_t>(bits.size()) == x.numel() || is_splat,
              "mismatched numel {} vs {}", bits.size(), x.numel());
  const auto numel = ret.numel();
  const auto back_type = x.eltype().storage_type();
  return DISPATCH_ALL_STORAGE_TYPES(back_type, [&]() {
    // According to K&R 2nd edition the results are implementation-dependent for
    // right shifts of signed values, but "usually" its arithmetic right shift.
    using S = std::make_signed<ScalarT>::type;
    MemRefView<S> _ret(ret);
    MemRefView<S> _x(x);
    pforeach(0, numel, [&](int64_t idx) {
      _ret[idx] = _x[idx] >> (is_splat ? bits[0] : bits[idx]);
    });
  });
}

void ring_rshift_impl(MemRef& ret, const MemRef& x, const Sizes& bits) {
  ENFORCE_EQ_ELSIZE_AND_SHAPE(ret, x);
  bool is_splat = bits.size() == 1;
  SPU_ENFORCE(static_cast<int64_t>(bits.size()) == x.numel() || is_splat,
              "mismatched numel {} vs {}", bits.size(), x.numel());
  const auto numel = ret.numel();
  const auto back_type = x.eltype().storage_type();
  return DISPATCH_ALL_STORAGE_TYPES(back_type, [&]() {
    using U = ScalarT;
    MemRefView<U> _ret(ret);
    MemRefView<U> _x(x);
    pforeach(0, numel, [&](int64_t idx) {
      _ret[idx] = _x[idx] >> (is_splat ? bits[0] : bits[idx]);
    });
  });
}

void ring_lshift_impl(MemRef& ret, const MemRef& x, const Sizes& bits) {
  ENFORCE_EQ_ELSIZE_AND_SHAPE(ret, x);
  bool is_splat = bits.size() == 1;
  SPU_ENFORCE(static_cast<int64_t>(bits.size()) == x.numel() || is_splat,
              "mismatched numel {} vs {}", bits.size(), x.numel());
  const auto numel = ret.numel();
  const auto back_type = x.eltype().storage_type();
  return DISPATCH_ALL_STORAGE_TYPES(back_type, [&]() {
    MemRefView<ScalarT> _ret(ret);
    MemRefView<ScalarT> _x(x);
    pforeach(0, numel, [&](int64_t idx) {
      _ret[idx] = _x[idx] << (is_splat ? bits[0] : bits[idx]);
    });
  });
}

void ring_bitrev_impl(MemRef& ret, const MemRef& x, size_t start, size_t end) {
  ENFORCE_EQ_ELSIZE_AND_SHAPE(ret, x);

  const auto back_type = x.eltype().storage_type();
  const auto numel = ret.numel();

  return DISPATCH_ALL_STORAGE_TYPES(back_type, [&]() {
    using U = ScalarT;

    // optimize: use faster reverse method.
    auto bitrev_fn = [&](U in) -> U {
      U tmp = 0U;
      for (size_t idx = start; idx < end; idx++) {
        if (in & ((U)1 << idx)) {
          tmp |= (U)1 << (end - 1 - idx + start);
        }
      }

      U mask = ((U)1U << end) - ((U)1U << start);
      return (in & ~mask) | tmp;
    };

    MemRefView<U> _ret(ret);
    MemRefView<U> _x(x);
    pforeach(0, numel, [&](int64_t idx) { _ret[idx] = bitrev_fn(_x[idx]); });
  });
}

void ring_bitmask_impl(MemRef& ret, const MemRef& x, size_t low, size_t high) {
  ENFORCE_EQ_ELSIZE_AND_SHAPE(ret, x);

  const auto back_type = x.eltype().storage_type();
  const auto numel = ret.numel();

  SPU_ENFORCE(low < high && high <= SizeOf(back_type) * 8);

  return DISPATCH_ALL_STORAGE_TYPES(back_type, [&]() {
    using U = ScalarT;
    U mask = 0;
    if (high - low < SizeOf(back_type) * 8) {
      mask = (U)1U << (high - low);
    }
    mask = (mask - 1) << low;

    auto mark_fn = [&](U el) { return el & mask; };

    MemRefView<U> _ret(ret);
    MemRefView<U> _x(x);
    pforeach(0, numel, [&](int64_t idx) { _ret[idx] = mark_fn(_x[idx]); });
  });
}

}  // namespace

// debug only
void ring_print(const MemRef& x, std::string_view name) {
  SPU_ENFORCE_RING(x);

  const auto back_type = x.eltype().storage_type();
  DISPATCH_ALL_STORAGE_TYPES(back_type, [&]() {
    using U = ScalarT;

    std::string out;
    out += fmt::format("{} = {{", name);
    MemRefView<U> _x(x);
    for (int64_t idx = 0; idx < x.numel(); idx++) {
      const auto& current_v = _x[idx];
      if (idx != 0) {
        out += fmt::format(", {0:X}", current_v);
      } else {
        out += fmt::format("{0:X}", current_v);
      }
    }
    out += fmt::format("}}\n");
    SPDLOG_INFO(out);
  });
}

void ring_rand(MemRef& in) {
  uint64_t cnt = 0;
  ring_rand(in, yacl::crypto::SecureRandSeed(), &cnt);
}

void ring_rand(MemRef& in, uint128_t prg_seed, uint64_t* prg_counter) {
  constexpr yacl::crypto::SymmetricCrypto::CryptoType kCryptoType =
      yacl::crypto::SymmetricCrypto::CryptoType::AES128_ECB;
  constexpr uint128_t kAesInitialVector = 0U;

  *prg_counter = yacl::crypto::FillPRand(
      kCryptoType, prg_seed, kAesInitialVector, *prg_counter,
      absl::MakeSpan(in.data<char>(), in.buf()->size()));
}

void ring_rand_range(MemRef& in, uint128_t min, uint128_t max) {
  constexpr yacl::crypto::SymmetricCrypto::CryptoType kCryptoType =
      yacl::crypto::SymmetricCrypto::CryptoType::AES128_ECB;
  constexpr uint64_t kAesInitialVector = 0U;
  uint64_t cnt = 0;

  auto numel = in.numel();
  DISPATCH_ALL_STORAGE_TYPES(in.eltype().storage_type(), [&]() {
    if constexpr (std::is_same_v<ScalarT, uint8_t> ||
                  std::is_same_v<ScalarT, uint16_t> ||
                  std::is_same_v<ScalarT, uint32_t>) {
      std::vector<uint32_t> rand_range(numel);
      yacl::crypto::FillPRandWithLtN<uint32_t>(
          kCryptoType, yacl::crypto::SecureRandSeed(), kAesInitialVector, cnt,
          absl::MakeSpan(rand_range), static_cast<uint32_t>(max - min + 1));
      auto iter = in.begin();
      for (auto idx = 0; idx < numel; ++idx, ++iter) {
        iter.getScalarValue<ScalarT>() =
            rand_range[idx] + static_cast<ScalarT>(min);
      }
    } else if constexpr (std::is_same_v<ScalarT, uint64_t>) {
      std::vector<uint64_t> rand_range(numel);
      yacl::crypto::FillPRandWithLtN<uint64_t>(
          kCryptoType, yacl::crypto::SecureRandSeed(), kAesInitialVector, cnt,
          absl::MakeSpan(rand_range), static_cast<uint64_t>(max - min + 1));
      auto iter = in.begin();
      for (auto idx = 0; idx < numel; ++idx, ++iter) {
        iter.getScalarValue<uint64_t>() =
            rand_range[idx] + static_cast<uint64_t>(min);
      }
    } else {
      std::vector<uint128_t> rand_range(numel);
      yacl::crypto::FillPRandWithLtN<uint128_t>(
          kCryptoType, yacl::crypto::SecureRandSeed(), kAesInitialVector, cnt,
          absl::MakeSpan(rand_range), max - min + 1);
      auto iter = in.begin();
      for (auto idx = 0; idx < numel; ++idx, ++iter) {
        iter.getScalarValue<uint128_t>() =
            rand_range[idx] + static_cast<uint128_t>(min);
      }
    }
  });
}

void ring_assign(MemRef& x, const MemRef& y) {
  SPU_ENFORCE_RING(x);
  ENFORCE_EQ_SHAPE(x, y);

  const auto numel = x.numel();

  bool is_unsigned = isUnsigned(y.eltype().semantic_type());

  DISPATCH_ALL_STORAGE_TYPES(x.eltype().storage_type(), [&]() {
    if (is_unsigned) {
      using XT = ScalarT;
      MemRefView<XT> _x(x);
      DISPATCH_ALL_STORAGE_TYPES(y.eltype().storage_type(), [&]() {
        using YT = ScalarT;
        MemRefView<YT> _y(y);
        pforeach(0, numel,
                 [&](int64_t idx) { _x[idx] = static_cast<XT>(_y[idx]); });
      });
    } else {
      using XT = std::make_signed_t<ScalarT>;
      MemRefView<XT> _x(x);
      DISPATCH_ALL_STORAGE_TYPES(y.eltype().storage_type(), [&]() {
        using YT = std::make_signed_t<ScalarT>;
        MemRefView<YT> _y(y);
        pforeach(0, numel,
                 [&](int64_t idx) { _x[idx] = static_cast<XT>(_y[idx]); });
      });
    }
  });
}

void ring_zeros(MemRef& in) {
  auto numel = in.numel();

  DISPATCH_ALL_STORAGE_TYPES(in.eltype().storage_type(), [&]() {
    MemRefView<ScalarT> _in(in);
    pforeach(0, numel, [&](int64_t idx) { _in[idx] = ScalarT(0); });
  });
}

void ring_ones(MemRef& in) {
  auto numel = in.numel();

  DISPATCH_ALL_STORAGE_TYPES(in.eltype().storage_type(), [&]() {
    MemRefView<ScalarT> _in(in);
    pforeach(0, numel, [&](int64_t idx) { _in[idx] = ScalarT(1); });
  });
}

void ring_randbit(MemRef& in) {
  auto numel = in.numel();

  DISPATCH_ALL_STORAGE_TYPES(in.eltype().storage_type(), [&]() {
    auto rand_bytes = yacl::crypto::RandBytes(numel, false);
    MemRefView<ScalarT> _in(in);
    for (auto idx = 0; idx < numel; ++idx) {
      _in[idx] = static_cast<ScalarT>(rand_bytes[idx]) & 0x1;
    }
  });
}

void ring_msb(MemRef& out, const MemRef& in) {
  SPU_ENFORCE(out.eltype().semantic_type() == SE_1,
              "Expect 1bit semantic ring, got = {}",
              out.eltype().semantic_type());
  MemRefView<bool> _out(out);

  const int64_t numel = in.numel();
  auto rshift_bits = in.elsize() * 8 - 1;

  DISPATCH_ALL_STORAGE_TYPES(in.eltype().storage_type(), [&]() {
    MemRefView<ScalarT> _in(in);
    pforeach(0, numel,
             [&](int64_t idx) { _out[idx] = (_in[idx] >> rshift_bits); });
  });
}

MemRef ring_not(const MemRef& x) {
  MemRef ret(x.eltype(), x.shape());
  ring_not_impl(ret, x);
  return ret;
}

void ring_not_(MemRef& x) { ring_not_impl(x, x); }

MemRef ring_neg(const MemRef& x) {
  MemRef res(x.eltype(), x.shape());
  ring_neg_impl(res, x);
  return res;
}

void ring_neg_(MemRef& x) { ring_neg_impl(x, x); }

MemRef ring_add(const MemRef& x, const MemRef& y) {
  MemRef res(x.eltype(), x.shape());
  ring_add_impl(res, x, y);
  return res;
}

void ring_add_(MemRef& x, const MemRef& y) { ring_add_impl(x, x, y); }

MemRef ring_sub(const MemRef& x, const MemRef& y) {
  MemRef res(x.eltype(), x.shape());
  ring_sub_impl(res, x, y);
  return res;
}

void ring_sub_(MemRef& x, const MemRef& y) { ring_sub_impl(x, x, y); }

MemRef ring_mul(const MemRef& x, const MemRef& y) {
  auto lhs_st = x.eltype().semantic_type();
  auto rhs_st = y.eltype().semantic_type();
  auto ret_set = std::max(lhs_st, rhs_st);
  auto ret_sst = std::max(x.eltype().storage_type(), y.eltype().storage_type());
  MemRef res(makeType<RingTy>(ret_set, SizeOf(ret_sst) * 8), x.shape());

  MemRef x_;
  MemRef y_;

  if (x.eltype().storage_type() != res.eltype().storage_type()) {
    x_ = MemRef(res.eltype(), x.shape());
    ring_assign(x_, x);
  } else {
    x_ = x;
  }
  if (y.eltype().storage_type() != res.eltype().storage_type()) {
    y_ = MemRef(res.eltype(), y.shape());
    ring_assign(y_, y);
  } else {
    y_ = y;
  }

  ring_mul_impl(res, x_, y_);
  return res;
}

void ring_mul_(MemRef& x, const MemRef& y) { ring_mul_impl(x, x, y); }

void ring_mul_impl(MemRef& ret, const MemRef& x, uint128_t y) {
  ENFORCE_EQ_ELSIZE_AND_SHAPE(ret, x);

  const auto numel = x.numel();
  const auto back_type = x.eltype().storage_type();
  DISPATCH_ALL_STORAGE_TYPES(back_type, [&]() {
    using U = std::make_unsigned<ScalarT>::type;
    MemRefView<U> _x(x);
    MemRefView<U> _ret(ret);
    pforeach(0, numel, [&](int64_t idx) { _ret[idx] = _x[idx] * y; });
  });
}

MemRef ring_mul(const MemRef& x, uint128_t y) {
  MemRef res(x.eltype(), x.shape());
  ring_mul_impl(res, x, y);
  return res;
}

void ring_mul_(MemRef& x, uint128_t y) { ring_mul_impl(x, x, y); }

MemRef ring_mul(MemRef&& x, uint128_t y) {
  ring_mul_impl(x, x, y);
  return std::move(x);
}

void ring_mmul_impl(MemRef& z, const MemRef& lhs, const MemRef& rhs) {
  SPU_ENFORCE(lhs.eltype().isa<RingTy>(), "lhs not ring, got={}", lhs.eltype());
  SPU_ENFORCE(rhs.eltype().isa<RingTy>(), "rhs not ring, got={}", rhs.eltype());
  SPU_ENFORCE(lhs.eltype().storage_type() == rhs.eltype().storage_type(),
              "lhs = {}, rhs = {}", lhs.eltype().storage_type(),
              rhs.eltype().storage_type());

  const auto back_type = lhs.eltype().storage_type();
  DISPATCH_ALL_STORAGE_TYPES(back_type, [&]() {
    const auto lhs_stride_scale = lhs.elsize() / sizeof(ScalarT);
    const auto rhs_stride_scale = rhs.elsize() / sizeof(ScalarT);
    const auto ret_stride_scale = z.elsize() / sizeof(ScalarT);
    const auto M = lhs.shape()[0];
    const auto K = lhs.shape()[1];
    const auto N = rhs.shape()[1];

    const auto LDA = lhs_stride_scale * lhs.strides()[0];
    const auto IDA = lhs_stride_scale * lhs.strides()[1];
    const auto LDB = rhs_stride_scale * rhs.strides()[0];
    const auto IDB = rhs_stride_scale * rhs.strides()[1];
    const auto LDC = ret_stride_scale * z.strides()[0];
    const auto IDC = ret_stride_scale * z.strides()[1];

    linalg::matmul(M, N, K, lhs.data<const ScalarT>(), LDA, IDA,
                   rhs.data<const ScalarT>(), LDB, IDB, z.data<ScalarT>(), LDC,
                   IDC);
  });
}

MemRef ring_mmul(const MemRef& lhs, const MemRef& rhs) {
  SPU_ENFORCE(lhs.shape().size() == 2 && rhs.shape().size() == 2);
  SPU_ENFORCE(lhs.shape()[1] == rhs.shape()[0],
              "contracting dim mismatch, lhs = {}, rhs = {}", lhs.shape()[1],
              rhs.shape()[0]);

  auto lhs_st = lhs.eltype().semantic_type();
  auto rhs_st = rhs.eltype().semantic_type();
  auto ret_set = std::max(lhs_st, rhs_st);
  auto ret_sst =
      std::max(lhs.eltype().storage_type(), rhs.eltype().storage_type());
  MemRef res(makeType<RingTy>(ret_set, SizeOf(ret_sst) * 8),
             {lhs.shape()[0], rhs.shape()[1]});

  MemRef lhs_;
  MemRef rhs_;

  if (lhs.eltype().storage_type() != res.eltype().storage_type()) {
    lhs_ = MemRef(res.eltype(), lhs.shape());
    ring_assign(lhs_, lhs);
  } else {
    lhs_ = lhs;
  }
  if (rhs.eltype().storage_type() != res.eltype().storage_type()) {
    rhs_ = MemRef(res.eltype(), rhs.shape());
    ring_assign(rhs_, rhs);
  } else {
    rhs_ = rhs;
  }

  ring_mmul_impl(res, lhs_, rhs_);

  return res;
}

void ring_mmul_(MemRef& out, const MemRef& lhs, const MemRef& rhs) {
  SPU_ENFORCE(lhs.shape()[1] == rhs.shape()[0],
              "contracting dim mismatch, lhs = {}, rhs = {}", lhs.shape()[1],
              rhs.shape()[0]);

  ring_mmul_impl(out, lhs, rhs);
}

MemRef ring_and(const MemRef& x, const MemRef& y) {
  MemRef res(x.eltype(), x.shape());
  ring_and_impl(res, x, y);
  return res;
}

void ring_and_(MemRef& x, const MemRef& y) { ring_and_impl(x, x, y); }

MemRef ring_xor(const MemRef& x, const MemRef& y) {
  MemRef res(x.eltype(), x.shape());
  ring_xor_impl(res, x, y);
  return res;
}

void ring_xor_(MemRef& x, const MemRef& y) { ring_xor_impl(x, x, y); }

void ring_equal(MemRef& ret, const MemRef& x, const MemRef& y) {
  ENFORCE_EQ_ELSIZE_AND_SHAPE(y, x);
  const auto back_type = x.eltype().storage_type();
  const int64_t numel = ret.numel();
  DISPATCH_ALL_STORAGE_TYPES(back_type, [&]() {
    MemRefView<ScalarT> _x(x);
    MemRefView<ScalarT> _y(y);
    MemRefView<bool> _ret(ret);
    pforeach(0, numel, [&](int64_t idx) { _ret[idx] = _x[idx] == _y[idx]; });
  });
}

MemRef ring_arshift(const MemRef& x, const Sizes& bits) {
  MemRef res(x.eltype(), x.shape());
  ring_arshift_impl(res, x, bits);
  return res;
}

void ring_arshift_(MemRef& x, const Sizes& bits) {
  ring_arshift_impl(x, x, bits);
}

MemRef ring_rshift(const MemRef& x, const Sizes& bits) {
  MemRef res(x.eltype(), x.shape());
  ring_rshift_impl(res, x, bits);
  return res;
}

void ring_rshift_(MemRef& x, const Sizes& bits) {
  ring_rshift_impl(x, x, bits);
}

MemRef ring_lshift(const MemRef& x, const Sizes& bits) {
  MemRef res(x.eltype(), x.shape());
  ring_lshift_impl(res, x, bits);
  return res;
}

void ring_lshift_(MemRef& x, const Sizes& bits) {
  ring_lshift_impl(x, x, bits);
}

MemRef ring_bitrev(const MemRef& x, size_t start, size_t end) {
  MemRef res(x.eltype(), x.shape());
  ring_bitrev_impl(res, x, start, end);
  return res;
}

void ring_bitrev_(MemRef& x, size_t start, size_t end) {
  ring_bitrev_impl(x, x, start, end);
}

MemRef ring_bitmask(const MemRef& x, size_t low, size_t high) {
  MemRef ret(x.eltype(), x.shape());
  ring_bitmask_impl(ret, x, low, high);
  return ret;
}

void ring_bitmask_(MemRef& x, size_t low, size_t high) {
  ring_bitmask_impl(x, x, low, high);
}

MemRef ring_sum(absl::Span<MemRef const> arrs) {
  SPU_ENFORCE(!arrs.empty(), "expected non empty, got size={}", arrs.size());

  if (arrs.size() == 1) {
    return arrs[0];
  }

  SPU_ENFORCE(arrs.size() >= 2);
  auto res = ring_add(arrs[0], arrs[1]);
  for (size_t idx = 2; idx < arrs.size(); idx++) {
    ring_add_(res, arrs[idx]);
  }
  return res;
}

bool ring_all_equal(const MemRef& x, const MemRef& y, size_t abs_err) {
  ENFORCE_EQ_SHAPE(x, y);
  SPU_ENFORCE(x.eltype().semantic_type() == y.eltype().semantic_type(),
              "lhs = {}, rhs = {}", x.eltype(), y.eltype());

  auto numel = x.numel();

  return DISPATCH_ALL_STORAGE_TYPES(x.eltype().storage_type(), [&]() {
    using T = std::make_signed_t<ScalarT>;
    MemRefView<T> _x(x);

    return DISPATCH_ALL_STORAGE_TYPES(y.eltype().storage_type(), [&]() {
      using T = std::make_signed_t<ScalarT>;
      MemRefView<T> _y(y);

      for (int64_t idx = 0; idx < numel; ++idx) {
        auto x_el = _x[idx];
        auto y_el = _y[idx];
        if (std::abs(x_el - y_el) > static_cast<T>(abs_err)) {
          fmt::print("error: {0} {1} abs_err: {2}\n", x_el, y_el, abs_err);
          return false;
        }
      }
      return true;
    });
  });
}

bool ring_all_equal(const MemRef& x, int64_t val) {
  auto numel = x.numel();

  return DISPATCH_ALL_STORAGE_TYPES(x.eltype().storage_type(), [&]() {
    using T = std::make_signed_t<ScalarT>;

    MemRefView<T> _x(x);

    bool passed = true;
    for (int64_t idx = 0; idx < numel; ++idx) {
      auto x_el = _x[idx];
      if (_x[idx] != (T)val) {
        fmt::print("error: {0} != {1} at {2}\n", x_el, val, idx);
        return false;
      }
    }
    return passed;
  });
}

std::vector<uint8_t> ring_cast_boolean(const MemRef& x) {
  // SPU_ENFORCE_RING(x);
  const auto back_type = x.eltype().storage_type();

  auto numel = x.numel();
  std::vector<uint8_t> res(numel);

  DISPATCH_ALL_STORAGE_TYPES(back_type, [&]() {
    MemRefView<ScalarT> _x(x);
    pforeach(0, numel, [&](int64_t idx) {
      res[idx] = static_cast<uint8_t>(_x[idx] & 0x1);
    });
  });

  return res;
}

MemRef ring_select(const std::vector<uint8_t>& c, const MemRef& x,
                   const MemRef& y) {
  ENFORCE_EQ_ELSIZE_AND_SHAPE(x, y);
  SPU_ENFORCE(x.numel() == y.numel());
  SPU_ENFORCE(x.numel() == static_cast<int64_t>(c.size()));

  const auto back_type = x.eltype().storage_type();
  MemRef z(x.eltype(), x.shape());
  const int64_t numel = c.size();

  DISPATCH_ALL_STORAGE_TYPES(back_type, [&]() {
    MemRefView<ScalarT> _x(x);
    MemRefView<ScalarT> _y(y);
    MemRefView<ScalarT> _z(z);

    pforeach(0, numel,
             [&](int64_t idx) { _z[idx] = (c[idx] ? _y[idx] : _x[idx]); });
  });

  return z;
}

MemRef ring_select(const MemRef& c, const MemRef& x, const MemRef& y) {
  ENFORCE_EQ_ELSIZE_AND_SHAPE(x, y);
  SPU_ENFORCE(c.shape() == y.shape());
  SPU_ENFORCE(c.shape() == x.shape());

  const auto back_type = x.eltype().storage_type();
  MemRef z(x.eltype(), x.shape());
  const int64_t numel = c.numel();

  MemRefView<bool> _c(c);

  DISPATCH_ALL_STORAGE_TYPES(back_type, [&]() {
    MemRefView<ScalarT> _x(x);
    MemRefView<ScalarT> _y(y);
    MemRefView<ScalarT> _z(z);

    pforeach(0, numel,
             [&](int64_t idx) { _z[idx] = (_c[idx] ? _y[idx] : _x[idx]); });
  });

  return z;
}

std::vector<MemRef> ring_rand_additive_splits(const MemRef& arr,
                                              size_t num_splits) {
  SPU_ENFORCE(num_splits > 1, "num split {} be greater than 1 ", num_splits);

  std::vector<MemRef> splits(num_splits);
  splits[0] = arr.clone();

  for (size_t idx = 1; idx < num_splits; idx++) {
    splits[idx] = MemRef(arr.eltype(), arr.shape());
    ring_rand(splits[idx]);
    ring_sub_(splits[0], splits[idx]);
  }

  return splits;
}

std::vector<MemRef> ring_rand_boolean_splits(const MemRef& arr,
                                             size_t num_splits) {
  SPU_ENFORCE(num_splits > 1, "num split {} be greater than 1 ", num_splits);

  std::vector<MemRef> splits(num_splits);
  splits[0] = arr.clone();

  for (size_t idx = 1; idx < num_splits; idx++) {
    splits[idx] = MemRef(arr.eltype(), arr.shape());
    ring_rand(splits[idx]);
    ring_xor_(splits[0], splits[idx]);
  }

  return splits;
}
}  // namespace spu::mpc
