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

#define PFOR_GRAIN_SIZE 4096

#include "libspu/mpc/utils/ring_ops.h"

#include <cstring>
#include <random>

#include "absl/types/span.h"
#include "yacl/crypto/tools/prg.h"
#include "yacl/crypto/utils/rand.h"
#include "yacl/utils/parallel.h"

#include "libspu/mpc/utils/linalg.h"

// TODO: ArrayRef is simple enough, consider using other SIMD libraries.
namespace spu::mpc {
namespace {

constexpr char kModule[] = "RingOps";

#define SPU_ENFORCE_RING(x)                                           \
  SPU_ENFORCE((x).eltype().isa<Ring2k>(), "expect ring type, got={}", \
              (x).eltype());

#define ENFORCE_EQ_ELSIZE_AND_SHAPE(lhs, rhs)                                  \
  SPU_ENFORCE((lhs).eltype().as<Ring2k>()->field() ==                          \
                  (rhs).eltype().as<Ring2k>()->field(),                        \
              "type mismatch lhs={}, rhs={}", (lhs).eltype(), (rhs).eltype()); \
  SPU_ENFORCE((lhs).shape() == (rhs).shape(),                                  \
              "numel mismatch, lhs={}, rhs={}", lhs, rhs);

#define DEF_UNARY_RING_OP(NAME, OP)                                     \
  void NAME##_impl(NdArrayRef& ret, const NdArrayRef& x) {              \
    ENFORCE_EQ_ELSIZE_AND_SHAPE(ret, x);                                \
    const auto field = x.eltype().as<Ring2k>()->field();                \
    const int64_t numel = ret.numel();                                  \
    return DISPATCH_ALL_FIELDS(field, kModule, [&]() {                  \
      using T = std::make_signed_t<ring2k_t>;                           \
      NdArrayView<T> _x(x);                                             \
      NdArrayView<T> _ret(ret);                                         \
      pforeach(0, numel, [&](int64_t idx) { _ret[idx] = OP _x[idx]; }); \
    });                                                                 \
  }

DEF_UNARY_RING_OP(ring_not, ~);
DEF_UNARY_RING_OP(ring_neg, -);

#undef DEF_UNARY_RING_OP

#define DEF_BINARY_RING_OP(NAME, OP)                                  \
  void NAME##_impl(NdArrayRef& ret, const NdArrayRef& x,              \
                   const NdArrayRef& y) {                             \
    ENFORCE_EQ_ELSIZE_AND_SHAPE(ret, x);                              \
    ENFORCE_EQ_ELSIZE_AND_SHAPE(ret, y);                              \
    const auto field = x.eltype().as<Ring2k>()->field();              \
    const int64_t numel = ret.numel();                                \
    return DISPATCH_ALL_FIELDS(field, kModule, [&]() {                \
      NdArrayView<ring2k_t> _x(x);                                    \
      NdArrayView<ring2k_t> _y(y);                                    \
      NdArrayView<ring2k_t> _ret(ret);                                \
      pforeach(0, numel,                                              \
               [&](int64_t idx) { _ret[idx] = _x[idx] OP _y[idx]; }); \
    });                                                               \
  }

DEF_BINARY_RING_OP(ring_add, +)
DEF_BINARY_RING_OP(ring_sub, -)
DEF_BINARY_RING_OP(ring_mul, *)
DEF_BINARY_RING_OP(ring_equal, ==)

DEF_BINARY_RING_OP(ring_and, &);
DEF_BINARY_RING_OP(ring_xor, ^);

#undef DEF_BINARY_RING_OP

void ring_arshift_impl(NdArrayRef& ret, const NdArrayRef& x, size_t bits) {
  ENFORCE_EQ_ELSIZE_AND_SHAPE(ret, x);
  const auto numel = ret.numel();
  const auto field = x.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, kModule, [&]() {
    // According to K&R 2nd edition the results are implementation-dependent for
    // right shifts of signed values, but "usually" its arithmetic right shift.
    using S = std::make_signed<ring2k_t>::type;
    NdArrayView<S> _ret(ret);
    NdArrayView<S> _x(x);
    pforeach(0, numel, [&](int64_t idx) { _ret[idx] = _x[idx] >> bits; });
  });
}

void ring_rshift_impl(NdArrayRef& ret, const NdArrayRef& x, size_t bits) {
  ENFORCE_EQ_ELSIZE_AND_SHAPE(ret, x);
  const auto numel = ret.numel();
  const auto field = x.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, kModule, [&]() {
    using U = ring2k_t;
    NdArrayView<U> _ret(ret);
    NdArrayView<U> _x(x);
    pforeach(0, numel, [&](int64_t idx) { _ret[idx] = _x[idx] >> bits; });
  });
}

void ring_lshift_impl(NdArrayRef& ret, const NdArrayRef& x, size_t bits) {
  ENFORCE_EQ_ELSIZE_AND_SHAPE(ret, x);
  const auto numel = ret.numel();
  const auto field = x.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, kModule, [&]() {
    NdArrayView<ring2k_t> _ret(ret);
    NdArrayView<ring2k_t> _x(x);
    pforeach(0, numel, [&](int64_t idx) { _ret[idx] = _x[idx] << bits; });
  });
}

void ring_bitrev_impl(NdArrayRef& ret, const NdArrayRef& x, size_t start,
                      size_t end) {
  ENFORCE_EQ_ELSIZE_AND_SHAPE(ret, x);

  const auto field = x.eltype().as<Ring2k>()->field();
  const auto numel = ret.numel();

  return DISPATCH_ALL_FIELDS(field, kModule, [&]() {
    using U = ring2k_t;

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

    NdArrayView<U> _ret(ret);
    NdArrayView<U> _x(x);
    pforeach(0, numel, [&](int64_t idx) { _ret[idx] = bitrev_fn(_x[idx]); });
  });
}

void ring_bitmask_impl(NdArrayRef& ret, const NdArrayRef& x, size_t low,
                       size_t high) {
  ENFORCE_EQ_ELSIZE_AND_SHAPE(ret, x);

  const auto field = x.eltype().as<Ring2k>()->field();
  const auto numel = ret.numel();

  SPU_ENFORCE(low < high && high <= SizeOf(field) * 8);

  return DISPATCH_ALL_FIELDS(field, kModule, [&]() {
    using U = ring2k_t;
    U mask = 0;
    if (high - low < SizeOf(field) * 8) {
      mask = (U)1U << (high - low);
    }
    mask = (mask - 1) << low;

    auto mark_fn = [&](U el) { return el & mask; };

    NdArrayView<U> _ret(ret);
    NdArrayView<U> _x(x);
    pforeach(0, numel, [&](int64_t idx) { _ret[idx] = mark_fn(_x[idx]); });
  });
}

}  // namespace

// debug only
void ring_print(const NdArrayRef& x, std::string_view name) {
  SPU_ENFORCE_RING(x);

  const auto field = x.eltype().as<Ring2k>()->field();
  DISPATCH_ALL_FIELDS(field, kModule, [&]() {
    using U = ring2k_t;

    std::string out;
    out += fmt::format("{} = {{", name);
    NdArrayView<U> _x(x);
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

NdArrayRef ring_rand(FieldType field, const Shape& shape) {
  uint64_t cnt = 0;
  return ring_rand(field, shape, yacl::crypto::RandSeed(), &cnt);
}

NdArrayRef ring_rand(FieldType field, const Shape& shape, uint128_t prg_seed,
                     uint64_t* prg_counter) {
  constexpr yacl::crypto::SymmetricCrypto::CryptoType kCryptoType =
      yacl::crypto::SymmetricCrypto::CryptoType::AES128_CTR;
  constexpr uint128_t kAesInitialVector = 0U;

  NdArrayRef res(makeType<RingTy>(field), shape);
  *prg_counter = yacl::crypto::FillPRand(
      kCryptoType, prg_seed, kAesInitialVector, *prg_counter,
      absl::MakeSpan(res.data<char>(), res.buf()->size()));

  return res;
}

NdArrayRef ring_rand_range(FieldType field, const Shape& shape, int32_t min,
                           int32_t max) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int32_t> dis(min, max);

  NdArrayRef x(makeType<RingTy>(field), shape);
  auto numel = x.numel();

  DISPATCH_ALL_FIELDS(field, kModule, [&]() {
    SPU_ENFORCE(sizeof(ring2k_t) >= sizeof(int32_t));

    auto iter = x.begin();
    for (auto idx = 0; idx < numel; ++idx, ++iter) {
      iter.getScalarValue<ring2k_t>() = static_cast<ring2k_t>(dis(gen));
    }
  });

  return x;
}

void ring_assign(NdArrayRef& x, const NdArrayRef& y) {
  SPU_ENFORCE_RING(x);
  ENFORCE_EQ_ELSIZE_AND_SHAPE(x, y);

  const auto numel = x.numel();

  const auto field = x.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, kModule, [&]() {
    NdArrayView<ring2k_t> _y(y);
    NdArrayView<ring2k_t> _x(x);
    pforeach(0, numel, [&](int64_t idx) { _x[idx] = _y[idx]; });
  });
}

NdArrayRef ring_zeros(FieldType field, const Shape& shape) {
  NdArrayRef ret(makeType<RingTy>(field), shape);
  auto numel = ret.numel();

  return DISPATCH_ALL_FIELDS(field, kModule, [&]() {
    NdArrayView<ring2k_t> _ret(ret);
    pforeach(0, numel, [&](int64_t idx) { _ret[idx] = ring2k_t(0); });
    return ret;
  });
}

NdArrayRef ring_ones(FieldType field, const Shape& shape) {
  NdArrayRef ret(makeType<RingTy>(field), shape);
  auto numel = ret.numel();

  return DISPATCH_ALL_FIELDS(field, kModule, [&]() {
    NdArrayView<ring2k_t> _ret(ret);
    pforeach(0, numel, [&](int64_t idx) { _ret[idx] = ring2k_t(1); });
    return ret;
  });
}

NdArrayRef ring_randbit(FieldType field, const Shape& shape) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(0, RAND_MAX);

  NdArrayRef ret(makeType<RingTy>(field), shape);
  auto numel = ret.numel();

  return DISPATCH_ALL_FIELDS(field, kModule, [&]() {
    NdArrayView<ring2k_t> _ret(ret);
    for (auto idx = 0; idx < numel; ++idx) {
      _ret[idx] = distrib(gen) & 0x1;
    }
    return ret;
  });
}

NdArrayRef ring_not(const NdArrayRef& x) {
  NdArrayRef ret(x.eltype(), x.shape());
  ring_not_impl(ret, x);
  return ret;
}

void ring_not_(NdArrayRef& x) { ring_not_impl(x, x); }

NdArrayRef ring_neg(const NdArrayRef& x) {
  NdArrayRef res(x.eltype(), x.shape());
  ring_neg_impl(res, x);
  return res;
}

void ring_neg_(NdArrayRef& x) { ring_neg_impl(x, x); }

NdArrayRef ring_add(const NdArrayRef& x, const NdArrayRef& y) {
  NdArrayRef res(x.eltype(), x.shape());
  ring_add_impl(res, x, y);
  return res;
}

void ring_add_(NdArrayRef& x, const NdArrayRef& y) { ring_add_impl(x, x, y); }

NdArrayRef ring_sub(const NdArrayRef& x, const NdArrayRef& y) {
  NdArrayRef res(x.eltype(), x.shape());
  ring_sub_impl(res, x, y);
  return res;
}

void ring_sub_(NdArrayRef& x, const NdArrayRef& y) { ring_sub_impl(x, x, y); }

NdArrayRef ring_mul(const NdArrayRef& x, const NdArrayRef& y) {
  NdArrayRef res(x.eltype(), x.shape());
  ring_mul_impl(res, x, y);
  return res;
}
void ring_mul_(NdArrayRef& x, const NdArrayRef& y) { ring_mul_impl(x, x, y); }

void ring_mul_impl(NdArrayRef& ret, const NdArrayRef& x, uint128_t y) {
  ENFORCE_EQ_ELSIZE_AND_SHAPE(ret, x);

  const auto numel = x.numel();
  const auto field = x.eltype().as<Ring2k>()->field();
  DISPATCH_ALL_FIELDS(field, kModule, [&]() {
    using U = std::make_unsigned<ring2k_t>::type;
    NdArrayView<U> _x(x);
    NdArrayView<U> _ret(ret);
    pforeach(0, numel, [&](int64_t idx) { _ret[idx] = _x[idx] * y; });
  });
}

NdArrayRef ring_mul(const NdArrayRef& x, uint128_t y) {
  NdArrayRef res(x.eltype(), x.shape());
  ring_mul_impl(res, x, y);
  return res;
}

void ring_mul_(NdArrayRef& x, uint128_t y) { ring_mul_impl(x, x, y); }

void ring_mmul_impl(NdArrayRef& z, const NdArrayRef& lhs,
                    const NdArrayRef& rhs) {
  SPU_ENFORCE(lhs.eltype().isa<Ring2k>(), "lhs not ring, got={}", lhs.eltype());
  SPU_ENFORCE(rhs.eltype().isa<Ring2k>(), "rhs not ring, got={}", rhs.eltype());

  const auto field = lhs.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, kModule, [&]() {
    const auto lhs_stride_scale = lhs.elsize() / sizeof(ring2k_t);
    const auto rhs_stride_scale = rhs.elsize() / sizeof(ring2k_t);
    const auto ret_stride_scale = z.elsize() / sizeof(ring2k_t);
    const auto M = lhs.shape()[0];
    const auto K = lhs.shape()[1];
    const auto N = rhs.shape()[1];

    const auto LDA = lhs_stride_scale * lhs.strides()[0];
    const auto IDA = lhs_stride_scale * lhs.strides()[1];
    const auto LDB = rhs_stride_scale * rhs.strides()[0];
    const auto IDB = rhs_stride_scale * rhs.strides()[1];
    const auto LDC = ret_stride_scale * z.strides()[0];
    const auto IDC = ret_stride_scale * z.strides()[1];

    linalg::matmul(M, N, K, lhs.data<const ring2k_t>(), LDA, IDA,
                   rhs.data<const ring2k_t>(), LDB, IDB, z.data<ring2k_t>(),
                   LDC, IDC);
  });
}

NdArrayRef ring_mmul(const NdArrayRef& lhs, const NdArrayRef& rhs) {
  SPU_ENFORCE(lhs.shape().size() == 2 && rhs.shape().size() == 2);
  SPU_ENFORCE(lhs.shape()[1] == rhs.shape()[0],
              "contracting dim mismatch, lhs = {}, rhs = {}", lhs.shape()[1],
              rhs.shape()[0]);

  NdArrayRef ret(lhs.eltype(), {lhs.shape()[0], rhs.shape()[1]});

  ring_mmul_impl(ret, lhs, rhs);

  return ret;
}

void ring_mmul_(NdArrayRef& out, const NdArrayRef& lhs, const NdArrayRef& rhs) {
  SPU_ENFORCE(lhs.shape()[1] == rhs.shape()[0],
              "contracting dim mismatch, lhs = {}, rhs = {}", lhs.shape()[1],
              rhs.shape()[0]);

  ring_mmul_impl(out, lhs, rhs);
}

NdArrayRef ring_and(const NdArrayRef& x, const NdArrayRef& y) {
  NdArrayRef res(x.eltype(), x.shape());
  ring_and_impl(res, x, y);
  return res;
}

void ring_and_(NdArrayRef& x, const NdArrayRef& y) { ring_and_impl(x, x, y); }

NdArrayRef ring_xor(const NdArrayRef& x, const NdArrayRef& y) {
  NdArrayRef res(x.eltype(), x.shape());
  ring_xor_impl(res, x, y);
  return res;
}

void ring_xor_(NdArrayRef& x, const NdArrayRef& y) { ring_xor_impl(x, x, y); }

NdArrayRef ring_equal(const NdArrayRef& x, const NdArrayRef& y) {
  NdArrayRef res(x.eltype(), x.shape());
  ring_equal_impl(res, x, y);
  return res;
}

void ring_equal(NdArrayRef& x, const NdArrayRef& y) {
  ring_equal_impl(x, x, y);
}

NdArrayRef ring_arshift(const NdArrayRef& x, size_t bits) {
  NdArrayRef res(x.eltype(), x.shape());
  ring_arshift_impl(res, x, bits);
  return res;
}

void ring_arshift_(NdArrayRef& x, size_t bits) {
  ring_arshift_impl(x, x, bits);
}

NdArrayRef ring_rshift(const NdArrayRef& x, size_t bits) {
  NdArrayRef res(x.eltype(), x.shape());
  ring_rshift_impl(res, x, bits);
  return res;
}

void ring_rshift_(NdArrayRef& x, size_t bits) { ring_rshift_impl(x, x, bits); }

NdArrayRef ring_lshift(const NdArrayRef& x, size_t bits) {
  NdArrayRef res(x.eltype(), x.shape());
  ring_lshift_impl(res, x, bits);
  return res;
}

void ring_lshift_(NdArrayRef& x, size_t bits) { ring_lshift_impl(x, x, bits); }

NdArrayRef ring_bitrev(const NdArrayRef& x, size_t start, size_t end) {
  NdArrayRef res(x.eltype(), x.shape());
  ring_bitrev_impl(res, x, start, end);
  return res;
}

void ring_bitrev_(NdArrayRef& x, size_t start, size_t end) {
  ring_bitrev_impl(x, x, start, end);
}

NdArrayRef ring_bitmask(const NdArrayRef& x, size_t low, size_t high) {
  NdArrayRef ret(x.eltype(), x.shape());
  ring_bitmask_impl(ret, x, low, high);
  return ret;
}

void ring_bitmask_(NdArrayRef& x, size_t low, size_t high) {
  ring_bitmask_impl(x, x, low, high);
}

NdArrayRef ring_sum(absl::Span<NdArrayRef const> arrs) {
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

bool ring_all_equal(const NdArrayRef& x, const NdArrayRef& y, size_t abs_err) {
  ENFORCE_EQ_ELSIZE_AND_SHAPE(x, y);

  auto numel = x.numel();

  const auto field = x.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using T = std::make_signed_t<ring2k_t>;

    NdArrayView<T> _x(x);
    NdArrayView<T> _y(y);

    bool passed = true;
    for (int64_t idx = 0; idx < numel; ++idx) {
      auto x_el = _x[idx];
      auto y_el = _y[idx];
      if (std::abs(x_el - y_el) > static_cast<T>(abs_err)) {
        fmt::print("error: {0} {1} abs_err: {2}\n", x_el, y_el, abs_err);
        return false;
      }
    }
    return passed;
  });
}

std::vector<uint8_t> ring_cast_boolean(const NdArrayRef& x) {
  // SPU_ENFORCE_RING(x);
  const auto field = x.eltype().as<Ring2k>()->field();

  auto numel = x.numel();
  std::vector<uint8_t> res(numel);

  DISPATCH_ALL_FIELDS(field, kModule, [&]() {
    NdArrayView<ring2k_t> _x(x);
    pforeach(0, numel, [&](int64_t idx) {
      res[idx] = static_cast<uint8_t>(_x[idx] & 0x1);
    });
  });

  return res;
}

NdArrayRef ring_select(const std::vector<uint8_t>& c, const NdArrayRef& x,
                       const NdArrayRef& y) {
  ENFORCE_EQ_ELSIZE_AND_SHAPE(x, y);
  SPU_ENFORCE(x.numel() == y.numel());
  SPU_ENFORCE(x.numel() == static_cast<int64_t>(c.size()));

  const auto field = x.eltype().as<Ring2k>()->field();
  NdArrayRef z(x.eltype(), x.shape());
  const int64_t numel = c.size();

  DISPATCH_ALL_FIELDS(field, kModule, [&]() {
    NdArrayView<ring2k_t> _x(x);
    NdArrayView<ring2k_t> _y(y);
    NdArrayView<ring2k_t> _z(z);

    pforeach(0, numel,
             [&](int64_t idx) { _z[idx] = (c[idx] ? _y[idx] : _x[idx]); });
  });

  return z;
}

std::vector<NdArrayRef> ring_rand_additive_splits(const NdArrayRef& arr,
                                                  size_t num_splits) {
  const auto field = arr.eltype().as<Ring2k>()->field();
  SPU_ENFORCE(num_splits > 1, "num split {} be greater than 1 ", num_splits);

  std::vector<NdArrayRef> splits(num_splits);
  splits[0] = arr.clone();

  for (size_t idx = 1; idx < num_splits; idx++) {
    splits[idx] = ring_rand(field, arr.shape());
    ring_sub_(splits[0], splits[idx]);
  }

  return splits;
}

std::vector<NdArrayRef> ring_rand_boolean_splits(const NdArrayRef& arr,
                                                 size_t num_splits) {
  const auto field = arr.eltype().as<Ring2k>()->field();
  SPU_ENFORCE(num_splits > 1, "num split {} be greater than 1 ", num_splits);

  std::vector<NdArrayRef> splits(num_splits);
  splits[0] = arr.clone();

  for (size_t idx = 1; idx < num_splits; idx++) {
    splits[idx] = ring_rand(field, arr.shape());
    ring_xor_(splits[0], splits[idx]);
  }

  return splits;
}

}  // namespace spu::mpc
