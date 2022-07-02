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

#include "spu/mpc/util/ring_ops.h"

#include <cstring>
#include <random>

#include "absl/types/span.h"
#include "yasl/crypto/pseudo_random_generator.h"
#include "yasl/utils/parallel.h"

#include "spu/core/array_ref.h"
#include "spu/mpc/util/linalg.h"

// TODO: ArrayRef is simple enough, consider using other SIMD libraries.
namespace spu::mpc {
namespace {

constexpr char kModule[] = "RingOps";

void strided_copy(int64_t numel, int64_t elsize, void* dst, int64_t dstride,
                  void const* src, int64_t sstride) {
  // WARN: the following method does not work
  // due to https://github.com/xtensor-stack/xtensor/issues/2330
  // return DISPATCH_ALL_FIELDS(field, kModule, [&]() {
  //  auto _x = xt_mutable_adapt<ring2k_t>(x);
  //  const auto& _y = xt_adapt<ring2k_t>(y);
  //  _x = _y;
  //});

  const char* src_itr = static_cast<const char*>(src);
  char* dst_itr = static_cast<char*>(dst);

  yasl::parallel_for(0, numel, PFOR_GRAIN_SIZE,
                     [&](int64_t begin, int64_t end) {
                       for (int64_t idx = begin; idx < end; ++idx) {
                         std::memcpy(&dst_itr[idx * dstride * elsize],
                                     &src_itr[idx * sstride * elsize], elsize);
                       }
                     });
}

#define YASL_ENFORCE_RING(x)                                         \
  YASL_ENFORCE(x.eltype().isa<Ring2k>(), "expect ring type, got={}", \
               x.eltype());

#define ENFORCE_EQ_ELSIZE_AND_NUMEL(lhs, rhs)                                \
  YASL_ENFORCE(lhs.eltype().as<Ring2k>()->field() ==                         \
                   rhs.eltype().as<Ring2k>()->field(),                       \
               "type mismatch lhs={}, rhs={}", lhs.eltype(), rhs.eltype());  \
  YASL_ENFORCE(lhs.numel() == rhs.numel(), "numel mismatch, lhs={}, rhs={}", \
               lhs.numel(), rhs.numel());

#define DEF_UNARY_RING_OP(NAME, COP)                                   \
  void NAME##_impl(ArrayRef& ret, const ArrayRef& x) {                 \
    ENFORCE_EQ_ELSIZE_AND_NUMEL(ret, x);                               \
    const auto field = x.eltype().as<Ring2k>()->field();               \
    const int64_t numel = ret.numel();                                 \
    return DISPATCH_ALL_FIELDS(field, kModule, [&]() {                 \
      const auto* x_itr = &x.at<ring2k_t>(0);                          \
      const int64_t x_stride = x.stride();                             \
      auto* z_itr = &ret.at<ring2k_t>(0);                              \
      const int64_t z_stride = ret.stride();                           \
      yasl::parallel_for(                                              \
          0, numel, PFOR_GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
            for (int64_t idx = begin; idx < end; ++idx) {              \
              z_itr[idx * z_stride] = COP x_itr[idx * x_stride];       \
            }                                                          \
          });                                                          \
    });                                                                \
  }

#define DEF_BINARY_RING_OP(NAME, COP)                                     \
  void NAME##_impl(ArrayRef& ret, const ArrayRef& x, const ArrayRef& y) { \
    ENFORCE_EQ_ELSIZE_AND_NUMEL(ret, x);                                  \
    ENFORCE_EQ_ELSIZE_AND_NUMEL(ret, y);                                  \
    const auto field = x.eltype().as<Ring2k>()->field();                  \
    const int64_t numel = ret.numel();                                    \
    return DISPATCH_ALL_FIELDS(field, kModule, [&]() {                    \
      const auto* x_itr = &x.at<ring2k_t>(0);                             \
      const int64_t x_stride = x.stride();                                \
      const auto* y_itr = &y.at<ring2k_t>(0);                             \
      const int64_t y_stride = y.stride();                                \
      auto* z_itr = &ret.at<ring2k_t>(0);                                 \
      const int64_t z_stride = ret.stride();                              \
                                                                          \
      yasl::parallel_for(                                                 \
          0, numel, PFOR_GRAIN_SIZE, [&](int64_t begin, int64_t end) {    \
            for (int64_t idx = begin; idx < end; ++idx) {                 \
              z_itr[idx * z_stride] =                                     \
                  x_itr[idx * x_stride] COP y_itr[idx * y_stride];        \
            }                                                             \
          });                                                             \
    });                                                                   \
  }

DEF_UNARY_RING_OP(ring_not, ~);
DEF_UNARY_RING_OP(ring_neg, -);

DEF_BINARY_RING_OP(ring_add, +);
DEF_BINARY_RING_OP(ring_sub, -);
DEF_BINARY_RING_OP(ring_mul, *);
DEF_BINARY_RING_OP(ring_and, &);
DEF_BINARY_RING_OP(ring_xor, ^);
DEF_BINARY_RING_OP(ring_equal, ==);

void ring_arshift_impl(ArrayRef& ret, const ArrayRef& x, size_t bits) {
  ENFORCE_EQ_ELSIZE_AND_NUMEL(ret, x);
  const auto field = x.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, kModule, [&]() {
    // According to K&R 2nd edition the results are implementation-dependent for
    // right shifts of signed values, but "usually" its arithmetic right shift.
    using S = std::make_signed<ring2k_t>::type;

    const auto* x_itr = &x.at<S>(0);
    const int64_t x_stride = x.stride();
    auto* z_itr = &ret.at<S>(0);
    const int64_t z_stride = ret.stride();
    const auto numel = ret.numel();

    yasl::parallel_for(
        0, numel, PFOR_GRAIN_SIZE, [&](int64_t begin, int64_t end) {
          for (int64_t idx = begin; idx < end; ++idx) {
            z_itr[idx * z_stride] = x_itr[idx * x_stride] >> bits;
          }
        });
  });
}

void ring_rshift_impl(ArrayRef& ret, const ArrayRef& x, size_t bits) {
  ENFORCE_EQ_ELSIZE_AND_NUMEL(ret, x);
  const auto field = x.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, kModule, [&]() {
    using U = std::make_unsigned<ring2k_t>::type;

    const auto* x_itr = &x.at<U>(0);
    const int64_t x_stride = x.stride();
    auto* z_itr = &ret.at<U>(0);
    const int64_t z_stride = ret.stride();
    const auto numel = ret.numel();

    yasl::parallel_for(
        0, numel, PFOR_GRAIN_SIZE, [&](int64_t begin, int64_t end) {
          for (int64_t idx = begin; idx < end; ++idx) {
            z_itr[idx * z_stride] = x_itr[idx * x_stride] >> bits;
          }
        });
  });
}

void ring_lshift_impl(ArrayRef& ret, const ArrayRef& x, size_t bits) {
  ENFORCE_EQ_ELSIZE_AND_NUMEL(ret, x);
  const auto field = x.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, kModule, [&]() {
    const auto* x_itr = &x.at<ring2k_t>(0);
    const int64_t x_stride = x.stride();
    auto* z_itr = &ret.at<ring2k_t>(0);
    const int64_t z_stride = ret.stride();
    const auto numel = ret.numel();

    yasl::parallel_for(
        0, numel, PFOR_GRAIN_SIZE, [&](int64_t begin, int64_t end) {
          for (int64_t idx = begin; idx < numel; ++idx) {
            z_itr[idx * z_stride] = x_itr[idx * x_stride] << bits;
          }
        });
  });
}

void ring_bitrev_impl(ArrayRef& ret, const ArrayRef& x, size_t start,
                      size_t end) {
  ENFORCE_EQ_ELSIZE_AND_NUMEL(ret, x);

  const auto field = x.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, kModule, [&]() {
    using U = typename std::make_unsigned<ring2k_t>::type;
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

    const auto* x_itr = &x.at<U>(0);
    const int64_t x_stride = x.stride();
    auto* z_itr = &ret.at<U>(0);
    const int64_t z_stride = ret.stride();
    const auto numel = ret.numel();

    yasl::parallel_for(
        0, numel, PFOR_GRAIN_SIZE, [&](int64_t start_idx, int64_t end_idx) {
          for (int64_t idx = start_idx; idx < end_idx; ++idx) {
            z_itr[idx * z_stride] = bitrev_fn(x_itr[idx * x_stride]);
          }
        });
  });
}

}  // namespace

// debug only
void ring_print(const ArrayRef& x, std::string_view name) {
  YASL_ENFORCE_RING(x);

  const auto field = x.eltype().as<Ring2k>()->field();
  DISPATCH_ALL_FIELDS(field, kModule, [&]() {
    using U = std::make_unsigned<ring2k_t>::type;

    const U* x_itr = &x.at<U>(0);
    const int64_t x_stride = x.stride();

    fmt::print("{} = {{", name);
    for (int64_t idx = 0; idx < x.numel(); idx++) {
      if (idx != 0) {
        fmt::print(", {0:X}", x_itr[idx * x_stride]);
      } else {
        fmt::print("{0:X}", x_itr[idx * x_stride]);
      }
    }
    fmt::print("}}\n");
  });
}

ArrayRef ring_rand(FieldType field, size_t size) {
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<uint64_t> dis;
  uint64_t cnt = dis(gen);
  return ring_rand(field, size, 0, &cnt);
}

ArrayRef ring_rand(FieldType field, size_t size, uint128_t prg_seed,
                   uint64_t* prg_counter) {
  constexpr yasl::SymmetricCrypto::CryptoType kCryptoType =
      yasl::SymmetricCrypto::CryptoType::AES128_ECB;
  constexpr uint128_t kAesInitialVector = 0U;

  ArrayRef res(makeType<RingTy>(field), size);
  *prg_counter = yasl::FillPseudoRandom(
      kCryptoType, prg_seed, kAesInitialVector, *prg_counter,
      absl::MakeSpan(static_cast<char*>(res.data()), res.buf()->size()));

  return res;
}

ArrayRef ring_rand_range(FieldType field, size_t size, int32_t min,
                         int32_t max) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int32_t> dis(min, max);

  ArrayRef x(makeType<RingTy>(field), size);

  DISPATCH_ALL_FIELDS(field, kModule, [&]() {
    YASL_ENFORCE(sizeof(ring2k_t) >= sizeof(int32_t));
    auto* x_itr = &x.at<ring2k_t>(0);
    const auto x_stride = x.stride();

    for (auto idx = 0; idx < x.numel(); idx++) {
      x_itr[idx * x_stride] = static_cast<ring2k_t>(dis(gen));
    }
  });

  return x;
}

void ring_assign(ArrayRef& x, const ArrayRef& y) {
  YASL_ENFORCE_RING(x);
  YASL_ENFORCE(x.numel() == y.numel());
  YASL_ENFORCE(x.elsize() == y.elsize());

  const int64_t elsize = x.elsize();
  const int64_t numel = x.numel();
  strided_copy(numel, elsize, x.data(), x.stride(), y.data(), y.stride());

  // WARN: the following method does not work
  // due to https://github.com/xtensor-stack/xtensor/issues/2330
  // return DISPATCH_ALL_FIELDS(field, kModule, [&]() {
  //  auto _x = xt_mutable_adapt<ring2k_t>(x);
  //  const auto& _y = xt_adapt<ring2k_t>(y);
  //  _x = _y;
  //});
}

ArrayRef ring_zeros(FieldType field, size_t size) {
  // TODO(jint) zero strides.
  ArrayRef res(makeType<RingTy>(field), size);
  std::memset(res.data(), 0, res.buf()->size());
  return res;
}

ArrayRef ring_ones(FieldType field, size_t size) {
  ArrayRef res = ring_zeros(field, size);

  return DISPATCH_ALL_FIELDS(field, kModule, [&]() {
    ArrayRef ret(makeType<RingTy>(field), size);
    auto* x_itr = &ret.at<ring2k_t>(0);
    const int64_t x_stride = ret.stride();

    yasl::parallel_for(0, size, PFOR_GRAIN_SIZE, [&](size_t begin, size_t end) {
      for (size_t idx = begin; idx < end; ++idx) {
        x_itr[idx * x_stride] = 1;
      }
    });
    return ret;
  });
}

ArrayRef ring_randbit(FieldType field, size_t size) {
  return DISPATCH_ALL_FIELDS(field, kModule, [&]() {
    ArrayRef ret(makeType<RingTy>(field), size);
    auto* x_itr = &ret.at<ring2k_t>(0);
    const int64_t x_stride = ret.stride();
    auto engine = std::default_random_engine(std::random_device{}());
    for (size_t idx = 0; idx < size; idx++) {
      x_itr[idx * x_stride] = engine() & 0x1;
    }
    return ret;
  });
}

ArrayRef ring_not(const ArrayRef& x) {
  ArrayRef ret(x.eltype(), x.numel());
  ring_not_impl(ret, x);
  return ret;
}

void ring_not_(ArrayRef& x) { ring_not_impl(x, x); }

ArrayRef ring_neg(const ArrayRef& x) {
  ArrayRef res(x.eltype(), x.numel());
  ring_neg_impl(res, x);
  return res;
}

void ring_neg_(ArrayRef& x) { ring_neg_impl(x, x); }

ArrayRef ring_add(const ArrayRef& x, const ArrayRef& y) {
  ArrayRef res(x.eltype(), x.numel());
  ring_add_impl(res, x, y);
  return res;
}

void ring_add_(ArrayRef& x, const ArrayRef& y) { ring_add_impl(x, x, y); }

ArrayRef ring_sub(const ArrayRef& x, const ArrayRef& y) {
  ArrayRef res(x.eltype(), x.numel());
  ring_sub_impl(res, x, y);
  return res;
}

void ring_sub_(ArrayRef& x, const ArrayRef& y) { ring_sub_impl(x, x, y); }

ArrayRef ring_mul(const ArrayRef& x, const ArrayRef& y) {
  ArrayRef res(x.eltype(), x.numel());
  ring_mul_impl(res, x, y);
  return res;
}

void ring_mul_(ArrayRef& x, const ArrayRef& y) { ring_mul_impl(x, x, y); }

ArrayRef ring_mmul(const ArrayRef& lhs, const ArrayRef& rhs, size_t M, size_t N,
                   size_t K) {
  YASL_ENFORCE(lhs.eltype().isa<Ring2k>(), "lhs not ring, got={}",
               lhs.eltype());
  YASL_ENFORCE(rhs.eltype().isa<Ring2k>(), "rhs not ring, got={}",
               rhs.eltype());
  YASL_ENFORCE(static_cast<size_t>(lhs.numel()) >= M * K);
  YASL_ENFORCE(static_cast<size_t>(rhs.numel()) >= K * N);

  const auto field = lhs.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, kModule, [&]() {
    ArrayRef ret(lhs.eltype(), M * N);
    const auto& lhs_strides = lhs.stride();
    const auto lhs_stride_scale = lhs.elsize() / sizeof(ring2k_t);
    const auto& rhs_strides = rhs.stride();
    const auto rhs_stride_scale = rhs.elsize() / sizeof(ring2k_t);
    const auto& ret_strides = ret.stride();
    const auto ret_stride_scale = ret.elsize() / sizeof(ring2k_t);

    linalg::matmul(
        M, N, K, static_cast<const ring2k_t*>(lhs.data()),
        lhs_stride_scale * K * lhs_strides, lhs_stride_scale * lhs_strides,
        static_cast<const ring2k_t*>(rhs.data()),
        rhs_stride_scale * N * rhs_strides, rhs_stride_scale * rhs_strides,
        static_cast<ring2k_t*>(ret.data()), ret_stride_scale * N * ret_strides,
        ret_stride_scale * ret_strides);
    return ret;
  });
}

ArrayRef ring_and(const ArrayRef& x, const ArrayRef& y) {
  ArrayRef res(x.eltype(), x.numel());
  ring_and_impl(res, x, y);
  return res;
}

void ring_and_(ArrayRef& x, const ArrayRef& y) { ring_and_impl(x, x, y); }

ArrayRef ring_xor(const ArrayRef& x, const ArrayRef& y) {
  ArrayRef res(x.eltype(), x.numel());
  ring_xor_impl(res, x, y);
  return res;
}

void ring_xor_(ArrayRef& x, const ArrayRef& y) { ring_xor_impl(x, x, y); }

ArrayRef ring_equal(const ArrayRef& x, const ArrayRef& y) {
  ArrayRef res(x.eltype(), x.numel());
  ring_equal_impl(res, x, y);
  return res;
}

void ring_equal_(ArrayRef& x, const ArrayRef& y) { ring_equal_impl(x, x, y); }

ArrayRef ring_arshift(const ArrayRef& x, size_t bits) {
  ArrayRef res(x.eltype(), x.numel());
  ring_arshift_impl(res, x, bits);
  return res;
}

void ring_arshift_(ArrayRef& x, size_t bits) { ring_arshift_impl(x, x, bits); }

ArrayRef ring_rshift(const ArrayRef& x, size_t bits) {
  ArrayRef res(x.eltype(), x.numel());
  ring_rshift_impl(res, x, bits);
  return res;
}

void ring_rshift_(ArrayRef& x, size_t bits) { ring_rshift_impl(x, x, bits); }

ArrayRef ring_lshift(const ArrayRef& x, size_t bits) {
  ArrayRef res(x.eltype(), x.numel());
  ring_lshift_impl(res, x, bits);
  return res;
}

void ring_lshift_(ArrayRef& x, size_t bits) { ring_lshift_impl(x, x, bits); }

ArrayRef ring_bitrev(const ArrayRef& x, size_t start, size_t end) {
  ArrayRef res(x.eltype(), x.numel());
  ring_bitrev_impl(res, x, start, end);
  return res;
}

void ring_bitrev_(ArrayRef& x, size_t start, size_t end) {
  ring_bitrev_impl(x, x, start, end);
}

ArrayRef ring_sum(absl::Span<ArrayRef const> arrs) {
  YASL_ENFORCE(!arrs.empty(), "expected non empty, got size={}", arrs.size());

  if (arrs.size() == 1) {
    return arrs[0];
  }

  YASL_ENFORCE(arrs.size() >= 2);
  auto res = ring_add(arrs[0], arrs[1]);
  for (size_t idx = 2; idx < arrs.size(); idx++) {
    ring_add_(res, arrs[idx]);
  }
  return res;
}

bool ring_all_equal(const ArrayRef& x, const ArrayRef& y, size_t abs_err) {
  ENFORCE_EQ_ELSIZE_AND_NUMEL(x, y);
  YASL_ENFORCE(x.numel() == y.numel());

  const auto field = x.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    const auto* x_itr = &x.at<ring2k_t>(0);
    const int64_t x_stride = x.stride();
    auto* y_itr = &y.at<ring2k_t>(0);
    const int64_t y_stride = y.stride();
    for (int64_t idx = 0; idx < x.numel(); idx++) {
      auto x_el = x_itr[idx * x_stride];
      auto y_el = y_itr[idx * y_stride];
      if (std::abs(y_el - x_el) > static_cast<ring2k_t>(abs_err)) {
        fmt::print("error: {0:X} {1:X}\n", x_el, y_el);
        return false;
      }
    }
    return true;
  });
}

std::vector<bool> ring_as_bool(const ArrayRef& x) {
  YASL_ENFORCE_RING(x);
  const auto field = x.eltype().as<Ring2k>()->field();

  std::vector<bool> res(x.numel());

  // Assign boolean vector in a parallel fashion is not safe, so this loop
  // cannot parallelized
  DISPATCH_ALL_FIELDS(field, kModule, [&]() {
    const auto* x_itr = &x.at<ring2k_t>(0);
    const int64_t x_stride = x.stride();
    for (int64_t idx = 0; idx < x.numel(); idx++) {
      auto x_el = x_itr[idx * x_stride];
      YASL_ENFORCE(x_el == 0 || x_el == 1);
      res[idx] = (x_el == 1);
    }
  });

  return res;
}

ArrayRef ring_select(const std::vector<uint8_t>& c, const ArrayRef& x,
                     const ArrayRef& y) {
  ENFORCE_EQ_ELSIZE_AND_NUMEL(x, y);
  YASL_ENFORCE(x.numel() == y.numel());
  YASL_ENFORCE(x.numel() == static_cast<int64_t>(c.size()));

  const auto field = x.eltype().as<Ring2k>()->field();
  ArrayRef z(x.eltype(), x.numel());

  DISPATCH_ALL_FIELDS(field, kModule, [&]() {
    const ring2k_t* x_ptr = &x.at<ring2k_t>(0);
    const ring2k_t* y_ptr = &y.at<ring2k_t>(0);
    ring2k_t* z_ptr = &z.at<ring2k_t>(0);

    const auto x_stride = x.stride();
    const auto y_stride = y.stride();
    const auto z_stride = z.stride();
    const auto numel = c.size();

    yasl::parallel_for(
        0, numel, PFOR_GRAIN_SIZE, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; i++) {
            z_ptr[i * z_stride] =
                (c[i] ? y_ptr[i * y_stride] : x_ptr[i * x_stride]);
          }
        });
  });

  return z;
}

std::vector<ArrayRef> ring_rand_splits(const ArrayRef& arr, size_t num_splits) {
  YASL_ENFORCE(num_splits > 1, "num split be greater than 1 ", num_splits);

  const auto field = arr.eltype().as<Ring2k>()->field();

  std::vector<ArrayRef> splits;
  for (size_t idx = 0; idx < num_splits; idx++) {
    splits.push_back(ring_rand(field, arr.numel()));
  }

  // fix the first random splits.
  ArrayRef s = ring_sum(splits);
  splits[0] = ring_add(splits[0], ring_sub(arr, s));

  return splits;
}

}  // namespace spu::mpc
