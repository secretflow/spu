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

#define EIGEN_HAS_OPENMP
#include "Eigen/Core"
#include "absl/types/span.h"
#include "yacl/crypto/tools/prg.h"
#include "yacl/crypto/utils/rand.h"
#include "yacl/utils/parallel.h"

#include "libspu/core/array_ref.h"
#include "libspu/mpc/utils/linalg.h"

// TODO: ArrayRef is simple enough, consider using other SIMD libraries.
namespace spu::mpc {
namespace {

constexpr char kModule[] = "RingOps";

#define SPU_ENFORCE_RING(x)                                           \
  SPU_ENFORCE((x).eltype().isa<Ring2k>(), "expect ring type, got={}", \
              (x).eltype());

#define ENFORCE_EQ_ELSIZE_AND_NUMEL(lhs, rhs)                                  \
  SPU_ENFORCE((lhs).eltype().as<Ring2k>()->field() ==                          \
                  (rhs).eltype().as<Ring2k>()->field(),                        \
              "type mismatch lhs={}, rhs={}", (lhs).eltype(), (rhs).eltype()); \
  SPU_ENFORCE((lhs).numel() == (rhs).numel(),                                  \
              "numel mismatch, lhs={}, rhs={}", (lhs).numel(), (rhs).numel());

#define DEF_UNARY_RING_OP_EIGEN(NAME, FNAME)                              \
  void NAME##_impl(ArrayRef& ret, const ArrayRef& x) {                    \
    ENFORCE_EQ_ELSIZE_AND_NUMEL(ret, x);                                  \
    const auto field = x.eltype().as<Ring2k>()->field();                  \
    const int64_t numel = ret.numel();                                    \
    return DISPATCH_ALL_FIELDS(field, kModule, [&]() {                    \
      using T = std::make_signed_t<ring2k_t>;                             \
      FNAME(numel, &x.at<T>(0), x.stride(), &ret.at<T>(0), ret.stride()); \
    });                                                                   \
  }

#define DEF_BINARY_RING_OP_EIGEN(NAME, FNAME)                             \
  void NAME##_impl(ArrayRef& ret, const ArrayRef& x, const ArrayRef& y) { \
    ENFORCE_EQ_ELSIZE_AND_NUMEL(ret, x);                                  \
    ENFORCE_EQ_ELSIZE_AND_NUMEL(ret, y);                                  \
    const auto field = x.eltype().as<Ring2k>()->field();                  \
    const int64_t numel = ret.numel();                                    \
    return DISPATCH_ALL_FIELDS(field, kModule, [&]() {                    \
      FNAME(numel, &x.at<ring2k_t>(0), x.stride(), &y.at<ring2k_t>(0),    \
            y.stride(), &ret.at<ring2k_t>(0), ret.stride());              \
    });                                                                   \
  }

DEF_UNARY_RING_OP_EIGEN(ring_not, linalg::bitwise_not);
DEF_UNARY_RING_OP_EIGEN(ring_neg, linalg::negate);

DEF_BINARY_RING_OP_EIGEN(ring_add, linalg::add)
DEF_BINARY_RING_OP_EIGEN(ring_sub, linalg::sub)
DEF_BINARY_RING_OP_EIGEN(ring_mul, linalg::mul)
DEF_BINARY_RING_OP_EIGEN(ring_equal, linalg::equal)

DEF_BINARY_RING_OP_EIGEN(ring_and, linalg::bitwise_and);
DEF_BINARY_RING_OP_EIGEN(ring_xor, linalg::bitwise_xor);

void ring_arshift_impl(ArrayRef& ret, const ArrayRef& x, size_t bits) {
  ENFORCE_EQ_ELSIZE_AND_NUMEL(ret, x);
  const auto numel = ret.numel();
  const auto field = x.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, kModule, [&]() {
    // According to K&R 2nd edition the results are implementation-dependent for
    // right shifts of signed values, but "usually" its arithmetic right shift.
    using S = std::make_signed<ring2k_t>::type;

    linalg::rshift(numel, &x.at<S>(0), x.stride(), &ret.at<S>(0), ret.stride(),
                   bits);
  });
}

void ring_rshift_impl(ArrayRef& ret, const ArrayRef& x, size_t bits) {
  ENFORCE_EQ_ELSIZE_AND_NUMEL(ret, x);
  const auto numel = ret.numel();
  const auto field = x.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, kModule, [&]() {
    using U = ring2k_t;

    linalg::rshift(numel, &x.at<U>(0), x.stride(), &ret.at<U>(0), ret.stride(),
                   bits);
  });
}

void ring_lshift_impl(ArrayRef& ret, const ArrayRef& x, size_t bits) {
  ENFORCE_EQ_ELSIZE_AND_NUMEL(ret, x);
  const auto numel = ret.numel();
  const auto field = x.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, kModule, [&]() {
    linalg::lshift(numel, &x.at<ring2k_t>(0), x.stride(), &ret.at<ring2k_t>(0),
                   ret.stride(), bits);
  });
}

void ring_bitrev_impl(ArrayRef& ret, const ArrayRef& x, size_t start,
                      size_t end) {
  ENFORCE_EQ_ELSIZE_AND_NUMEL(ret, x);

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

    linalg::unaryWithOp(numel, &x.at<U>(0), x.stride(), &ret.at<U>(0),
                        ret.stride(), bitrev_fn);
  });
}

void ring_bitmask_impl(ArrayRef& ret, const ArrayRef& x, size_t low,
                       size_t high) {
  ENFORCE_EQ_ELSIZE_AND_NUMEL(ret, x);

  const auto field = x.eltype().as<Ring2k>()->field();
  const auto numel = ret.numel();

  SPU_ENFORCE(low < high && high <= SizeOf(field) * 8);

  return DISPATCH_ALL_FIELDS(field, kModule, [&]() {
    using U = ring2k_t;
    U mask = (((U)1U << (high - low)) - 1) << low;

    auto mark_fn = [&](U el) { return el & mask; };

    linalg::unaryWithOp(numel, &x.at<U>(0), x.stride(), &ret.at<U>(0),
                        ret.stride(), mark_fn);
  });
}

}  // namespace

// debug only
void ring_print(const ArrayRef& x, std::string_view name) {
  SPU_ENFORCE_RING(x);

  const auto field = x.eltype().as<Ring2k>()->field();
  DISPATCH_ALL_FIELDS(field, kModule, [&]() {
    using U = ring2k_t;

    auto x_eigen = Eigen::Map<const Eigen::ArrayX<U>, 0,
                              Eigen::InnerStride<Eigen::Dynamic>>(
        &x.at<U>(0), x.numel(), Eigen::InnerStride<Eigen::Dynamic>(x.stride()));

    fmt::print("{} = {{", name);
    for (int64_t idx = 0; idx < x.numel(); idx++) {
      if (idx != 0) {
        fmt::print(", {0:X}", x_eigen[idx]);
      } else {
        fmt::print("{0:X}", x_eigen[idx]);
      }
    }
    fmt::print("}}\n");
  });
}

ArrayRef ring_rand(FieldType field, size_t size) {
  uint64_t cnt = 0;
  return ring_rand(field, size, yacl::crypto::RandSeed(), &cnt);
}

ArrayRef ring_rand(FieldType field, size_t size, uint128_t prg_seed,
                   uint64_t* prg_counter) {
  constexpr yacl::crypto::SymmetricCrypto::CryptoType kCryptoType =
      yacl::crypto::SymmetricCrypto::CryptoType::AES128_CTR;
  constexpr uint128_t kAesInitialVector = 0U;

  ArrayRef res(makeType<RingTy>(field), size);
  *prg_counter = yacl::crypto::FillPRand(
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
    SPU_ENFORCE(sizeof(ring2k_t) >= sizeof(int32_t));
    auto x_eigen = Eigen::Map<Eigen::ArrayX<ring2k_t>, 0,
                              Eigen::InnerStride<Eigen::Dynamic>>(
        &x.at<ring2k_t>(0), x.numel(),
        Eigen::InnerStride<Eigen::Dynamic>(x.stride()));

    for (auto idx = 0; idx < x.numel(); idx++) {
      x_eigen[idx] = static_cast<ring2k_t>(dis(gen));
    }
  });

  return x;
}

void ring_assign(ArrayRef& x, const ArrayRef& y) {
  SPU_ENFORCE_RING(x);
  ENFORCE_EQ_ELSIZE_AND_NUMEL(x, y);

  const auto numel = x.numel();

  const auto field = x.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, kModule, [&]() {
    linalg::assign(numel, &y.at<ring2k_t>(0), y.stride(), &x.at<ring2k_t>(0),
                   x.stride());
  });
}

ArrayRef ring_zeros(FieldType field, size_t size) {
  return DISPATCH_ALL_FIELDS(field, kModule, [&]() {
    ArrayRef ret(makeType<RingTy>(field), size);
    linalg::setConstantValue(ret.numel(), &ret.at<ring2k_t>(0), ret.stride(),
                             ring2k_t(0));
    return ret;
  });
}

ArrayRef ring_zeros_packed(FieldType field, size_t size) {
  return DISPATCH_ALL_FIELDS(field, kModule, [&]() {
    auto ty = makeType<RingTy>(field);
    ArrayRef ret = makeConstantArrayRef(ty, size);
    std::memset(ret.data(), ring2k_t(0), ty.size());
    return ret;
  });
}

ArrayRef ring_ones(FieldType field, size_t size) {
  return DISPATCH_ALL_FIELDS(field, kModule, [&]() {
    ArrayRef ret(makeType<RingTy>(field), size);
    linalg::setConstantValue(ret.numel(), &ret.at<ring2k_t>(0), ret.stride(),
                             ring2k_t(1));
    return ret;
  });
}

ArrayRef ring_randbit(FieldType field, size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(0, RAND_MAX);

  return DISPATCH_ALL_FIELDS(field, kModule, [&]() {
    ArrayRef ret(makeType<RingTy>(field), size);
    auto x_eigen = Eigen::Map<Eigen::VectorX<ring2k_t>, 0,
                              Eigen::InnerStride<Eigen::Dynamic>>(
        &ret.at<ring2k_t>(0), ret.numel(),
        Eigen::InnerStride<Eigen::Dynamic>(ret.stride()));
    for (size_t idx = 0; idx < size; idx++) {
      x_eigen[idx] = distrib(gen) & 0x1;
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

void ring_mul_impl(ArrayRef& ret, const ArrayRef& x, uint128_t y) {
  ENFORCE_EQ_ELSIZE_AND_NUMEL(ret, x);

  const auto numel = x.numel();
  const auto field = x.eltype().as<Ring2k>()->field();
  DISPATCH_ALL_FIELDS(field, kModule, [&]() {
    using U = std::make_unsigned<ring2k_t>::type;

    auto x_data = ArrayView<U>(x);
    auto ret_data = ArrayView<U>(ret);
    yacl::parallel_for(0, numel, PFOR_GRAIN_SIZE,
                       [&](int64_t beg, int64_t end) {
                         for (int64_t idx = beg; idx < end; ++idx) {
                           ret_data[idx] = x_data[idx] * y;
                         }
                       });
  });
}

ArrayRef ring_mul(const ArrayRef& x, uint128_t y) {
  ArrayRef res(x.eltype(), x.numel());
  ring_mul_impl(res, x, y);
  return res;
}

void ring_mul_(ArrayRef& x, uint128_t y) { ring_mul_impl(x, x, y); }

void ring_mmul_(ArrayRef& z, const ArrayRef& lhs, const ArrayRef& rhs, size_t M,
                size_t N, size_t K) {
  SPU_ENFORCE(lhs.eltype().isa<Ring2k>(), "lhs not ring, got={}", lhs.eltype());
  SPU_ENFORCE(rhs.eltype().isa<Ring2k>(), "rhs not ring, got={}", rhs.eltype());
  SPU_ENFORCE(static_cast<size_t>(lhs.numel()) >= M * K);
  SPU_ENFORCE(static_cast<size_t>(rhs.numel()) >= K * N);

  const auto field = lhs.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, kModule, [&]() {
    const auto& lhs_strides = lhs.stride();
    const auto lhs_stride_scale = lhs.elsize() / sizeof(ring2k_t);
    const auto& rhs_strides = rhs.stride();
    const auto rhs_stride_scale = rhs.elsize() / sizeof(ring2k_t);
    const auto& ret_strides = z.stride();
    const auto ret_stride_scale = z.elsize() / sizeof(ring2k_t);

    linalg::matmul(
        M, N, K, static_cast<const ring2k_t*>(lhs.data()),
        lhs_stride_scale * K * lhs_strides, lhs_stride_scale * lhs_strides,
        static_cast<const ring2k_t*>(rhs.data()),
        rhs_stride_scale * N * rhs_strides, rhs_stride_scale * rhs_strides,
        static_cast<ring2k_t*>(z.data()), ret_stride_scale * N * ret_strides,
        ret_stride_scale * ret_strides);
  });
}

ArrayRef ring_mmul(const ArrayRef& lhs, const ArrayRef& rhs, size_t m, size_t n,
                   size_t k) {
  ArrayRef ret(lhs.eltype(), m * n);
  ring_mmul_(ret, lhs, rhs, m, n, k);
  return ret;
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

void ring_equal(ArrayRef& x, const ArrayRef& y) { ring_equal_impl(x, x, y); }

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

ArrayRef ring_bitmask(const ArrayRef& x, size_t low, size_t high) {
  ArrayRef ret(x.eltype(), x.numel());
  ring_bitmask_impl(ret, x, low, high);
  return ret;
}

void ring_bitmask_(ArrayRef& x, size_t low, size_t high) {
  ring_bitmask_impl(x, x, low, high);
}

ArrayRef ring_sum(absl::Span<ArrayRef const> arrs) {
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

bool ring_all_equal(const ArrayRef& x, const ArrayRef& y, size_t abs_err) {
  ENFORCE_EQ_ELSIZE_AND_NUMEL(x, y);
  SPU_ENFORCE(x.numel() == y.numel());

  const auto field = x.eltype().as<Ring2k>()->field();
  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using T = std::make_signed_t<ring2k_t>;

    auto x_eigen = Eigen::Map<const Eigen::VectorX<T>, 0,
                              Eigen::InnerStride<Eigen::Dynamic>>(
        &x.at<T>(0), x.numel(), Eigen::InnerStride<Eigen::Dynamic>(x.stride()));
    auto y_eigen = Eigen::Map<const Eigen::VectorX<T>, 0,
                              Eigen::InnerStride<Eigen::Dynamic>>(
        &y.at<T>(0), y.numel(), Eigen::InnerStride<Eigen::Dynamic>(y.stride()));
    for (int64_t idx = 0; idx < x.numel(); idx++) {
      auto x_el = x_eigen[idx];
      auto y_el = y_eigen[idx];
      if (std::abs(x_el - y_el) > static_cast<T>(abs_err)) {
        fmt::print("error: {0:X} {1:X} abs_err: {2:X}\n", x_el, y_el, abs_err);
        return false;
      }
    }
    return true;
  });
}

std::vector<uint8_t> ring_cast_boolean(const ArrayRef& x) {
  SPU_ENFORCE_RING(x);
  const auto field = x.eltype().as<Ring2k>()->field();

  std::vector<uint8_t> res(x.numel());
  DISPATCH_ALL_FIELDS(field, kModule, [&]() {
    auto x_eigen = Eigen::Map<const Eigen::VectorX<ring2k_t>, 0,
                              Eigen::InnerStride<Eigen::Dynamic>>(
        &x.at<ring2k_t>(0), x.numel(),
        Eigen::InnerStride<Eigen::Dynamic>(x.stride()));
    yacl::parallel_for(0, x.numel(), PFOR_GRAIN_SIZE,
                       [&](size_t start, size_t end) {
                         for (size_t i = start; i < end; i++) {
                           res[i] = static_cast<uint8_t>(x_eigen[i] & 0x1);
                         }
                       });
  });

  return res;
}

ArrayRef ring_select(const std::vector<uint8_t>& c, const ArrayRef& x,
                     const ArrayRef& y) {
  ENFORCE_EQ_ELSIZE_AND_NUMEL(x, y);
  SPU_ENFORCE(x.numel() == y.numel());
  SPU_ENFORCE(x.numel() == static_cast<int64_t>(c.size()));

  const auto field = x.eltype().as<Ring2k>()->field();
  ArrayRef z(x.eltype(), x.numel());
  const int64_t numel = c.size();

  DISPATCH_ALL_FIELDS(field, kModule, [&]() {
    linalg::select(numel, c.data(), &y.at<ring2k_t>(0), y.stride(),
                   &x.at<ring2k_t>(0), x.stride(), &z.at<ring2k_t>(0),
                   z.stride());
  });

  return z;
}

std::vector<ArrayRef> ring_rand_additive_splits(const ArrayRef& arr,
                                                size_t num_splits) {
  const auto field = arr.eltype().as<Ring2k>()->field();
  SPU_ENFORCE(num_splits > 1, "num split {} be greater than 1 ", num_splits);

  std::vector<ArrayRef> splits(num_splits);
  splits[0] = arr.clone();

  for (size_t idx = 1; idx < num_splits; idx++) {
    splits[idx] = ring_rand(field, arr.numel());
    ring_sub_(splits[0], splits[idx]);
  }

  return splits;
}

std::vector<ArrayRef> ring_rand_boolean_splits(const ArrayRef& arr,
                                               size_t num_splits) {
  const auto field = arr.eltype().as<Ring2k>()->field();
  SPU_ENFORCE(num_splits > 1, "num split {} be greater than 1 ", num_splits);

  std::vector<ArrayRef> splits(num_splits);
  splits[0] = arr.clone();

  for (size_t idx = 1; idx < num_splits; idx++) {
    splits[idx] = ring_rand(field, arr.numel());
    ring_xor_(splits[0], splits[idx]);
  }

  return splits;
}

}  // namespace spu::mpc
