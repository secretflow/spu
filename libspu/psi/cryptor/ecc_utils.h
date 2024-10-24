// Copyright 2022 Ant Group Co., Ltd.
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

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "openssl/bn.h"
#include "openssl/ec.h"
#include "yacl/crypto/base/hash/hash_utils.h"
#include "yacl/utils/parallel.h"

#include "libspu/core/prelude.h"

namespace spu::psi {

inline constexpr size_t kEcPointCompressLength = 33;
inline constexpr size_t kEc256KeyLength = 32;

// Deletes a BN_CTX.
struct BnCtxDeleter {
 public:
  void operator()(BN_CTX* bn_ctx) { BN_CTX_free(bn_ctx); }
};
using BnCtxPtr = std::unique_ptr<BN_CTX, BnCtxDeleter>;

// Deletes a BIGNUM.
struct BnDeleter {
 public:
  void operator()(BIGNUM* bn) { BN_clear_free(bn); }
};
using BigNumPtr = std::unique_ptr<BIGNUM, BnDeleter>;

// Deletes a EC_GROUP.
struct ECGroupDeleter {
 public:
  void operator()(EC_GROUP* group) { EC_GROUP_free(group); }
};
using ECGroupPtr = std::unique_ptr<EC_GROUP, ECGroupDeleter>;

// Deletes an EC_POINT.
struct ECPointDeleter {
 public:
  void operator()(EC_POINT* point) { EC_POINT_clear_free(point); }
};
using ECPointPtr = std::unique_ptr<EC_POINT, ECPointDeleter>;

struct BigNumSt {
  BigNumSt() : bn_ptr(BN_new()) {}

  explicit BigNumSt(absl::string_view bytes) : bn_ptr(BN_new()) {
    FromBytes(bytes);
  }

  explicit BigNumSt(absl::Span<const uint8_t> bytes) : bn_ptr(BN_new()) {
    FromBytes(bytes);
  }

  BIGNUM* get() { return bn_ptr.get(); }

  const BIGNUM* get() const { return bn_ptr.get(); }

  std::string ToBytes() {
    std::string bytes(kEc256KeyLength, '\0');

    BN_bn2binpad(bn_ptr.get(), reinterpret_cast<uint8_t*>(bytes.data()),
                 kEc256KeyLength);

    return bytes;
  }

  void FromBytes(absl::Span<const uint8_t> bytes) {
    SPU_ENFORCE(nullptr != BN_bin2bn(bytes.data(), bytes.size(), bn_ptr.get()));
  }

  void FromBytes(absl::string_view bytes) {
    SPU_ENFORCE(nullptr !=
                BN_bin2bn(reinterpret_cast<const uint8_t*>(bytes.data()),
                          bytes.size(), bn_ptr.get()));
  }

  void FromBytes(absl::Span<const uint8_t> bytes, const BigNumSt& p) {
    BigNumSt bn_m(bytes);
    BnCtxPtr bn_ctx(yacl::CheckNotNull(BN_CTX_new()));

    SPU_ENFORCE(BN_nnmod(bn_ptr.get(), bn_m.get(), p.get(), bn_ctx.get()) == 1);
  }

  void FromBytes(absl::string_view bytes, const BigNumSt& p) {
    BigNumSt bn_m(bytes);
    BnCtxPtr bn_ctx(yacl::CheckNotNull(BN_CTX_new()));

    SPU_ENFORCE(BN_nnmod(bn_ptr.get(), bn_m.get(), p.get(), bn_ctx.get()) == 1);
  }

  BigNumSt Inverse(const BigNumSt& p) {
    BnCtxPtr bn_ctx(yacl::CheckNotNull(BN_CTX_new()));

    BigNumSt bn_inv;

    BN_mod_inverse(bn_inv.get(), bn_ptr.get(), p.get(), bn_ctx.get());
    return bn_inv;
  }

  BigNumPtr bn_ptr;
};

struct EcGroupSt {
  explicit EcGroupSt(int ec_group_nid = NID_sm2)
      : EcGroupSt(EC_GROUP_new_by_curve_name(ec_group_nid)) {}

  explicit EcGroupSt(EC_GROUP* group) : group_ptr(group) {
    BnCtxPtr bn_ctx(yacl::CheckNotNull(BN_CTX_new()));

    SPU_ENFORCE(EC_GROUP_get_curve(group_ptr.get(), bn_p.get(), bn_a.get(),
                                   bn_b.get(), bn_ctx.get()) == 1);
    SPU_ENFORCE(EC_GROUP_get_order(group_ptr.get(), bn_n.get(), bn_ctx.get()) ==
                1);
  }

  EC_GROUP* get() { return group_ptr.get(); }

  const EC_GROUP* get() const { return group_ptr.get(); }

  BigNumSt bn_p;

  BigNumSt bn_a;
  BigNumSt bn_b;
  BigNumSt bn_n;

  ECGroupPtr group_ptr;
};

inline constexpr size_t kHashToCurveCounterGuard = 100;

struct EcPointSt {
  explicit EcPointSt(const EcGroupSt& group)
      : group_ref(group), point_ptr(EC_POINT_new(group_ref.get())) {}

  //
  // https://eprint.iacr.org/2009/226.pdf
  // 1.1 Try-and-Increment Method
  //
  static EcPointSt CreateEcPointByHashToCurve(absl::string_view m,
                                              const EcGroupSt& ec_group) {
    BnCtxPtr bn_ctx(yacl::CheckNotNull(BN_CTX_new()));

    EcPointSt ec_point(ec_group);

    BigNumSt bn_x;

    bn_x.FromBytes(yacl::crypto::Sha256(m), ec_group.bn_p);

    size_t counter = 0;

    while (true) {
      int ret = EC_POINT_set_compressed_coordinates(
          ec_group.get(), ec_point.get(), bn_x.get(), 0, bn_ctx.get());

      if (ret == 1) {
        break;
      } else {
        std::string bn_x_bytes = bn_x.ToBytes();

        // warn: if you change this hash method, do update bucket_psi.cc as well
        // std::vector<uint8_t> hash = yacl::crypto::Sm3(bn_x_bytes);
        const auto hash = yacl::crypto::Sha256(bn_x_bytes);
        bn_x.FromBytes(
            absl::string_view(reinterpret_cast<const char*>(hash.data()),
                              hash.size()),
            ec_group.bn_p);
      }
      counter++;
      SPU_ENFORCE(counter < kHashToCurveCounterGuard,
                  "HashToCurve exceed max loop({})", kHashToCurveCounterGuard);
    }

    return ec_point;
  }

  static EcPointSt CreateEcPointByHashToCurve(absl::Span<const uint8_t> m,
                                              const EcGroupSt& ec_group) {
    return CreateEcPointByHashToCurve(
        absl::string_view(reinterpret_cast<const char*>(m.data()), m.size()),
        ec_group);
  }

  EC_POINT* get() { return point_ptr.get(); }
  const EC_POINT* get() const { return point_ptr.get(); }
  const EcGroupSt& get_group() { return group_ref; }

  /**
   * @brief EC_POINT to bytes, compressed form
   *
   * @param bytes
   * @return size_t
   */
  size_t ToBytes(absl::Span<uint8_t> bytes) {
    BnCtxPtr bn_ctx(yacl::CheckNotNull(BN_CTX_new()));
    size_t length = EC_POINT_point2oct(group_ref.get(), point_ptr.get(),
                                       POINT_CONVERSION_COMPRESSED, nullptr, 0,
                                       bn_ctx.get());

    SPU_ENFORCE(length == kEcPointCompressLength, "{}!={}", length,
                kEcPointCompressLength);

    std::vector<uint8_t> point_compress_bytes(length);
    length = EC_POINT_point2oct(
        group_ref.get(), point_ptr.get(), POINT_CONVERSION_COMPRESSED,
        reinterpret_cast<uint8_t*>(point_compress_bytes.data()),
        point_compress_bytes.size(), bn_ctx.get());

    std::memcpy(bytes.data(), point_compress_bytes.data(), bytes.size());

    return length;
  }

  EcPointSt PointMul(const EcGroupSt& ec_group, const BigNumSt& bn_sk) {
    BnCtxPtr bn_ctx(yacl::CheckNotNull(BN_CTX_new()));
    EcPointSt ec_point(ec_group);
    int ret = EC_POINT_mul(ec_group.get(), ec_point.get(), nullptr,
                           point_ptr.get(), bn_sk.get(), bn_ctx.get());
    SPU_ENFORCE(ret == 1);

    return ec_point;
  }

  static EcPointSt BasePointMul(const EcGroupSt& group, const BigNumSt& bn_sk) {
    BnCtxPtr bn_ctx(yacl::CheckNotNull(BN_CTX_new()));
    EcPointSt ec_point(group);

    int ret = EC_POINT_mul(group.get(), ec_point.get(), bn_sk.get(), nullptr,
                           nullptr, bn_ctx.get());

    SPU_ENFORCE(ret == 1);

    return ec_point;
  }

  const EcGroupSt& group_ref;

  ECPointPtr point_ptr;
};

}  // namespace spu::psi
