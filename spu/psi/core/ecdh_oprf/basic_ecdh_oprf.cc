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

#include "spu/psi/core/ecdh_oprf/basic_ecdh_oprf.h"

#include "absl/strings/escaping.h"
#include "yasl/crypto/blake3_hash.h"
#include "yasl/crypto/hash_util.h"

#include "spu/psi/cryptor/ecc_utils.h"

extern "C" {
#include "FourQ_api.h"
#include "FourQ_internal.h"
}

namespace spu::psi {

namespace {
// use 96bit as the final compare value
constexpr size_t kEc256CompareLength = 12;

std::string HashItem(absl::string_view item, absl::string_view masked_item,
                     size_t hash_len, yasl::crypto::HashAlgorithm hash_type) {
  std::unique_ptr<yasl::crypto::HashInterface> hash_algo;
  switch (hash_type) {
    case yasl::crypto::HashAlgorithm::BLAKE3:
      hash_algo = std::make_unique<yasl::crypto::Blake3Hash>();
      break;
    default:
      hash_algo = std::make_unique<yasl::crypto::SslHash>(hash_type);
      break;
  }

  hash_algo->Update(item);
  hash_algo->Update(masked_item);
  std::vector<uint8_t> hash = hash_algo->CumulativeHash();

  YASL_ENFORCE(hash_len <= hash.size());

  std::string hash_str(hash_len, '\0');
  std::memcpy(hash_str.data(), hash.data(), hash_len);

  return hash_str;
}

/**
 * @brief do ec ponit mul
 *
 * @param sk_bytes  private key data
 * @param point_bytes comporessed ec point data
 * @param ec_group_nid ec group nid
 * @return std::string comporessed ec point data
 */
std::string EcPointMul(absl::string_view sk_bytes,
                       absl::string_view point_bytes, int ec_group_nid) {
  BnCtxPtr bn_ctx(yasl::CheckNotNull(BN_CTX_new()));
  EcGroupSt ec_group(ec_group_nid);
  BigNumSt bn_sk;

  YASL_ENFORCE(sk_bytes.size() == kEccKeySize);
  bn_sk.FromBytes(
      absl::string_view((const char *)sk_bytes.data(), sk_bytes.length()),
      ec_group.bn_n);

  EcPointSt ec_point(ec_group);
  EC_POINT_oct2point(ec_group.get(), ec_point.get(),
                     (const uint8_t *)point_bytes.data(), point_bytes.length(),
                     bn_ctx.get());

  EcPointSt ec_point2 = ec_point.PointMul(ec_group, bn_sk);

  std::string masked_point_bytes(kEcPointCompressLength, '\0');
  ec_point2.ToBytes(absl::MakeSpan((uint8_t *)masked_point_bytes.data(),
                                   masked_point_bytes.size()));

  return masked_point_bytes;
}

/**
 * @brief do ec ponit mul
 *
 * @param sk_bytes  private key data
 * @param item_bytes input data, map to ec point internal
 * @param ec_group_nid ec group nid
 * @return std::string comporessed ec point data
 */
std::string ItemMul(absl::string_view sk_bytes, absl::string_view item_bytes,
                    int ec_group_nid) {
  BnCtxPtr bn_ctx(yasl::CheckNotNull(BN_CTX_new()));
  EcGroupSt ec_group(ec_group_nid);
  BigNumSt bn_sk;

  YASL_ENFORCE(sk_bytes.size() == kEccKeySize);
  bn_sk.FromBytes(
      absl::string_view((const char *)sk_bytes.data(), sk_bytes.length()),
      ec_group.bn_n);

  EcPointSt ec_point =
      EcPointSt::CreateEcPointByHashToCurve(item_bytes, ec_group);

  EcPointSt ec_point2 = ec_point.PointMul(ec_group, bn_sk);
  std::string point_bytes(kEcPointCompressLength, '\0');
  ec_point2.ToBytes(
      absl::MakeSpan((uint8_t *)point_bytes.data(), point_bytes.size()));

  return point_bytes;
}

}  // namespace

std::string BasicEcdhOprfServer::Evaluate(
    absl::string_view blinded_element) const {
  return EcPointMul(
      absl::string_view((const char *)&private_key_[0], kEccKeySize),
      blinded_element, ec_group_nid_);
}

std::string BasicEcdhOprfServer::FullEvaluate(
    yasl::ByteContainerView input) const {
  absl::string_view input_sv =
      absl::string_view((const char *)input.data(), input.size());
  std::string point_bytes =
      ItemMul(absl::string_view((const char *)&private_key_[0], kEccKeySize),
              input_sv, ec_group_nid_);

  return HashItem(input_sv, point_bytes, GetCompareLength(), hash_type_);
}

size_t BasicEcdhOprfServer::GetCompareLength() const {
  if (compare_length_) {
    return compare_length_;
  }

  return kEc256CompareLength;
}

size_t BasicEcdhOprfServer::GetEcPointLength() const {
  return kEcPointCompressLength;
}

BasicEcdhOprfClient::BasicEcdhOprfClient(CurveType type) : curve_type_(type) {
  ec_group_nid_ = Sm2Cryptor::GetEcGroupId(type);
  BnCtxPtr bn_ctx(yasl::CheckNotNull(BN_CTX_new()));
  EcGroupSt ec_group(ec_group_nid_);
  BigNumSt bn_sk;

  bn_sk.FromBytes(
      absl::string_view((const char *)&private_key_[0], kEccKeySize),
      ec_group.bn_n);

  BigNumSt bn_sk_inv = bn_sk.Inverse(ec_group.bn_n);

  std::string sk_inv = bn_sk_inv.ToBytes();
  YASL_ENFORCE(sk_inv_.size() == sk_inv.length());

  std::memcpy(sk_inv_.data(), sk_inv.data(), sk_inv.length());
  (void)curve_type_;
}

std::string BasicEcdhOprfClient::Blind(absl::string_view input) const {
  return ItemMul(absl::string_view((const char *)&private_key_[0], kEccKeySize),
                 input, ec_group_nid_);
}

std::string BasicEcdhOprfClient::Finalize(
    absl::string_view item, absl::string_view evaluated_element) const {
  std::string unblinded_element = Unblind(evaluated_element);

  return HashItem(item, unblinded_element, GetCompareLength(), hash_type_);
}

std::string BasicEcdhOprfClient::Unblind(absl::string_view input) const {
  return EcPointMul(absl::string_view((const char *)&sk_inv_[0], kEccKeySize),
                    input, ec_group_nid_);
}

size_t BasicEcdhOprfClient::GetCompareLength() const {
  if (compare_length_) {
    return compare_length_;
  }

  return kEc256CompareLength;
}
size_t BasicEcdhOprfClient::GetEcPointLength() const {
  return kEcPointCompressLength;
}

// fourq
namespace {

std::string FourQPointMul(absl::string_view sk_bytes, point_t point) {
  point_t A;

  // clear_cofactor = 1 (TRUE) or 0 (FALSE)
  // whether cofactor clearing is required or not,
  //
  bool status = ecc_mul(point, (digit_t *)sk_bytes.data(), A, false);
  YASL_ENFORCE(status, "fourq ecc_mul error, status = {}", status);

  std::string masked_point_bytes(kEccKeySize, '\0');
  encode(A, (uint8_t *)masked_point_bytes.data());

  return masked_point_bytes;
}

std::string FourQPointMul(absl::string_view sk_bytes,
                          absl::Span<const uint8_t> point_bytes) {
  point_t A;
  ECCRYPTO_STATUS status = ECCRYPTO_ERROR_UNKNOWN;

  if ((point_bytes[15] & 0x80) != 0) {  // Is bit128(PublicKey) = 0?
    status = ECCRYPTO_ERROR_INVALID_PARAMETER;
    YASL_THROW("fourq invalid point status = {}", static_cast<int>(status));
  }

  // Also verifies that A is on the curve. If it is not, it fails
  status = decode(point_bytes.data(), A);
  YASL_ENFORCE(status == ECCRYPTO_SUCCESS, "fourq decode error, status={}",
               static_cast<int>(status));

  return FourQPointMul(sk_bytes, A);
}

void FourQHashToCurvePoint(absl::string_view input, point_t pt) {
  f2elm_t r;

  // blake3 hash
  std::vector<uint8_t> hash = yasl::crypto::Blake3(input);

  std::memcpy(r, hash.data(), hash.size());
  // Reduce r; note that this does not produce a perfectly uniform distribution
  // modulo 2^127-1, but it is good enough.
  mod1271(r[0]);
  mod1271(r[1]);

  HashToCurve(r, pt);
  return;
}

}  // namespace
std::string FourQBasicEcdhOprfServer::Evaluate(
    absl::string_view blinded_element) const {
  return FourQPointMul(
      absl::string_view((const char *)&private_key_[0], kEccKeySize),
      absl::MakeSpan((const uint8_t *)blinded_element.data(),
                     blinded_element.size()));
}

std::string FourQBasicEcdhOprfServer::FullEvaluate(
    yasl::ByteContainerView input) const {
  point_t pt;
  absl::string_view input_sv =
      absl::string_view((const char *)input.data(), input.size());
  FourQHashToCurvePoint(input_sv, pt);

  std::string pt_mul_bytes = FourQPointMul(
      absl::string_view((const char *)&private_key_[0], kEccKeySize), pt);

  return HashItem(input_sv, pt_mul_bytes, GetCompareLength(), hash_type_);
}

size_t FourQBasicEcdhOprfServer::GetCompareLength() const {
  if (compare_length_) {
    return compare_length_;
  }
  return kEc256CompareLength;
}

size_t FourQBasicEcdhOprfServer::GetEcPointLength() const {
  return kEccKeySize;
}

FourQBasicEcdhOprfClient::FourQBasicEcdhOprfClient() {
  to_Montgomery(const_cast<digit_t *>(
                    reinterpret_cast<const digit_t *>(&private_key_[0])),
                reinterpret_cast<digit_t *>(sk_inv_.data()));
  Montgomery_inversion_mod_order(reinterpret_cast<digit_t *>(sk_inv_.data()),
                                 reinterpret_cast<digit_t *>(sk_inv_.data()));
  from_Montgomery(reinterpret_cast<digit_t *>(sk_inv_.data()),
                  reinterpret_cast<digit_t *>(sk_inv_.data()));
}

std::string FourQBasicEcdhOprfClient::Blind(absl::string_view input) const {
  point_t pt;
  FourQHashToCurvePoint(input, pt);

  return FourQPointMul(
      absl::string_view((const char *)&private_key_[0], kEccKeySize), pt);
}

std::string FourQBasicEcdhOprfClient::Finalize(
    absl::string_view item, absl::string_view evaluated_element) const {
  std::string unblinded_element = Unblind(evaluated_element);

  return HashItem(item, unblinded_element, GetCompareLength(), hash_type_);
}

std::string FourQBasicEcdhOprfClient::Unblind(absl::string_view input) const {
  return FourQPointMul(
      absl::string_view((const char *)&sk_inv_[0], kEccKeySize),
      absl::MakeSpan((const uint8_t *)input.data(), input.size()));
}

size_t FourQBasicEcdhOprfClient::GetCompareLength() const {
  if (compare_length_) {
    return compare_length_;
  }

  return kEc256CompareLength;
}

size_t FourQBasicEcdhOprfClient::GetEcPointLength() const {
  return kEccKeySize;
}

}  // namespace spu::psi
