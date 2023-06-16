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

#include "libspu/psi/core/labeled_psi/psi_params.h"

#include <algorithm>
#include <cmath>
#include <string>

#include "spdlog/spdlog.h"

namespace spu::psi {

namespace {

// SEAL parameters
// from microsoft APSI example parameter sets
// https://github.com/microsoft/APSI/tree/main/parameters
std::vector<SEALParams> kSealParams = {
    {2048, 65537, 0, {48}},               // 0  *
    {4096, 40961, 0, {40, 32, 32}},       // 1
    {4096, 65537, 0, {40, 34, 30}},       // 2
    {4096, 65537, 0, {48, 30, 30}},       // 3
    {4096, 65537, 0, {48, 32, 24}},       // 4
    {4096, 65537, 0, {48, 34, 27}},       // 5
    {4096, 0, 18, {48, 32, 24}},          // 6  *
    {8192, 0, 21, {56, 56, 24, 24}},      // 7
    {8192, 0, 21, {56, 56, 56, 50}},      // 8
    {8192, 0, 21, {48, 56, 56, 48}},      // 9
    {8192, 0, 22, {56, 56, 56, 32}},      // 10
    {8192, 0, 22, {56, 56, 56, 50}},      // 11
    {8192, 0, 26, {50, 50, 50, 38, 30}},  // 12 *
    {8192, 65537, 0, {56, 48, 48}},       // 13
    {8192, 65537, 0, {56, 56, 30}},       // 14
};

}  // namespace

yacl::Buffer PsiParamsToBuffer(const apsi::PSIParams &psi_params) {
  proto::LabelPsiParamsProto psi_params_proto;

  psi_params_proto.set_hash_func_count(
      psi_params.table_params().hash_func_count);

  psi_params_proto.set_table_size(psi_params.table_params().table_size);
  psi_params_proto.set_max_items_per_bin(
      psi_params.table_params().max_items_per_bin);

  psi_params_proto.set_felts_per_item(psi_params.item_params().felts_per_item);

  psi_params_proto.set_ps_low_degree(psi_params.query_params().ps_low_degree);

  if (psi_params.query_params().query_powers.size() > 0) {
    for (const auto &power : psi_params.query_params().query_powers) {
      psi_params_proto.add_query_powers(power);
    }
  }

  auto *seal_params_proto = new proto::SealParamsProto();
  seal_params_proto->set_plain_modulus(
      psi_params.seal_params().plain_modulus().value());
  seal_params_proto->set_poly_modulus_degree(
      psi_params.seal_params().poly_modulus_degree());

  auto coeff_modulus = psi_params.seal_params().coeff_modulus();
  for (auto &coeff_moduli : coeff_modulus) {
    seal_params_proto->add_coeff_modulus(coeff_moduli.value());
  }

  psi_params_proto.set_allocated_seal_params(seal_params_proto);

  yacl::Buffer buffer(psi_params_proto.ByteSizeLong());
  psi_params_proto.SerializePartialToArray(buffer.data(), buffer.size());
  return buffer;
}

apsi::PSIParams ParsePsiParamsProto(const yacl::Buffer &buffer) {
  proto::LabelPsiParamsProto psi_params_proto;

  SPU_ENFORCE(psi_params_proto.ParseFromArray(buffer.data(), buffer.size()));

  return ParsePsiParamsProto(psi_params_proto);
}

apsi::PSIParams ParsePsiParamsProto(
    const proto::LabelPsiParamsProto &psi_params_proto) {
  apsi::PSIParams::ItemParams item_params;
  apsi::PSIParams::TableParams table_params;
  apsi::PSIParams::QueryParams query_params;
  apsi::PSIParams::SEALParams seal_params;

  item_params.felts_per_item = psi_params_proto.felts_per_item();

  table_params.hash_func_count = psi_params_proto.hash_func_count();
  table_params.table_size = psi_params_proto.table_size();
  table_params.max_items_per_bin = psi_params_proto.max_items_per_bin();

  query_params.ps_low_degree = psi_params_proto.ps_low_degree();

  if (psi_params_proto.query_powers_size() > 0) {
    for (int idx = 0; idx < psi_params_proto.query_powers_size(); ++idx) {
      query_params.query_powers.insert(psi_params_proto.query_powers(idx));
    }
  } else {
    for (size_t idx = 0; idx < table_params.max_items_per_bin; ++idx) {
      query_params.query_powers.insert(idx + 1);
    }
  }

  std::vector<seal::Modulus> coeff_modulus(
      psi_params_proto.seal_params().coeff_modulus_size());
  for (int idx = 0; idx < psi_params_proto.seal_params().coeff_modulus_size();
       ++idx) {
    coeff_modulus[idx] = psi_params_proto.seal_params().coeff_modulus(idx);
  }

  size_t poly_modulus_degree =
      psi_params_proto.seal_params().poly_modulus_degree();
  size_t plain_modulus = psi_params_proto.seal_params().plain_modulus();

  seal_params.set_poly_modulus_degree(poly_modulus_degree);

  seal_params.set_plain_modulus(plain_modulus);

  seal_params.set_coeff_modulus(coeff_modulus);

  apsi::PSIParams psi_params(item_params, table_params, query_params,
                             seal_params);

  return psi_params;
}

inline size_t GetHashTruncateSize(size_t nr, size_t ns,
                                  size_t stats_params = 40) {
  // reference:
  // Fast Private Set Intersection from Homomorphic Encryption
  // https://eprint.iacr.org/2017/299
  //  section 4.2
  // log2(Nx)+log2(Ny)+stats_params
  size_t l1 =
      std::ceil(std::log2(ns)) + std::ceil(std::log2(nr)) + stats_params;
  l1 = std::max(static_cast<size_t>(80), l1);

  return l1;
}

SEALParams GetSealParams(size_t nr, size_t ns) {
  if (nr == 1) {
    if (ns <= 3000000) {  // 3M
      return kSealParams[0];
    } else {
      return kSealParams[12];
    }
  }
  if ((nr <= 4096) && (ns <= 1000000)) {  // 1M
    return kSealParams[6];
  }

  return kSealParams[12];
}

apsi::PSIParams GetPsiParams(size_t nr, size_t ns) {
  SEALParams seal_params = GetSealParams(nr, ns);

  apsi::PSIParams::ItemParams item_params;
  apsi::PSIParams::TableParams table_params;
  apsi::PSIParams::QueryParams query_params;
  apsi::PSIParams::SEALParams apsi_seal_params;

  size_t hash_size = GetHashTruncateSize(nr, ns);
  item_params.felts_per_item = std::ceil(
      static_cast<double>(hash_size) / (seal_params.GetPlainModulusBits() - 1));

  table_params.hash_func_count = 3;
  if (nr == 1) {
    table_params.hash_func_count = 1;
  }
  size_t poly_item_count;
  poly_item_count =
      seal_params.poly_modulus_degree / item_params.felts_per_item;
  table_params.table_size = poly_item_count;

  size_t cuckoo_table_size = std::ceil(1.6 * nr);

  while (table_params.table_size < cuckoo_table_size) {
    table_params.table_size += poly_item_count;
  }

  table_params.max_items_per_bin =
      ns * table_params.hash_func_count / table_params.table_size;

  // receiver has just one item
  // sender's items size < 3 million
  if ((nr == 1) && (seal_params.poly_modulus_degree == 2048)) {
    if (ns <= 100000) {
      table_params.max_items_per_bin = 20;
    } else if (ns <= 256000) {
      table_params.max_items_per_bin = 35;
    } else {
      table_params.max_items_per_bin = 55;
    }

    query_params.ps_low_degree = 0;
    for (size_t idx = 0; idx < table_params.max_items_per_bin; ++idx) {
      query_params.query_powers.insert(idx + 1);
    }
  }

  // query_powers reference Challis and Robinson (2010)
  // http://emis.library.cornell.edu/journals/JIS/VOL13/Challis/challis6.pdf
  if (seal_params.poly_modulus_degree == 4096) {
    // 1M-256-288.json
    table_params.max_items_per_bin = 512;

    query_params.ps_low_degree = 0;
    query_params.query_powers = {
        1,   3,   4,   6,   10,  13,  15,  21,  29,  37,  45,  53,  61,  69,
        75,  77,  80,  84,  86,  87,  89,  90,  181, 183, 188, 190, 195, 197,
        206, 213, 214, 222, 230, 238, 246, 254, 261, 337, 338, 345, 353, 361,
        370, 372, 377, 379, 384, 386, 477, 479, 486, 487, 495, 503, 511};
  } else if (seal_params.poly_modulus_degree == 8192) {
    // 256M-4096.json
    table_params.max_items_per_bin = 4000;

    query_params.ps_low_degree = 310;
    query_params.query_powers = {1, 4, 10, 11, 28, 33, 78, 118, 143, 311, 1555};
  }

  // seal param
  apsi_seal_params.set_poly_modulus_degree(seal_params.poly_modulus_degree);

  if (seal_params.plain_modulus_bits > 0) {
    apsi_seal_params.set_plain_modulus(seal::PlainModulus::Batching(
        seal_params.poly_modulus_degree, seal_params.plain_modulus_bits));

  } else if (seal_params.plain_modulus > 0) {
    apsi_seal_params.set_plain_modulus(seal_params.plain_modulus);
  } else {
    SPU_THROW("SEALParams error, must set plain_modulus or plain_modulus_bits");
  }

  apsi_seal_params.set_coeff_modulus(seal::CoeffModulus::Create(
      seal_params.poly_modulus_degree, seal_params.coeff_modulus_bits));

  apsi::PSIParams psi_params(item_params, table_params, query_params,
                             apsi_seal_params);

  return psi_params;
}

}  // namespace spu::psi
