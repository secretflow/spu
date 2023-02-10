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

#include "libspu/pir/seal_pir.h"

#include <cmath>
#include <cstddef>
#include <utility>

#include "openssl/bn.h"
#include "seal/util/polyarithsmallmod.h"
#include "spdlog/spdlog.h"
#include "yacl/utils/parallel.h"

#include "libspu/core/prelude.h"

namespace spu::pir {

namespace {
// Number of coefficients needed to represent a database element
uint64_t CoefficientsPerElement(uint32_t logtp, uint64_t ele_size) {
  return std::ceil(8 * ele_size / static_cast<double>(logtp));
}

// Number of database elements that can fit in a single FV plaintext
uint64_t ElementsPerPtxt(uint32_t logt, uint64_t n, uint64_t ele_size) {
  uint64_t coeff_per_ele = CoefficientsPerElement(logt, ele_size);
  uint64_t ele_per_ptxt = n / coeff_per_ele;
  SPU_ENFORCE(ele_per_ptxt > 0);
  return ele_per_ptxt;
}

// Number of FV plaintexts needed to represent the database
uint64_t PlaintextsPerDb(uint32_t logtp, uint64_t n, uint64_t ele_num,
                         uint64_t ele_size) {
  uint64_t ele_per_ptxt = ElementsPerPtxt(logtp, n, ele_size);
  return std::ceil(static_cast<double>(ele_num) / ele_per_ptxt);
}

std::vector<uint64_t> BytesToCoeffs(uint32_t limit, const uint8_t *bytes,
                                    uint64_t size) {
  uint64_t size_out = CoefficientsPerElement(limit, size);
  std::vector<uint64_t> output(size_out);

  uint32_t room = limit;
  uint64_t *target = output.data();

  for (uint32_t i = 0; i < size; i++) {
    uint8_t src = bytes[i];
    uint32_t rest = 8;
    while (rest != 0) {
      if (room == 0) {
        target++;
        room = limit;
      }
      uint32_t shift = rest;
      if (room < rest) {
        shift = room;
      }
      *target = *target << shift;
      *target = *target | (src >> (8 - shift));
      src = src << shift;
      room -= shift;
      rest -= shift;
    }
  }

  *target = *target << room;
  return output;
}

void CoeffsToBytes(uint32_t limit, const seal::Plaintext &coeffs,
                   uint8_t *output, uint32_t size_out) {
  uint32_t room = 8;
  uint32_t j = 0;
  uint8_t *target = output;

  for (uint32_t i = 0; i < coeffs.coeff_count(); i++) {
    uint64_t src = coeffs[i];
    uint32_t rest = limit;
    while ((rest != 0) && j < size_out) {
      uint32_t shift = rest;
      if (room < rest) {
        shift = room;
      }
      target[j] = target[j] << shift;
      target[j] = target[j] | (src >> (limit - shift));
      src = src << shift;
      room -= shift;
      rest -= shift;
      if (room == 0) {
        j++;
        room = 8;
      }
    }
  }
}

void VectorToPlaintext(const std::vector<uint64_t> &coeffs,
                       seal::Plaintext *plain) {
  uint32_t coeff_count = coeffs.size();
  plain->resize(coeff_count);
  seal::util::set_uint(coeffs.data(), coeff_count, plain->data());
}

std::vector<uint64_t> GetDimensions(uint64_t plaintext_num, uint32_t d) {
  SPU_ENFORCE(d > 0);
  SPU_ENFORCE(plaintext_num > 0);

  std::vector<uint64_t> dimensions(d);

  for (uint32_t i = 0; i < d; i++) {
    dimensions[i] = std::max(
        static_cast<uint32_t>(2),
        static_cast<uint32_t>(std::floor(std::pow(plaintext_num, 1.0 / d))));
  }

  uint32_t product = 1;
  uint32_t j = 0;

  // if plaintext_num is not a d-power
  if (std::fabs(static_cast<double>(dimensions[0]) -
                std::pow(plaintext_num, 1.0 / d)) >
      std::numeric_limits<double>::epsilon()) {
    while (product < plaintext_num && j < d) {
      product = 1;
      dimensions[j++]++;
      for (uint32_t i = 0; i < d; i++) {
        product *= dimensions[i];
      }
    }
  }

  return dimensions;
}

std::vector<uint64_t> ComputeIndices(uint64_t query_index,
                                     std::vector<uint64_t> nvec) {
  uint32_t num = nvec.size();
  uint64_t product = 1;

  for (uint32_t i = 0; i < num; i++) {
    product *= nvec[i];
  }

  uint64_t j = query_index;
  std::vector<uint64_t> result;

  for (uint32_t i = 0; i < num; i++) {
    product /= nvec[i];
    uint64_t ji = j / product;

    result.push_back(ji);
    j -= ji * product;
  }

  return result;
}

}  // namespace

void SealPir::SetPolyModulusDegree(size_t degree) {
  seal_params_ =
      std::make_unique<seal::EncryptionParameters>(seal::scheme_type::bfv);

  seal_params_->set_poly_modulus_degree(degree);
  seal_params_->set_coeff_modulus(seal::CoeffModulus::BFVDefault(degree));

  if (degree == 8192) {
    seal_params_->set_plain_modulus(seal::PlainModulus::Batching(degree, 17));
    //} //else if (degree == 4096) {
    //  seal_params_->set_plain_modulus(seal::PlainModulus::Batching(degree,
    //  38));
  } else {
    SPU_THROW("poly_modulus_degree {} is not support.", degree);
  }

  context_ = std::make_unique<seal::SEALContext>(*(seal_params_));
}

void SealPir::SetPirParams(size_t element_number, size_t element_size) {
  uint32_t logt = std::floor(std::log2(seal_params_->plain_modulus().value()));
  uint32_t N = seal_params_->poly_modulus_degree();

  // number of FV plaintexts needed to represent all elements
  uint64_t plaintext_num =
      PlaintextsPerDb(logt, N, element_number, element_size);

  size_t d = 1;
  if (element_number > 8192) {
    d = 2;
  }
  std::vector<uint64_t> nvec = GetDimensions(plaintext_num, d);

  uint32_t expansion_ratio = 0;
  for (const auto &modulus : seal_params_->coeff_modulus()) {
    double logqi = std::log2(modulus.value());
    expansion_ratio += ceil(logqi / logt);
  }

  // dimension
  pir_params_.d = d;
  pir_params_.dbc = 6;
  pir_params_.n = plaintext_num;
  pir_params_.nvec = nvec;
  pir_params_.expansion_ratio = expansion_ratio
                                << 1;  // because one ciphertext = two polys
}

std::string SealPir::SerializePlaintexts(
    const std::vector<seal::Plaintext> &plains) {
  spu::pir::PlaintextsProto plains_proto;

  for (const auto &plain : plains) {
    std::string plain_bytes = SerializeSealObject<seal::Plaintext>(plain);

    plains_proto.add_data(plain_bytes.data(), plain_bytes.length());
  }
  return plains_proto.SerializeAsString();
}

std::vector<seal::Plaintext> SealPir::DeSerializePlaintexts(
    const std::string &plaintext_bytes, bool safe_load) {
  spu::pir::PlaintextsProto plains_proto;
  plains_proto.ParseFromArray(plaintext_bytes.data(), plaintext_bytes.length());

  std::vector<seal::Plaintext> plains(plains_proto.data_size());

  yacl::parallel_for(0, plains_proto.data_size(), 1,
                     [&](int64_t begin, int64_t end) {
                       for (int i = begin; i < end; ++i) {
                         plains[i] = DeSerializeSealObject<seal::Plaintext>(
                             plains_proto.data(i), safe_load);
                       }
                     });
  return plains;
}

yacl::Buffer SealPir::SerializeCiphertexts(
    const std::vector<seal::Ciphertext> &ciphers) {
  spu::pir::CiphertextsProto ciphers_proto;

  for (const auto &cipher : ciphers) {
    std::string cipher_bytes = SerializeSealObject<seal::Ciphertext>(cipher);

    ciphers_proto.add_ciphers(cipher_bytes.data(), cipher_bytes.length());
  }

  yacl::Buffer b(ciphers_proto.ByteSizeLong());
  ciphers_proto.SerializePartialToArray(b.data(), b.size());
  return b;
}

std::vector<seal::Ciphertext> SealPir::DeSerializeCiphertexts(
    const CiphertextsProto &ciphers_proto, bool safe_load) {
  std::vector<seal::Ciphertext> ciphers(ciphers_proto.ciphers_size());

  yacl::parallel_for(0, ciphers_proto.ciphers_size(), 1,
                     [&](int64_t begin, int64_t end) {
                       for (int i = begin; i < end; ++i) {
                         ciphers[i] = DeSerializeSealObject<seal::Ciphertext>(
                             ciphers_proto.ciphers(i), safe_load);
                       }
                     });
  return ciphers;
}

std::vector<seal::Ciphertext> SealPir::DeSerializeCiphertexts(
    const yacl::Buffer &ciphers_buffer, bool safe_load) {
  CiphertextsProto ciphers_proto;
  ciphers_proto.ParseFromArray(ciphers_buffer.data(), ciphers_buffer.size());

  return DeSerializeCiphertexts(ciphers_proto, safe_load);
}

yacl::Buffer SealPir::SerializeQuery(
    SealPirQueryProto *query_proto,
    const std::vector<std::vector<seal::Ciphertext>> &query_ciphers) {
  for (const auto &query_cipher : query_ciphers) {
    CiphertextsProto *ciphers_proto = query_proto->add_query_cipher();
    for (const auto &ciphertext : query_cipher) {
      std::string cipher_bytes =
          SerializeSealObject<seal::Ciphertext>(ciphertext);

      ciphers_proto->add_ciphers(cipher_bytes.data(), cipher_bytes.length());
    }
  }

  yacl::Buffer b(query_proto->ByteSizeLong());
  query_proto->SerializePartialToArray(b.data(), b.size());
  return b;
}

yacl::Buffer SealPir::SerializeQuery(
    const std::vector<std::vector<seal::Ciphertext>> &query_ciphers) {
  SealPirQueryProto query_proto;

  query_proto.set_query_size(0);
  query_proto.set_start_pos(0);

  return SerializeQuery(&query_proto, query_ciphers);
}

std::vector<std::vector<seal::Ciphertext>> SealPir::DeSerializeQuery(
    const SealPirQueryProto &query_proto, bool safe_load) {
  std::vector<std::vector<seal::Ciphertext>> pir_query(
      query_proto.query_cipher_size());

  yacl::parallel_for(
      0, query_proto.query_cipher_size(), 1, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          const auto &ciphers = query_proto.query_cipher(i);

          pir_query[i].resize(ciphers.ciphers_size());
          for (int j = 0; j < ciphers.ciphers_size(); ++j) {
            pir_query[i][j] = DeSerializeSealObject<seal::Ciphertext>(
                ciphers.ciphers(j), safe_load);
          }
        }
      });
  return pir_query;
}

std::vector<std::vector<seal::Ciphertext>> SealPir::DeSerializeQuery(
    const yacl::Buffer &query_buffer, bool safe_load) {
  SealPirQueryProto query_proto;
  query_proto.ParseFromArray(query_buffer.data(), query_buffer.size());

  return DeSerializeQuery(query_proto, safe_load);
}

#ifdef DEC_DEBUG_
// use client decrypt key get noise budget
SealPirServer::SealPirServer(
    const SealPirOptions &options,
    std::shared_ptr<IDbPlaintextStore> &plaintext_store, SealPirClient &client)
    : SealPir(options), plaintext_store_(plaintext_store), client_(client) {}
#else
SealPirServer::SealPirServer(const SealPirOptions &options,
                             std::shared_ptr<IDbPlaintextStore> plaintext_store)
    : SealPir(options), plaintext_store_(std::move(plaintext_store)) {}
#endif

// set database

void SealPirServer::SetDatabase(
    const std::shared_ptr<IDbElementProvider> &db_provider) {
  uint64_t db_size = options_.element_number * options_.element_size;

  SPU_ENFORCE(db_provider->GetDbByteSize() == db_size);

  uint32_t logt = std::floor(std::log2(seal_params_->plain_modulus().value()));
  uint32_t N = seal_params_->poly_modulus_degree();

  // number of FV plaintexts needed to represent all elements
  uint64_t plaintext_num;
  uint64_t db_num;
  if (options_.query_size == 0) {
    plaintext_num = PlaintextsPerDb(logt, N, options_.element_number,
                                    options_.element_size);
    db_num = 1;
  } else {
    plaintext_num =
        PlaintextsPerDb(logt, N, options_.query_size, options_.element_size);
    db_num = (options_.element_number + options_.query_size - 1) /
             options_.query_size;
  }

  //  number of FV plaintexts needed to create the d-dimensional matrix
  uint64_t prod = 1;
  for (auto n : pir_params_.nvec) {
    prod *= n;
  }
  uint64_t matrix_plaintexts = prod;
  SPU_ENFORCE(plaintext_num <= matrix_plaintexts);

  uint64_t ele_per_ptxt = ElementsPerPtxt(logt, N, options_.element_size);
  uint64_t bytes_per_ptxt = ele_per_ptxt * options_.element_size;

  uint64_t coeff_per_ptxt =
      ele_per_ptxt * CoefficientsPerElement(logt, options_.element_size);
  SPU_ENFORCE(coeff_per_ptxt <= N);

  plaintext_store_->SetSubDbNumber(db_num);

  for (size_t idx = 0; idx < db_num; ++idx) {
    std::vector<seal::Plaintext> db_vec;
    db_vec.reserve(matrix_plaintexts);

    uint32_t offset = idx * options_.query_size * options_.element_size;

    for (uint64_t i = 0; i < plaintext_num; i++) {
      uint64_t process_bytes = 0;

      if (db_size <= offset) {
        break;
      } else if (db_size < offset + bytes_per_ptxt) {
        process_bytes = db_size - offset;
      } else {
        process_bytes = bytes_per_ptxt;
      }

      // Get the coefficients of the elements that will be packed in plaintext i
      std::vector<uint8_t> element_bytes =
          db_provider->ReadElement(offset, process_bytes);
      std::vector<uint64_t> coefficients =
          BytesToCoeffs(logt, element_bytes.data(), process_bytes);
      offset += process_bytes;

      uint64_t used = coefficients.size();

      SPU_ENFORCE(used <= coeff_per_ptxt);

      // Pad the rest with 1s
      for (uint64_t j = 0; j < (N - used); j++) {
        coefficients.push_back(1);
      }

      seal::Plaintext plain;
      VectorToPlaintext(coefficients, &plain);
      db_vec.push_back(std::move(plain));
    }

    // Add padding to make database a matrix
    uint64_t current_plaintexts = db_vec.size();
    SPU_ENFORCE(current_plaintexts <= plaintext_num);

#ifdef DEC_DEBUG_
    SPDLOG_INFO(
        "adding: {} FV plaintexts of padding (equivalent to: {} elements",
        (matrix_plaintexts - current_plaintexts),
        (matrix_plaintexts - current_plaintexts) *
            elements_per_ptxt(logt, N, options_.element_size));
#endif

    std::vector<uint64_t> padding(N, 1);

    for (uint64_t i = 0; i < (matrix_plaintexts - current_plaintexts); i++) {
      seal::Plaintext plain;
      VectorToPlaintext(padding, &plain);
      db_vec.push_back(plain);
    }

    // pre process db
    yacl::parallel_for(0, db_vec.size(), 1, [&](int64_t begin, int64_t end) {
      for (uint32_t i = begin; i < end; i++) {
        evaluator_->transform_to_ntt_inplace(db_vec[i],
                                             context_->first_parms_id());
      }
    });
    plaintext_store_->SavePlaintexts(db_vec, idx);
  }
}

void SealPirServer::SetDatabase(
    const std::vector<yacl::ByteContainerView> &db_vec) {
  std::vector<uint8_t> db_flatten_bytes(db_vec.size() * options_.element_size);
  for (size_t idx = 0; idx < db_vec.size(); ++idx) {
    std::memcpy(&db_flatten_bytes[idx * options_.element_size],
                db_vec[idx].data(), db_vec[idx].size());
  }

  std::shared_ptr<IDbElementProvider> db_provider =
      std::make_shared<MemoryDbElementProvider>(db_flatten_bytes,
                                                options_.element_size);

  return SetDatabase(db_provider);
}

std::vector<seal::Ciphertext> SealPirServer::ExpandQuery(
    const seal::Ciphertext &encrypted, std::uint32_t m) {
  uint64_t plain_mod = seal_params_->plain_modulus().value();

  seal::GaloisKeys &galkey = galois_key_;

  // Assume that m is a power of 2. If not, round it to the next power of 2.
  uint32_t logm = std::ceil(std::log2(m));

  std::vector<int> galois_elts;
  auto n = seal_params_->poly_modulus_degree();
  SPU_ENFORCE(logm <= std::ceil(std::log2(n)), "m > n is not allowed.");

  galois_elts.reserve(std::ceil(std::log2(n)));
  for (int i = 0; i < std::ceil(std::log2(n)); i++) {
    galois_elts.push_back((n + seal::util::exponentiate_uint(2, i)) /
                          seal::util::exponentiate_uint(2, i));
  }

  std::vector<seal::Ciphertext> results(1);
  results[0] = encrypted;
  seal::Plaintext tempPt;
  for (size_t j = 0; j < logm - 1; j++) {
    std::vector<seal::Ciphertext> results2(1 << (j + 1));
    int step = 1 << j;
    seal::Plaintext pt0(n);
    seal::Plaintext pt1(n);

    pt0.set_zero();
    pt0[n - step] = plain_mod - 1;

    int index_raw = (n << 1) - (1 << j);
    int index = (index_raw * galois_elts[j]) % (n << 1);
    pt1.set_zero();
    pt1[index] = 1;

    // int nstep = -step;
    yacl::parallel_for(0, step, 1, [&](int64_t begin, int64_t end) {
      for (int k = begin; k < end; k++) {
        seal::Ciphertext c0;
        seal::Ciphertext c1;
        seal::Ciphertext t0;
        seal::Ciphertext t1;

        c0 = results[k];

        // SPDLOG_INFO("apply_galois j:{} k:{}", j, k);
        evaluator_->apply_galois(c0, galois_elts[j], galkey, t0);
        evaluator_->add(c0, t0, results2[k]);

        evaluator_->multiply_plain(c0, pt0, c1);
        evaluator_->multiply_plain(t0, pt1, t1);
        evaluator_->add(c1, t1, results2[k + step]);
      }
    });
    results = results2;
  }

  // Last step of the loop
  std::vector<seal::Ciphertext> results2(results.size() << 1);
  seal::Plaintext two("2");

  seal::Plaintext pt0(n);
  seal::Plaintext pt1(n);

  pt0.set_zero();
  pt0[n - results.size()] = plain_mod - 1;

  int index_raw = (n << 1) - (1 << (logm - 1));
  int index = (index_raw * galois_elts[logm - 1]) % (n << 1);
  pt1.set_zero();
  pt1[index] = 1;

  for (uint32_t k = 0; k < results.size(); k++) {
    if (k >= (m - (1 << (logm - 1)))) {  // corner case.
      evaluator_->multiply_plain(results[k], two,
                                 results2[k]);  // plain multiplication by 2.
    } else {
      seal::Ciphertext c0;
      seal::Ciphertext c1;
      seal::Ciphertext t0;
      seal::Ciphertext t1;

      c0 = results[k];
      evaluator_->apply_galois(c0, galois_elts[logm - 1], galkey, t0);
      evaluator_->add(c0, t0, results2[k]);

      evaluator_->multiply_plain(c0, pt0, c1);
      evaluator_->multiply_plain(t0, pt1, t1);
      evaluator_->add(c1, t1, results2[k + results.size()]);
    }
  }

  auto first = results2.begin();
  auto last = results2.begin() + m;
  std::vector<seal::Ciphertext> new_vec(first, last);
  return new_vec;
}

std::vector<seal::Ciphertext> SealPirServer::GenerateReply(
    const std::vector<std::vector<seal::Ciphertext>> &query_ciphers,
    size_t start_pos) {
  std::vector<uint64_t> nvec = pir_params_.nvec;
  uint64_t product = 1;

  for (auto n : nvec) {
    product *= n;
  }

  auto coeff_count = seal_params_->poly_modulus_degree();

  size_t sub_db_index = 0;
  if (options_.query_size > 0) {
    sub_db_index = start_pos / options_.query_size;
  }

  std::vector<seal::Plaintext> db_plaintexts =
      plaintext_store_->ReadPlaintexts(sub_db_index);
  std::vector<seal::Plaintext> *cur = &db_plaintexts;

  std::vector<seal::Plaintext> intermediate_plain;  // decompose....

  auto pool = seal::MemoryManager::GetPool();

  int N = seal_params_->poly_modulus_degree();

  int logt = std::floor(std::log2(seal_params_->plain_modulus().value()));

  for (uint32_t i = 0; i < nvec.size(); i++) {
    std::vector<seal::Ciphertext> expanded_query;

    uint64_t n_i = nvec[i];

    for (uint32_t j = 0; j < query_ciphers[i].size(); j++) {
      uint64_t total = N;
      if (j == query_ciphers[i].size() - 1) {
        total = n_i % N;
      }

      std::vector<seal::Ciphertext> expanded_query_part =
          ExpandQuery(query_ciphers[i][j], total);

      expanded_query.insert(
          expanded_query.end(),
          std::make_move_iterator(expanded_query_part.begin()),
          std::make_move_iterator(expanded_query_part.end()));
      expanded_query_part.clear();
    }
    if (expanded_query.size() != n_i) {
      SPDLOG_INFO(" size mismatch!!! {}-{}", expanded_query.size(), n_i);
    }

#ifdef DEC_DEBUG_
    SPDLOG_INFO("Checking expanded query, size = {}", expanded_query.size());

    seal::Plaintext tempPt;
    for (size_t h = 0; h < expanded_query.size(); h++) {
      client_.decryptor_->decrypt(expanded_query[h], tempPt);

      SPDLOG_INFO("h:{} noise budget = {}, tempPt: {}", h,
                  client_.decryptor_->invariant_noise_budget(expanded_query[h]),
                  tempPt.to_string());
    }

#endif

    // Transform expanded query to NTT, and ...
    yacl::parallel_for(
        0, expanded_query.size(), 1, [&](int64_t begin, int64_t end) {
          for (uint32_t jj = begin; jj < end; jj++) {
            evaluator_->transform_to_ntt_inplace(expanded_query[jj]);
          }
        });

    // Transform plaintext to NTT. If database is pre-processed, can skip
    if (i > 0) {
      yacl::parallel_for(0, (*cur).size(), 1, [&](int64_t begin, int64_t end) {
        for (uint32_t jj = begin; jj < end; jj++) {
          evaluator_->transform_to_ntt_inplace((*cur)[jj],
                                               context_->first_parms_id());
        }
      });
    }

#ifdef DEC_DEBUG_
    for (uint64_t k = 0; k < product; k++) {
      if ((*cur)[k].is_zero()) {
        SPDLOG_INFO("k: {}, product: {}-th ptxt = 0 ", (k + 1), product);
      }
    }
#endif

    product /= n_i;
    std::vector<seal::Ciphertext> intermediateCtxts(product);

    yacl::parallel_for(0, product, 1, [&](int64_t begin, int64_t end) {
      for (int k = begin; k < end; k++) {
        uint64_t j = 0;
        while ((*cur)[k + j * product].is_zero()) {
          j++;
        }

        evaluator_->multiply_plain(expanded_query[j], (*cur)[k + j * product],
                                   intermediateCtxts[k]);

        seal::Ciphertext temp;
        for (j += 1; j < n_i; j++) {
          if ((*cur)[k + j * product].is_zero()) {
            SPDLOG_INFO("cur[{}] is zero, k:{}, j:{}", (k + j * product), k, j);
            continue;
          }
          evaluator_->multiply_plain(expanded_query[j], (*cur)[k + j * product],
                                     temp);
          evaluator_->add_inplace(intermediateCtxts[k],
                                  temp);  // Adds to first component.
        }
      }
    });

    yacl::parallel_for(
        0, intermediateCtxts.size(), 1, [&](int64_t begin, int64_t end) {
          for (uint32_t jj = begin; jj < end; jj++) {
            evaluator_->transform_from_ntt_inplace(intermediateCtxts[jj]);
          }
        });

    if (i == nvec.size() - 1) {
      return intermediateCtxts;
    } else {
      intermediate_plain.clear();
      intermediate_plain.reserve(pir_params_.expansion_ratio * product);
      cur = &intermediate_plain;

      auto tempplain = seal::util::allocate<seal::Plaintext>(
          pir_params_.expansion_ratio * product, pool, coeff_count);

      for (uint64_t rr = 0; rr < product; rr++) {
        DecomposeToPlaintextsPtr(
            intermediateCtxts[rr],
            tempplain.get() + rr * pir_params_.expansion_ratio, logt);

        for (uint32_t jj = 0; jj < pir_params_.expansion_ratio; jj++) {
          auto offset = rr * pir_params_.expansion_ratio + jj;
          intermediate_plain.emplace_back(tempplain[offset]);
        }
      }
      product *= pir_params_.expansion_ratio;  // multiply by expansion rate.
    }
    SPDLOG_INFO("Server: {}-th recursion level finished", (i + 1));
  }
  SPDLOG_INFO("reply generated!  ");
  // This should never get here

  std::vector<seal::Ciphertext> fail(1);
  return fail;
}

inline void SealPirServer::DecomposeToPlaintextsPtr(
    const seal::Ciphertext &encrypted, seal::Plaintext *plain_ptr, int logt) {
  std::vector<seal::Plaintext> result;
  auto coeff_count = seal_params_->poly_modulus_degree();
  auto coeff_mod_count = seal_params_->coeff_modulus().size();
  auto encrypted_count = encrypted.size();

  uint64_t t1 = 1ULL << logt;  //  t1 <= t.

  uint64_t t1minusone = t1 - 1;
  // A triple for loop. Going over polys, moduli, and decomposed index.

  for (size_t i = 0; i < encrypted_count; i++) {
    const uint64_t *encrypted_pointer = encrypted.data(i);
    for (size_t j = 0; j < coeff_mod_count; j++) {
      // populate one poly at a time.
      // create a polynomial to store the current decomposition value
      // which will be copied into the array to populate it at the current
      // index.
      double logqj = std::log2(seal_params_->coeff_modulus()[j].value());
      // int expansion_ratio = ceil(logqj + exponent - 1) / exponent;
      int expansion_ratio = std::ceil(logqj / logt);

      uint64_t curexp = 0;
      for (int k = 0; k < expansion_ratio; k++) {
        // Decompose here
        for (size_t m = 0; m < coeff_count; m++) {
          plain_ptr[i * coeff_mod_count * expansion_ratio +
                    j * expansion_ratio + k][m] =
              (*(encrypted_pointer + m + (j * coeff_count)) >> curexp) &
              t1minusone;
        }
        curexp += logt;
      }
    }
  }
}

std::vector<seal::Plaintext> SealPirServer::DecomposeToPlaintexts(
    const seal::Ciphertext &encrypted) {
  std::vector<seal::Plaintext> result;
  auto coeff_count = seal_params_->poly_modulus_degree();
  auto coeff_mod_count = seal_params_->coeff_modulus().size();
  // auto plain_bit_count = seal_params_->plain_modulus().bit_count();
  auto encrypted_count = encrypted.size();

  // Generate powers of t.
  uint64_t plain_mod = seal_params_->plain_modulus().value();

  // A triple for loop. Going over polys, moduli, and decomposed index.
  for (size_t i = 0; i < encrypted_count; i++) {
    const uint64_t *encrypted_pointer = encrypted.data(i);
    for (size_t j = 0; j < coeff_mod_count; j++) {
      // populate one poly at a time.
      // create a polynomial to store the current decomposition value
      // which will be copied into the array to populate it at the current
      // index.
      int logqj = std::log2(seal_params_->coeff_modulus()[j].value());
      int expansion_ratio = std::ceil(logqj / std::log2(plain_mod));

      uint64_t cur = 1;
      for (int k = 0; k < expansion_ratio; k++) {
        // Decompose here
        seal::Plaintext temp(coeff_count);
        std::transform(
            encrypted_pointer + (j * coeff_count),
            encrypted_pointer + ((j + 1) * coeff_count), temp.data(),
            [cur, &plain_mod](auto &in) { return (in / cur) % plain_mod; });

        result.emplace_back(std::move(temp));
        cur *= plain_mod;
      }
    }
  }

  return result;
}

std::string SealPirServer::SerializeDbPlaintext(int db_index) {
  return SerializePlaintexts(*db_vec_[db_index].get());
}

void SealPirServer::DeSerializeDbPlaintext(
    const std::string &db_serialize_bytes, int db_index) {
  std::vector<seal::Plaintext> plaintext_vec =
      DeSerializePlaintexts(db_serialize_bytes);

  db_vec_[db_index] =
      std::make_unique<std::vector<seal::Plaintext>>(plaintext_vec);
}

void SealPirServer::RecvGaloisKeys(
    const std::shared_ptr<yacl::link::Context> &link_ctx) {
  yacl::Buffer galkey_buffer = link_ctx->Recv(
      link_ctx->NextRank(),
      fmt::format("recv galios key from rank-{}", link_ctx->Rank()));

  std::string galkey_str(galkey_buffer.size(), '\0');
  std::memcpy(galkey_str.data(), galkey_buffer.data(), galkey_buffer.size());

  auto galkey = DeSerializeSealObject<seal::GaloisKeys>(galkey_str);
  SetGaloisKeys(galkey);
}

void SealPirServer::DoPirAnswer(
    const std::shared_ptr<yacl::link::Context> &link_ctx) {
  yacl::Buffer query_buffer =
      link_ctx->Recv(link_ctx->NextRank(), fmt::format("recv query ciphers"));

  SealPirQueryProto query_proto;
  query_proto.ParseFromArray(query_buffer.data(), query_buffer.size());

  std::vector<std::vector<seal::Ciphertext>> query_ciphers =
      DeSerializeQuery(query_proto);
  std::vector<seal::Ciphertext> reply_ciphers =
      GenerateReply(query_ciphers, query_proto.start_pos());

  yacl::Buffer reply_buffer = SerializeCiphertexts(reply_ciphers);
  link_ctx->SendAsync(
      link_ctx->NextRank(), reply_buffer,
      fmt::format("send query reply size:{}", reply_buffer.size()));
}

// SealPirClient
SealPirClient::SealPirClient(const SealPirOptions &options) : SealPir(options) {
  keygen_ = std::make_unique<seal::KeyGenerator>(*context_);

  seal::PublicKey public_key_;
  keygen_->create_public_key(public_key_);

  encryptor_ = std::make_unique<seal::Encryptor>(*context_, public_key_);

  seal::SecretKey secret_key = keygen_->secret_key();

  decryptor_ = std::make_unique<seal::Decryptor>(*context_, secret_key);
}

seal::GaloisKeys SealPirClient::GenerateGaloisKeys() {
  // Generate the Galois keys needed for coeff_select.
  std::vector<uint32_t> galois_elts;
  int N = seal_params_->poly_modulus_degree();
  int logN = seal::util::get_power_of_two(N);

  galois_elts.reserve(logN);
  for (int i = 0; i < logN; i++) {
    galois_elts.push_back((N + seal::util::exponentiate_uint(2, i)) /
                          seal::util::exponentiate_uint(2, i));
  }

  seal::GaloisKeys galois_keys;
  keygen_->create_galois_keys(galois_elts, galois_keys);

  return galois_keys;
}

std::vector<std::vector<seal::Ciphertext>> SealPirClient::GenerateQuery(
    size_t index) {
  size_t query_indx = GetQueryIndex(index);

  indices_ = ComputeIndices(query_indx, pir_params_.nvec);

  ComputeInverseScales();

  std::vector<std::vector<seal::Ciphertext>> result(pir_params_.d);
  int N = seal_params_->poly_modulus_degree();

  seal::Plaintext pt(seal_params_->poly_modulus_degree());
  for (uint32_t i = 0; i < indices_.size(); i++) {
    uint32_t num_ptxts = ceil((pir_params_.nvec[i] + 0.0) / N);

    for (uint32_t j = 0; j < num_ptxts; j++) {
      pt.set_zero();
      if (indices_[i] > N * (j + 1) || indices_[i] < N * j) {
        // just encrypt zero
      } else {
        uint64_t real_index = indices_[i] - N * j;
        pt[real_index] = 1;
      }
      seal::Ciphertext dest;
      encryptor_->encrypt(pt, dest);

      result[i].push_back(dest);
    }
  }

  return result;
}

seal::Plaintext SealPirClient::DecodeReply(
    const std::vector<seal::Ciphertext> &reply) {
  uint32_t exp_ratio = pir_params_.expansion_ratio;
  uint32_t recursion_level = pir_params_.d;

  std::vector<seal::Ciphertext> temp = reply;

  uint64_t t = seal_params_->plain_modulus().value();

  for (uint32_t i = 0; i < recursion_level; i++) {
    std::vector<seal::Ciphertext> newtemp;
    std::vector<seal::Plaintext> tempplain;

    for (uint32_t j = 0; j < temp.size(); j++) {
      seal::Plaintext ptxt;
      decryptor_->decrypt(temp[j], ptxt);
#ifdef DEC_DEBUG_
      // SPDLOG_INFO("Client: reply noise budget = {}",
      //             decryptor_->invariant_noise_budget(temp[j]));
      //  SPDLOG_INFO("ptxt to_string: {}", ptxt.to_string());
#endif
      // multiply by inverse_scale for every coefficient of ptxt
      for (size_t h = 0; h < ptxt.coeff_count(); h++) {
        ptxt[h] *= inverse_scales_[recursion_level - 1 - i];

        ptxt[h] %= t;
      }
      tempplain.push_back(ptxt);

#ifdef DEC_DEBUG_
      // SPDLOG_INFO("recursion level : {} noise budget : {}", i,
      //             decryptor_->invariant_noise_budget(temp[j]));
#endif

      if ((j + 1) % exp_ratio == 0 && j > 0) {
        // Combine into one ciphertext.
        seal::Ciphertext combined = ComposeToCiphertext(tempplain);
        newtemp.push_back(combined);
        tempplain.clear();
      }
    }

    if (i == recursion_level - 1) {
      assert(temp.size() == 1);

      return tempplain[0];
    } else {
      tempplain.clear();
      temp = newtemp;
    }
  }

  // This should never be called
  assert(0);
  seal::Plaintext fail;
  return fail;
}

uint64_t SealPirClient::GetQueryIndex(uint64_t element_idx) {
  auto N = seal_params_->poly_modulus_degree();
  auto logt = std::floor(std::log2(seal_params_->plain_modulus().value()));

  auto ele_per_ptxt = ElementsPerPtxt(logt, N, options_.element_size);
  return static_cast<uint64_t>(element_idx / ele_per_ptxt);
}

uint64_t SealPirClient::GetQueryOffset(uint64_t element_idx) {
  uint32_t N = seal_params_->poly_modulus_degree();
  uint32_t logt = std::floor(std::log2(seal_params_->plain_modulus().value()));

  uint64_t ele_per_ptxt = ElementsPerPtxt(logt, N, options_.element_size);
  return element_idx % ele_per_ptxt;
}

std::vector<uint8_t> SealPirClient::PlaintextToBytes(
    const seal::Plaintext &plain) {
  uint32_t N = seal_params_->poly_modulus_degree();
  uint32_t logt = std::floor(std::log2(seal_params_->plain_modulus().value()));

  // Convert from FV plaintext (polynomial) to database element at the client
  std::vector<uint8_t> elements(N * logt / 8);

  CoeffsToBytes(logt, plain, elements.data(), (N * logt) / 8);

  return elements;
}

void SealPirClient::ComputeInverseScales() {
  SPU_ENFORCE(indices_.size() == pir_params_.nvec.size(), "size mismatch");

  int logt = std::floor(std::log2(seal_params_->plain_modulus().value()));

  uint64_t N = seal_params_->poly_modulus_degree();
  uint64_t t = seal_params_->plain_modulus().value();
  int logN = std::log2(N);
  int logm = logN;

  inverse_scales_.clear();

  for (size_t i = 0; i < pir_params_.nvec.size(); i++) {
    // uint64_t index_modN = indices_[i] % N;
    uint64_t numCtxt =
        ceil((pir_params_.nvec[i] + 0.0) / N);  // number of query ciphertexts.
    uint64_t batchId = indices_[i] / N;
    if (batchId == numCtxt - 1) {
      logm = std::ceil(std::log2((pir_params_.nvec[i] % N)));
    }

    uint64_t inverse_scale;

    int quo = logm / logt;
    int mod = logm % logt;
    inverse_scale = std::pow(2, logt - mod);

    if ((quo + 1) % 2 != 0) {
      inverse_scale =
          seal_params_->plain_modulus().value() - pow(2, logt - mod);
    }
    // get mod inverse
    {
      BN_CTX *bn_ctx = BN_CTX_new();
      BIGNUM *bn_t = BN_new();
      BIGNUM *bn_m = BN_new();
      BIGNUM *ret = BN_new();
      BN_set_word(bn_t, t);
      BN_set_word(bn_m, 1 << logm);

      BN_mod_inverse(ret, bn_m, bn_t, bn_ctx);
      inverse_scale = BN_get_word(ret);

      BN_free(bn_t);
      BN_free(bn_m);
      BN_free(ret);
      BN_CTX_free(bn_ctx);
    }
    inverse_scales_.push_back(inverse_scale);
    if ((inverse_scale << logm) % t != 1) {
      SPU_THROW("get inverse wrong");
    }
  }
}

seal::Ciphertext SealPirClient::ComposeToCiphertext(
    const std::vector<seal::Plaintext> &plains) {
  size_t encrypted_count = 2;
  auto coeff_count = seal_params_->poly_modulus_degree();
  auto coeff_mod_count = seal_params_->coeff_modulus().size();
  uint64_t plainMod = seal_params_->plain_modulus().value();
  int logt = std::floor(std::log2(plainMod));

  seal::Ciphertext result(*context_);
  result.resize(encrypted_count);

  // A triple for loop. Going over polys, moduli, and decomposed index.
  for (size_t i = 0; i < encrypted_count; i++) {
    uint64_t *encrypted_pointer = result.data(i);

    for (size_t j = 0; j < coeff_mod_count; j++) {
      // populate one poly at a time.
      // create a polynomial to store the current decomposition value
      // which will be copied into the array to populate it at the current
      // index.
      double logqj = std::log2(seal_params_->coeff_modulus()[j].value());
      int expansion_ratio = std::ceil(logqj / logt);
      uint64_t cur = 1;

      for (int k = 0; k < expansion_ratio; k++) {
        // Compose here
        const uint64_t *plain_coeff =
            plains[k + j * (expansion_ratio) +
                   i * (coeff_mod_count * expansion_ratio)]
                .data();

        for (size_t m = 0; m < coeff_count; m++) {
          if (k == 0) {
            *(encrypted_pointer + m + j * coeff_count) =
                *(plain_coeff + m) * cur;
          } else {
            *(encrypted_pointer + m + j * coeff_count) +=
                *(plain_coeff + m) * cur;
          }
        }

        cur <<= logt;
      }
    }
  }

  return result;
}

void SealPirClient::SendGaloisKeys(
    const std::shared_ptr<yacl::link::Context> &link_ctx) {
  seal::GaloisKeys galkey = GenerateGaloisKeys();

  std::string galkey_str = SerializeSealObject<seal::GaloisKeys>(galkey);
  yacl::Buffer galkey_buffer(galkey_str.data(), galkey_str.length());

  link_ctx->SendAsync(
      link_ctx->NextRank(), galkey_buffer,
      fmt::format("send galios key to rank-{}", link_ctx->Rank()));
}

std::vector<uint8_t> SealPirClient::DoPirQuery(
    const std::shared_ptr<yacl::link::Context> &link_ctx, size_t db_index) {
  size_t query_index = db_index;
  size_t start_pos = 0;
  //
  if (options_.query_size != 0) {
    query_index = db_index % options_.query_size;
    start_pos = db_index - query_index;
  }

  std::vector<std::vector<seal::Ciphertext>> query_ciphers =
      GenerateQuery(query_index);

  SealPirQueryProto query_proto;
  query_proto.set_query_size(options_.query_size);
  query_proto.set_start_pos(start_pos);

  yacl::Buffer query_buffer = SerializeQuery(&query_proto, query_ciphers);
  link_ctx->SendAsync(
      link_ctx->NextRank(), query_buffer,
      fmt::format("send query message({})", query_buffer.size()));

  yacl::Buffer reply_buffer =
      link_ctx->Recv(link_ctx->NextRank(), fmt::format("send query message"));

  std::vector<seal::Ciphertext> reply_ciphers =
      DeSerializeCiphertexts(reply_buffer);

  seal::Plaintext query_plain = DecodeReply(reply_ciphers);

  std::vector<uint8_t> plaintext_bytes = PlaintextToBytes(query_plain);

  std::vector<uint8_t> query_reply_data(options_.element_size);

  size_t offset = GetQueryOffset(query_index);

  std::memcpy(query_reply_data.data(),
              &plaintext_bytes[offset * options_.element_size],
              options_.element_size);

  return query_reply_data;
}

}  // namespace spu::pir
