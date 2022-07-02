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

#pragma once

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "seal/seal.h"
#include "yasl/base/byte_container_view.h"
#include "yasl/link/link.h"

#include "spu/pir/seal_pir_utils.h"

#include "spu/pir/serializable.pb.h"

namespace spu::pir {

//
// SealPIR paper:
//   PIR with compressed queries and amortized query processing
//   https://eprint.iacr.org/2017/1142.pdf
// code reference microsoft opensource implement:
// https://github.com/microsoft/SealPIR

struct SealPirOptions {
  // RLWE ploynomial degree
  size_t poly_modulus_degree;
  // db element number
  size_t element_number;
  // byte size of per element
  size_t element_size;
  // number of real query data
  size_t query_size = 0;
};

struct PirParams {
  std::uint64_t n;  // number of plaintexts in database
  std::uint32_t d;  // number of dimensions for the database (1 or 2)
  std::uint32_t expansion_ratio;  // ratio of ciphertext to plaintext
  std::uint32_t dbc;  // decomposition bit count (used by relinearization)
  std::vector<std::uint64_t> nvec;  // size of each of the d dimensions
};

class SealPir {
 public:
  explicit SealPir(const SealPirOptions &options) : options_(options) {
    SetPolyModulusDegree(options.poly_modulus_degree);

    evaluator_ = std::make_unique<seal::Evaluator>(*(context_.get()));

    if (options.query_size > 0) {
      SetPirParams(options.query_size, options.element_size);
    } else {
      SetPirParams(options.element_number, options.element_size);
    }
  }

  /**
   * @brief Set the seal parameter Poly Modulus Degree
   *
   * @param degree seal Poly degree 2048/4096/8192
   */
  void SetPolyModulusDegree(size_t degree);

  /**
   * @brief Set the Pir Params object
   *
   * @param element_number  db element_number
   * @param element_size   db element bytes
   */
  void SetPirParams(size_t element_number, size_t element_size);

  template <typename T>
  std::string SerializeSealObject(const T &object) {
    std::ostringstream output;
    object.save(output);
    return output.str();
  }

  template <typename T>
  T DeSerializeSealObject(const std::string &object_bytes,
                          bool safe_load = false) {
    T seal_object;
    std::istringstream object_input(object_bytes);
    if (safe_load) {
      seal_object.load(*(context_.get()), object_input);
    } else {
      seal_object.unsafe_load(*(context_.get()), object_input);
    }
    return seal_object;
  }

  std::string SerializePlaintexts(const std::vector<seal::Plaintext> &plains);

  std::vector<seal::Plaintext> DeSerializePlaintexts(
      const std::string &plaintext_bytes, bool safe_load = false);

  yasl::Buffer SerializeCiphertexts(
      const std::vector<seal::Ciphertext> &ciphers);

  std::vector<seal::Ciphertext> DeSerializeCiphertexts(
      const CiphertextsProto &ciphers_proto, bool safe_load = false);

  std::vector<seal::Ciphertext> DeSerializeCiphertexts(
      const yasl::Buffer &ciphers_bytes, bool safe_load = false);

  yasl::Buffer SerializeQuery(
      SealPirQueryProto *query_proto,
      const std::vector<std::vector<seal::Ciphertext>> &query_ciphers);

  yasl::Buffer SerializeQuery(
      const std::vector<std::vector<seal::Ciphertext>> &query_ciphers);

  std::vector<std::vector<seal::Ciphertext>> DeSerializeQuery(
      const yasl::Buffer &query_bytes, bool safe_load = false);

  std::vector<std::vector<seal::Ciphertext>> DeSerializeQuery(
      const SealPirQueryProto query_proto, bool safe_load = false);

 protected:
  SealPirOptions options_;
  PirParams pir_params_;
  std::unique_ptr<seal::EncryptionParameters> seal_params_;

  std::unique_ptr<seal::SEALContext> context_;
  std::unique_ptr<seal::Evaluator> evaluator_;
};

//
// general single server PIR protocol
//
//     PirClient           PirServer
//                          SetupDB          offline
// ======================================
//       query                               online
//             ------------>
//                            answer
//             <------------
//   extract result
//

//
// SealPIR protocol
//
//   SealPirClient         SealPirServer
//                          SetupDB          offline
// ==================================
//   SendGaloisKeys                          online
//             ------------>  SetGaloisKeys
// ----------------------------------
//    DoPirQuery               DoPirAnswer
//   GenerateQuery
//             ------------>  ExpandQuery
//                            GenerateReply
//             <------------
//   DecodeReply
//

class SealPirClient;

class SealPirServer : public SealPir {
 public:
#ifdef DEC_DEBUG_
  SealPirServer(const SealPirOptions &options, SealPirClient &client);

#else
  SealPirServer(const SealPirOptions &options,
                const std::shared_ptr<IDbPlaintextStore> &plaintext_store);
#endif

  // read db data, convert to Seal::Plaintext
  void SetDatabase(const std::shared_ptr<IDbElementProvider> &db_provider);
  void SetDatabase(const std::vector<yasl::ByteContainerView> &db_vec);

  // set client GaloisKeys
  void SetGaloisKeys(const seal::GaloisKeys &galkey) { galois_key_ = galkey; }

  // expand one query Seal:Ciphertext
  std::vector<seal::Ciphertext> ExpandQuery(const seal::Ciphertext &encrypted,
                                            std::uint32_t m);

  // GenerateReply for query_ciphers
  std::vector<seal::Ciphertext> GenerateReply(
      const std::vector<std::vector<seal::Ciphertext>> &query_ciphers,
      size_t start_pos = 0);

  std::string SerializeDbPlaintext(int db_index = 0);
  void DeSerializeDbPlaintext(const std::string &db_serialize_bytes,
                              int db_index = 0);

  // receive, deserialize, and set client GaloisKeys
  void RecvGaloisKeys(const std::shared_ptr<yasl::link::Context> &link_ctx);

  // receive client query, and answer
  void DoPirAnswer(const std::shared_ptr<yasl::link::Context> &link_ctx);

 private:
  // for debug use, get noise budget
#ifdef DEC_DEBUG_
  SealPirClient &client_;
#endif
  std::vector<std::unique_ptr<std::vector<seal::Plaintext>>> db_vec_;
  std::shared_ptr<IDbPlaintextStore> plaintext_store_;

  seal::GaloisKeys galois_key_;

  void DecomposeToPlaintextsPtr(const seal::Ciphertext &encrypted,
                                seal::Plaintext *plain_ptr, int logt);
  std::vector<seal::Plaintext> DecomposeToPlaintexts(
      const seal::Ciphertext &encrypted);
};

class SealPirClient : public SealPir {
 public:
  explicit SealPirClient(const SealPirOptions &options);

  // db_index to  seal::Plaintexts index and offset
  uint64_t GetQueryIndex(uint64_t element_idx);
  uint64_t GetQueryOffset(uint64_t element_idx);

  // get Seal::Ciphertext
  std::vector<std::vector<seal::Ciphertext>> GenerateQuery(size_t index);

  // decode server's answer reply
  seal::Plaintext DecodeReply(const std::vector<seal::Ciphertext> &reply);

  // send GaloisKeys to server
  void SendGaloisKeys(const std::shared_ptr<yasl::link::Context> &link_ctx);

  // generate GaloisKeys
  seal::GaloisKeys GenerateGaloisKeys();

  void ComputeInverseScales();

  // when Dimension > 1
  // Compose plaintexts to ciphertext
  seal::Ciphertext ComposeToCiphertext(
      const std::vector<seal::Plaintext> &plains);

  // plaintext coefficient to bytes
  std::vector<uint8_t> PlaintextToBytes(const seal::Plaintext &plain);

  // PirQuery
  std::vector<uint8_t> DoPirQuery(
      const std::shared_ptr<yasl::link::Context> &link_ctx, size_t db_index);

 private:
  std::unique_ptr<seal::KeyGenerator> keygen_;

  std::unique_ptr<seal::Encryptor> encryptor_;
  std::unique_ptr<seal::Decryptor> decryptor_;

  std::vector<uint64_t> indices_;  // the indices for retrieval.
  std::vector<uint64_t> inverse_scales_;

  // set friend class
  friend class SealPirServer;
};
}  // namespace spu::pir
