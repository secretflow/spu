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
#include <optional>

#include "libspu/mpc/cheetah/rlwe/types.h"
namespace spu::mpc::cheetah {

class LWECt {
 public:
  LWECt();

  ~LWECt();

  // Gvien RLWE(\sum_{i} a_iX^i), k to create a valid LWE ciphertext that
  // decrypts to `a_k`
  LWECt(const RLWECt &rlwe, size_t coeff_index,
        const seal::SEALContext &context);

  LWECt &NegateInplace(const seal::SEALContext &context);

  LWECt &AddInplace(const LWECt &oth, const seal::SEALContext &context);

  LWECt &AddPlainInplace(const std::vector<uint64_t> &plain,
                         const seal::SEALContext &context);

  LWECt &SubInplace(const LWECt &oth, const seal::SEALContext &context);

  LWECt &SubPlainInplace(const std::vector<uint64_t> &plain,
                         const seal::SEALContext &context);

  // self += other without doing the modulus reduction
  // The `other` LWE is extracted from the RLWE on-the-fly
  LWECt &AddLazyInplace(const RLWECt &rlwe, size_t coeff_index,
                        const seal::SEALContext &context);

  // self -= other without doing the modulus reduction
  // The `other` LWE is extracted from the RLWE on-the-fly
  LWECt &SubLazyInplace(const RLWECt &rlwe, size_t coeff_index,
                        const seal::SEALContext &context);

  // Simply cast the LWE as an RLWE such that
  // the decryption of the RLWE gives the same value in the 0-th coefficient
  // i.e., Dec(LWE) = multiplier * Dec(RLWE)[0]
  //
  // Ref to Section 3.3 ``Efficient Homomorphic Conversion Between (Ring) LWE
  // Ciphertexts`` https://eprint.iacr.org/2020/015.pdf
  void CastAsRLWE(const seal::SEALContext &context, uint64_t multiplier,
                  RLWECt *out) const;

  void Reduce(const seal::SEALContext &context);

  bool IsNeedReduce() const { return lazy_counter_ > 0; }

  bool IsValid() const { return poly_deg_ > 0; }

  seal::parms_id_type parms_id() const;

  seal::parms_id_type &parms_id();

  size_t poly_modulus_degree() const { return poly_deg_; }

  size_t coeff_modulus_size() const { return cnst_term_.size(); }

  // Returns an upper bound on the size of the LWECt, as if it was written
  // to an output stream.
  size_t save_size(seal::compr_mode_type compr_mode =
                       seal::Serialization::compr_mode_default) const;

  // Saves the LWECt to a given memory location. The output is in binary
  // format and not human-readable.
  size_t save(seal::seal_byte *buffer, size_t size,
              seal::compr_mode_type compr_mode =
                  seal::Serialization::compr_mode_default) const {
    using namespace std::placeholders;
    return seal::Serialization::Save(std::bind(&LWECt::save_members, this, _1),
                                     save_size(seal::compr_mode_type::none),
                                     buffer, size, compr_mode, false);
  }

  // Loads an LWECt from an input stream overwriting the current LWECt.
  // The loaded ciphertext is verified to be valid for the given SEALContext.
  void load(const seal::SEALContext &context, const seal::seal_byte *buffer,
            size_t size);

  // Loads an LWECt from a given memory location overwriting the current
  // plaintext. No checking of the validity of the ciphertext data against
  // encryption parameters is performed. This function should not be used
  // unless the ciphertext comes from a fully trusted source.
  void unsafe_load(const seal::SEALContext &context,
                   const seal::seal_byte *buffer, size_t size) {
    using namespace std::placeholders;
    seal::Serialization::Load(
        std::bind(&LWECt::load_members, this, context, _1, _2), buffer, size,
        false);
  }

 private:
  void save_members(std::ostream &stream) const;

  void load_members(const seal::SEALContext &context, std::istream &stream,
                    seal::SEALVersion version);

  uint64_t maximum_lazy_{0};
  uint64_t lazy_counter_{0};

  size_t poly_deg_{0};
  std::vector<uint64_t> cnst_term_;
  RLWEPt vec_;
};

// We aim to lazy the moduluo reduction in x = x + y mod p given y \in [0, p).
// Because the accumulator `x` is stored in uint64_t, and thus we can lazy
// 62 - log_2(p) times.
size_t MaximumLazy(const seal::SEALContext &context);

// clang-format off
// Wrap an RLWE(m(X)) and view it as an LWE of the k-th coefficient i.e., LWE(m[k]).
// This wrapper class is used to avoid intensive memory allocation.
// clang-format on
class PhantomLWECt {
 public:
  PhantomLWECt() = default;

  ~PhantomLWECt() = default;

  void WrapIt(const RLWECt &ct, size_t coeff_index);

  seal::parms_id_type parms_id() const { return pid_; }

  seal::parms_id_type &parms_id() { return pid_; }

  size_t poly_modulus_degree() const;

  size_t coeff_modulus_size() const;

  bool IsValid() const;

  // Simply cast the LWE as an RLWE such that
  // the decryption of the RLWE gives the same value in the 0-th coefficient
  // i.e., Dec(LWE) = multiplier * Dec(RLWE)[0]
  //
  // Ref to Section 3.3 ``Efficient Homomorphic Conversion Between (Ring) LWE
  // Ciphertexts`` https://eprint.iacr.org/2020/015.pdf
  void CastAsRLWE(const seal::SEALContext &context, uint64_t multiplier,
                  RLWECt *out) const;

 private:
  size_t coeff_index_ = 0;
  seal::parms_id_type pid_ = seal::parms_id_zero;
  const RLWECt *base_ = nullptr;
};

}  // namespace spu::mpc::cheetah
