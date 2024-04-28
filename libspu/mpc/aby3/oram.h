// Copyright 2024 Ant Group Co., Ltd.
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

#include "yacl/crypto/block_cipher/symmetric_crypto.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/mpc/kernel.h"

namespace spu::mpc::aby3 {

// Ashared index, Ashared database
class OramOneHotAA : public OramOneHotKernel {
 public:
  static constexpr char kBindName[] = "oram_onehot_aa";

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  int64_t s) const override;
};

// Ashared index, Public database
class OramOneHotAP : public OramOneHotKernel {
 public:
  static constexpr char kBindName[] = "oram_onehot_ap";

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  int64_t s) const override;
};

class OramReadOA : public OramReadKernel {
 public:
  static constexpr char kBindName[] = "oram_read_aa";

  ce::CExpr latency() const override {
    // 1 * rotate: 1
    return ce::Const(1);
  }

  ce::CExpr comm() const override {
    // 1 * rotate: k
    auto n = ce::Variable("n", "cols of database");
    return ce::K() * n;
  }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& onehot,
                  const NdArrayRef& db, int64_t offset) const override;
};

class OramReadOP : public OramReadKernel {
 public:
  static constexpr char kBindName[] = "oram_read_ap";

  ce::CExpr latency() const override {
    // 1 * rotate: 1
    return ce::Const(1);
  }

  ce::CExpr comm() const override {
    // 1 * rotate: k
    auto n = ce::Variable("n", "cols of database");
    return ce::K() * n;
  }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& onehot,
                  const NdArrayRef& db, int64_t offset) const override;
};
}  // namespace spu::mpc::aby3

namespace spu::mpc::oram {

using DpfKeyT = uint128_t;
using CorrectionFlagT = uint8_t;

enum class DpfGenCtrl { P2P0 = 0, P0P1 = 1, P1P2 = 2 };
enum class OpKind { Mul, And };

// ref: Scaling ORAM for Secure Computation
// https://eprint.iacr.org/2017/827.pdf
class OramDpf {
 public:
  std::vector<DpfKeyT> cw;  // correction words on each layer
  std::vector<std::array<CorrectionFlagT, 2>>
      cwt;  // correction bit for leftchild and right child on each layer
  std::vector<DpfKeyT> final_v;  // for b2a
  std::vector<CorrectionFlagT> final_e;

  OramDpf() = delete;
  explicit OramDpf(int64_t numel, DpfKeyT root_seed, uint128_t aes_key,
                   uint128_t target_point)
      : cw(Log2Ceil(numel), 0),
        cwt(Log2Ceil(numel), std::array<CorrectionFlagT, 2>{0, 0}),
        final_v(numel, 0),
        final_e(numel, 0),
        target_point_(target_point),
        depth_(Log2Ceil(numel)),
        numel_(numel),
        root_seed_(root_seed),
        aes_crypto_(yacl::crypto::SymmetricCrypto::CryptoType::AES128_ECB,
                    aes_key, 1){};

  // genrate 2pc-dpf according to 'ctrl'
  void gen(KernelEvalContext* ctx, DpfGenCtrl ctrl);
  std::vector<DpfKeyT> lengthDoubling(const std::vector<DpfKeyT>& input);

 private:
  uint128_t target_point_;
  int64_t depth_;
  int64_t numel_;
  DpfKeyT root_seed_;
  yacl::crypto::SymmetricCrypto aes_crypto_;
};

template <typename T>
class OramContext {
 public:
  // in boolean share after genDpf, in arithmetic after conversion
  std::vector<std::vector<T>> dpf_e;
  // v for conversion
  std::vector<std::vector<T>> convert_help_v;

  OramContext() = default;
  explicit OramContext(int64_t dpf_size)
      : dpf_e(2, std::vector<T>(dpf_size)),
        convert_help_v(2, std::vector<T>(dpf_size)),
        dpf_size_(dpf_size){};

  void genDpf(KernelEvalContext* ctx, DpfGenCtrl ctrl, uint128_t aes_key,
              uint128_t target_point);

  // ref: Duoram: A Bandwidth-Efficient Distributed ORAM for 2- and 3-Party
  // Computation
  // Appendix D
  // https://eprint.iacr.org/2022/1747
  void onehotB2A(KernelEvalContext* ctx, DpfGenCtrl ctrl);

 private:
  int64_t dpf_size_;
};

std::pair<std::vector<uint128_t>, std::vector<uint128_t>> genAesKey(
    KernelEvalContext* ctx, int64_t index_times);

template <typename T>
using Triple = std::tuple<T, T, T>;

template <typename T>
Triple<std::vector<T>> genOramBeaverPrim(KernelEvalContext* ctx, int64_t num,
                                         OpKind op, size_t adjust_rank);

template <typename T>
void genOramBeaverHelper(KernelEvalContext* ctx, int64_t num, OpKind op);
}  // namespace spu::mpc::oram