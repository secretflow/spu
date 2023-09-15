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

#include <utility>
#include <vector>

#include "libspu/core/ndarray_ref.h"

namespace spu::mpc {

// The basic io interface of protocols.
class IoInterface {
 public:
  virtual ~IoInterface() = default;

  // Make shares (public/secret) from plaintext encoded in ring.
  //
  // @param raw, with type as RingTy.
  // @param vis, visibility the result.
  // @param owner_rank, only valid when vis is VIS_SECRET
  // @return a list of random values, each of which could be send to one mpc
  //         engine.
  //
  // TODO: currently, the `private type` is transparent to compiler.
  //
  // This function does NOT handle encoding stuffs, it's the upper layer(hal)'s
  // responsibility to encode to it to ring.
  virtual std::vector<NdArrayRef> toShares(const NdArrayRef& raw,
                                           Visibility vis,
                                           int owner_rank = -1) const = 0;

  virtual Type getShareType(Visibility vis, int owner_rank = -1) const = 0;

  // Make a secret from a bit array, if the element type is large than one bit,
  // only the lsb is considered.
  //
  // @param raw, with type as PtType.
  virtual std::vector<NdArrayRef> makeBitSecret(
      const NdArrayRef& raw) const = 0;

  virtual size_t getBitSecretShareSize(size_t numel) const = 0;

  virtual bool hasBitSecretSupport() const = 0;

  // Reconstruct shares into a RingTy value.
  //
  // @param shares, a list of secret shares.
  // @return a revealed value in ring2k space.
  virtual NdArrayRef fromShares(
      const std::vector<NdArrayRef>& shares) const = 0;
};

//
class BaseIo : public IoInterface {
 protected:
  FieldType const field_;
  size_t const world_size_;

 public:
  explicit BaseIo(FieldType field, size_t world_size)
      : field_(field), world_size_(world_size) {}

  std::vector<NdArrayRef> makeBitSecret(const NdArrayRef&) const override {
    SPU_THROW("should not be here");
  }

  size_t getBitSecretShareSize(size_t) const override {
    SPU_THROW("should not be here");
  }
  bool hasBitSecretSupport() const override { return false; }
};

}  // namespace spu::mpc
