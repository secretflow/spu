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

#include <map>
#include <memory>
#include <vector>

#include "libspu/core/context.h"
#include "libspu/core/pt_buffer_view.h"
#include "libspu/core/value.h"
#include "libspu/device/symbol_table.h"
#include "libspu/mpc/io_interface.h"

namespace spu::device {

// The IO progress refers to two independent progresses:
// - infeed: transfer (plaintext) data from input owner to SPU (encrypted).
// - outfeed: transfer (encrypted) data from SPU to result owner (plaintext).
//
// Infeed can be further split to 3 sub-steps.
// 1. encrypt (make_shares)               on data provider.
// 2. dispatch (shares to runtime slices) by the scheduler.
// 3. set (share to runtime slice)        on spu runtime slice.
//
// Outfeed can also be split into 3 sub-steps.
// 1. get (share from runtime slice)      on spu runtime slice.
// 2. gather (shares form runtime slices) by the scheduler.
// 3. decrypt (reconstruct shares)        on data receiver.
//
// Since SPU is designed to be easily integrated with upper schedule system,
// step-2 (dispatch & gather) is left to upper system to do, so SPU itself does
// NOT provide infeed/outfeed interface directly. Instead, SPU provides:
//
// ```code
//   interface IoClient:
//     make_shares :: Plaintext -> Visibility -> [Share]
//     reconstruct :: [Share] -> Plaintext
//
//   interface RuntimeEnv:
//     set_var :: string -> Share -> Nil
//     get_var :: string -> Share
// ```
//
// # The colocated situation
//
// When the IoClient is colocated with SPU runtime slice, we can directly do the
// transfer (schedule) since we already have the runtime link, the performance
// will be better.
//
// ```code
//    interface ColocatedIo:
//      infeed :: Plaintext -> Value
//      outfeed :: Value -> Plaintext
// ```
//
// But the above does not working well since, because:
// 1. from runtime perspective of view, since SPU runtime works in a SPMD
//    manner, all runtime slice must call `infeed` simultaneously.
// 2. from data owner's perspective, only one party is providing/receiving the
//    data, it should be called from that party only.
//
// So we add a host/device methods for it.
// - data owner/receiver always call hostXXX method, in a free manner.
// - runtime always call deviceXXX method, in a SPMD manner.
// - between host/device visiting, a sync should be called, in a SPMD manner.
//
// For example, suppose we have three hosts, each of which both as data provider
// and runtime provider.
//
// P0:   io.hostSetVar('x', x_pri)
// P1:   io.hostSetVar('y', y_pri)
// ALL:  io.sync()
// ALL:  Value x = io.deviceGetVar('x', ...)
// ALL:  Value y = io.deviceGetVar('y', ...)
// ALL:  Value z = spu_function(x, y)
// ALL:  io.deviceSetVar('z', z)
// P2:   z_pri = io.hostGetVar('z')
//
// The rule is: before the first SPMD (ALL) host acts together, we should call
// sync first.
//
// TODO: reveal_to is not supported.

class IoClient {
  size_t const world_size_;

  RuntimeConfig const config_;

  std::unique_ptr<mpc::IoInterface> base_io_;

 public:
  explicit IoClient(size_t world_size, const RuntimeConfig &config);

  // Make shares from plaintext buffer view.
  // Valid owner must be >= 0, and -1 indicates outsourcing model.
  std::vector<spu::Value> makeShares(const PtBufferView &bv, Visibility vtype,
                                     int owner_rank = -1);

  // Combine shares to a plaintext ndarray.
  NdArrayRef combineShares(absl::Span<spu::Value const> values);
};

class ColocatedIo {
  // The hardware context.
  SPUContext *sctx_;

  // the place that variables will be store/load.
  SymbolTable symbols_;

  // un-synchronized data.
  struct PrivData {
    NdArrayRef arr;
    Visibility vtype;
  };
  std::map<std::string, PrivData> unsynced_;

 public:
  explicit ColocatedIo(SPUContext *sctx);

  size_t getWorldSize() const { return sctx_->lctx()->WorldSize(); }

  size_t getRank() const { return sctx_->lctx()->Rank(); }

  void hostSetVar(const std::string &name, const PtBufferView &bv,
                  Visibility vtype = VIS_SECRET);

  // Get a variable from this io context.
  // 1. if a new variable in current host's unsynced list , it will be fetch
  //    directly, in this case, if a same named variable exist in synchronized
  //    set, other hosts may be different result.
  // 2. fetch the variable from synchronized value set, the variable must be a
  //    public or private to current caller.
  NdArrayRef hostGetVar(const std::string &name) const;

  SymbolTable &deviceSymbols() { return symbols_; }

  //
  void deviceSetVar(const std::string &name, const spu::Value &var);

  //
  spu::Value deviceGetVar(const std::string &name) const;

  //
  bool deviceHasVar(const std::string &name) const;

  void sync();
};

}  // namespace spu::device
