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

#include "spu/device/io.h"

#include "spu/core/encoding.h"
#include "spu/hal/constants.h"
#include "spu/mpc/factory.h"

namespace spu::device {

IoClient::IoClient(size_t world_size, RuntimeConfig config)
    : world_size_(world_size), config_(std::move(config)) {
  base_io_ =
      mpc::Factory::CreateIO(config_.protocol(), config_.field(), world_size_);
}

std::vector<hal::Value> IoClient::makeShares(PtBufferView bv,
                                             Visibility vtype) {
  // FIXME(jint), this should be in the io context.
  const size_t fxp_bits = getDefaultFxpBits(config_);
  YASL_ENFORCE(fxp_bits != 0, "fxp should never be zero, please check default");

  // encode to ring.
  DataType dtype;
  NdArrayRef encoded =
      encodeToRing(xt_to_ndarray(bv), config_.field(), fxp_bits, &dtype);

  // make shares.
  std::vector<NdArrayRef> shares;
  {
    auto flat_shares = base_io_->toShares(flatten(encoded), vtype);
    YASL_ENFORCE(flat_shares.size() == world_size_);
    shares.reserve(world_size_);
    for (const auto &flat_share : flat_shares) {
      shares.push_back(unflatten(flat_share, encoded.shape()));
    }
  }

  // build value.
  std::vector<hal::Value> result;
  result.reserve(world_size_);
  for (size_t idx = 0; idx < world_size_; idx++) {
    result.emplace_back(shares[idx], dtype);
  }
  return result;
}

NdArrayRef IoClient::combineShares(absl::Span<hal::Value const> values) {
  YASL_ENFORCE(values.size() == world_size_,
               "wrong number of shares, got={}, expect={}", values.size(),
               world_size_);

  // FIXME(jint), this should be in the io context.
  const size_t fxp_bits = getDefaultFxpBits(config_);
  YASL_ENFORCE(fxp_bits != 0, "fxp should never be zero, please check default");

  // reconstruct to ring buffer.
  NdArrayRef encoded;
  {
    // get all flatten shares.
    std::vector<ArrayRef> flat_shares;
    for (const auto &val : values) {
      flat_shares.push_back(flatten(val.data()));
    }

    ArrayRef flat_encoded = base_io_->fromShares(flat_shares);
    encoded = unflatten(flat_encoded, values.at(0).shape());
  }

  // decode from ring.
  const DataType dtype = values.front().dtype();
  return decodeFromRing(encoded, dtype, fxp_bits);
}

ColocatedIo::ColocatedIo(HalContext *hctx) : hctx_(hctx) {}

void ColocatedIo::hostSetVar(const std::string &name, PtBufferView bv,
                             Visibility vtype) {
  unsynced_[name] = {xt_to_ndarray(bv), vtype};
}

NdArrayRef ColocatedIo::hostGetVar(const std::string &name) const {
  const auto itr = unsynced_.find(name);
  if (itr != unsynced_.end()) {
    return itr->second.arr;
  }

  const hal::Value &v = symbols_.getVar(name);

  if (v.isPublic()) {
    return dump_public(hctx_, v);
  } else if (v.isSecret()) {
    YASL_THROW("not implemented");
    // TODO: test the secret's owner is self,
    // - if yes, reconstruct it
    // - else raise an error.
  } else {
    YASL_THROW("invalid value {}", v);
  }
}

void ColocatedIo::deviceSetVar(const std::string &name, const hal::Value &var) {
  symbols_.setVar(name, var);
}

hal::Value ColocatedIo::deviceGetVar(const std::string &name) const {
  return symbols_.getVar(name);
}

bool ColocatedIo::deviceHasVar(const std::string &name) const {
  return symbols_.hasVar(name);
}

// Before all2all
//   Alice: {x0, x1, x2}
//   Bob:   {y0, y1, y2}
//   Carol: {z0, z1, z2}
// After:
//   Alice: {x0, y0, z0}
//   Bob:   {x1, y1, z1}
//   Carol: {x2, y2, z2}
static std::vector<SymbolTableProto>
all2all(const std::shared_ptr<yasl::link::Context> &lctx,
        const std::vector<SymbolTableProto> &rows) {
  // TODO: implement all2all in yasl::link
  for (size_t idx = 0; idx < lctx->WorldSize(); idx++) {
    if (idx == lctx->Rank()) {
      continue;
    }
    yasl::Buffer buf;
    buf.resize(rows[idx].ByteSizeLong());
    YASL_ENFORCE(rows[idx].SerializeToArray(buf.data(), buf.size()));
    lctx->SendAsync(idx, std::move(buf), "all2all");
  }

  std::vector<SymbolTableProto> cols;
  for (size_t idx = 0; idx < lctx->WorldSize(); idx++) {
    if (idx == lctx->Rank()) {
      cols.push_back(rows[idx]);
      continue;
    }
    auto data = lctx->Recv(idx, "all2all");
    SymbolTableProto vars;
    YASL_ENFORCE(vars.ParseFromArray(data.data(), data.size()));
    cols.push_back(std::move(vars));
  }

  return cols;
}

void ColocatedIo::sync() {
  // TODO: optimize this.
  //
  // Intuition, if the input-provider is colocated with runtime, we can send
  // less information to other hosts. i.e.
  //
  // For additive share, we can let the data owner's share as origin value, and
  // other parties as zero.
  //   P0  P1  P2
  //   x   0   0
  //
  // For replicated share from P0, we can let x3 to be zero, and send x1, x2 to
  // P2, P1 respectively, the communication will be halved.
  //   P0  P1  P2
  //   x1      x1
  //   x2  x2
  //       0   0
  //
  // Currently, we implement the naive method, that is, using hal link context
  // to send all shares to others.

  const auto &lctx = hctx_->lctx();

  IoClient io(lctx->WorldSize(), hctx_->rt_config());
  std::vector<SymbolTableProto> shares_per_party(lctx->WorldSize());
  for (const auto &[name, priv] : unsynced_) {
    const auto &arr = priv.arr;
    YASL_ENFORCE(arr.eltype().isa<PtTy>(), "unsupported type={}", arr.eltype());

    PtBufferView bv(arr.data(), arr.eltype().as<PtTy>()->pt_type(), arr.shape(),
                    arr.strides());

    auto shares = io.makeShares(bv, priv.vtype);
    YASL_ENFORCE(shares.size() == lctx->WorldSize());

    for (size_t idx = 0; idx < shares.size(); idx++) {
      shares_per_party[idx].mutable_symbols()->insert(
          {name, shares[idx].toProto()});
    }
  }

  std::vector<SymbolTableProto> values_per_party =
      all2all(lctx, shares_per_party);

  std::set<std::string> all_names;
  for (const auto &values : values_per_party) {
    for (const auto &[name, _] : values.symbols()) {
      YASL_ENFORCE(all_names.find(name) == all_names.end(),
                   "name duplicated {}", name);
      all_names.insert(name);
    }
  }

  for (const auto &values : values_per_party) {
    for (const auto &[name, proto] : values.symbols()) {
      symbols_.setVar(name, hal::Value::fromProto(proto));
    }
  }

  unsynced_.clear();
}

} // namespace spu::device
