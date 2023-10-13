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

#include "libspu/device/io.h"

#include <utility>

#include "yacl/link/algorithm/allgather.h"

#include "libspu/core/config.h"
#include "libspu/core/encoding.h"
#include "libspu/core/pt_buffer_view.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/mpc/factory.h"

namespace spu::device {

IoClient::IoClient(size_t world_size, const RuntimeConfig &config)
    : world_size_(world_size), config_(makeFullRuntimeConfig(config)) {
  base_io_ = mpc::Factory::CreateIO(config_, world_size_);
}

size_t IoClient::getShareSize(const PtBufferView &bv, Visibility vtype,
                              int owner_rank) {
  if (bv.pt_type == PT_BOOL && vtype == VIS_SECRET &&
      base_io_->hasBitSecretSupport()) {
    return base_io_->getBitSecretShareSize(bv.shape.numel());
  } else {
    return base_io_->getShareType(vtype, owner_rank).size() * bv.shape.numel();
  }
}

std::vector<spu::Value> IoClient::makeShares(const PtBufferView &bv,
                                             Visibility vtype, int owner_rank) {
  const size_t fxp_bits = config_.fxp_fraction_bits();
  SPU_ENFORCE(fxp_bits != 0, "fxp should never be zero, please check default");

  if (bv.pt_type == PT_BOOL && vtype == VIS_SECRET &&
      base_io_->hasBitSecretSupport()) {
    auto shares = base_io_->makeBitSecret(bv);
    SPU_ENFORCE(shares.size() == world_size_);

    std::vector<spu::Value> result;
    result.reserve(world_size_);
    for (const auto &share : shares) {
      result.emplace_back(share, DataType::DT_I1);
    }
    return result;
  }

  if (bv.pt_type == PT_CF32 || bv.pt_type == PT_CF64) {
    auto s_type = bv.pt_type == PT_CF32 ? PT_F32 : PT_F64;
    auto offset = bv.pt_type == PT_CF32 ? sizeof(float) : sizeof(double);

    Strides ds = bv.strides;
    for (auto &s : ds) {
      s *= 2;
    }

    PtBufferView real_view(bv.ptr, s_type, bv.shape, ds);
    PtBufferView imag_view(static_cast<const std::byte *>(bv.ptr) + offset,
                           s_type, bv.shape, ds);

    auto r_shares = makeShares(real_view, vtype, owner_rank);
    auto i_shares = makeShares(imag_view, vtype, owner_rank);

    std::vector<spu::Value> result;
    result.reserve(world_size_);
    for (size_t idx = 0; idx < world_size_; ++idx) {
      result.emplace_back(r_shares[idx].data(), i_shares[idx].data(),
                          r_shares[idx].dtype());
    }
    return result;
  }

  // encode to ring.
  DataType dtype;
  NdArrayRef encoded = encodeToRing(bv, config_.field(), fxp_bits, &dtype);

  // make shares.
  std::vector<NdArrayRef> shares = base_io_->toShares(encoded, vtype);

  // build value.
  std::vector<spu::Value> result;
  result.reserve(world_size_);
  for (size_t idx = 0; idx < world_size_; idx++) {
    result.emplace_back(shares[idx], dtype);
  }
  return result;
}

PtType IoClient::getPtType(absl::Span<spu::Value const> values) {
  const DataType dtype = values.front().dtype();
  if (values.front().isComplex()) {
    if (dtype == DT_F32) {
      return PT_CF32;
    } else {
      SPU_ENFORCE(dtype == DT_F64);
      return PT_CF64;
    }
  } else {
    return getDecodeType(dtype);
  }
}

void IoClient::combineShares(absl::Span<Value const> values,
                             PtBufferView *out) {
  SPU_ENFORCE(values.size() == world_size_,
              "wrong number of shares, got={}, expect={}", values.size(),
              world_size_);

  if (values.front().isComplex()) {
    Strides ds = out->strides;
    for (auto &s : ds) {
      s *= 2;
    }

    auto s_type = values.front().dtype() == DT_F32 ? PT_F32 : PT_F64;
    auto offset =
        values.front().dtype() == DT_F32 ? sizeof(float) : sizeof(double);

    PtBufferView real_pv(out->ptr, s_type, out->shape, ds);
    PtBufferView imag_pv(static_cast<std::byte *>(out->ptr) + offset, s_type,
                         out->shape, ds);
    {
      std::vector<Value> reals(values.size());
      for (size_t idx = 0; idx < values.size(); ++idx) {
        reals[idx] = Value(values[idx].data(), values[idx].dtype());
      }
      combineShares(reals, &real_pv);
    }
    {
      std::vector<spu::Value> imags(values.size());
      for (size_t idx = 0; idx < values.size(); ++idx) {
        imags[idx] = Value(values[idx].imag().value(), values[idx].dtype());
      }
      combineShares(imags, &imag_pv);
    }
    return;
  }

  const size_t fxp_bits = config_.fxp_fraction_bits();
  SPU_ENFORCE(fxp_bits != 0, "fxp should never be zero, please check default");

  // reconstruct to ring buffer.
  NdArrayRef encoded;
  {
    // get all flatten shares.
    std::vector<NdArrayRef> shares;
    for (const auto &val : values) {
      shares.push_back(val.data());
    }

    encoded = base_io_->fromShares(shares);
  }

  // decode from ring.
  const DataType dtype = values.front().dtype();
  decodeFromRing(encoded, dtype, fxp_bits, out);
}

ColocatedIo::ColocatedIo(SPUContext *sctx) : sctx_(sctx) {}

void ColocatedIo::hostSetVar(const std::string &name, const PtBufferView &bv,
                             Visibility vtype) {
  unsynced_[name] = {convertToNdArray(bv), vtype};
}

NdArrayRef ColocatedIo::hostGetVar(const std::string &name) const {
  const auto itr = unsynced_.find(name);
  if (itr != unsynced_.end()) {
    return itr->second.arr;
  }

  const spu::Value &v = symbols_.getVar(name);

  if (v.isPublic()) {
    return kernel::hal::dump_public(sctx_, v);
  } else if (v.isSecret()) {
    SPU_THROW("not implemented");
    // TODO: test the secret's owner is self,
    // - if yes, reconstruct it
    // - else raise an error.
  } else {
    SPU_THROW("invalid value {}", v);
  }
}

void ColocatedIo::deviceSetVar(const std::string &name, const spu::Value &var) {
  symbols_.setVar(name, var);
}

spu::Value ColocatedIo::deviceGetVar(const std::string &name) const {
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

using SymbolTableProto = std::unordered_map<std::string, ValueProto>;

static std::vector<SymbolTableProto> all2all(
    const std::shared_ptr<yacl::link::Context> &lctx,
    const std::vector<SymbolTableProto> &rows) {
  std::vector<size_t> party_var_count;
  {
    const auto party_var_count_str = yacl::link::AllGather(
        lctx, std::to_string(rows[0].size()), "all2all_var_count");

    for (const auto &c : party_var_count_str) {
      size_t count = 0;
      SPU_ENFORCE(absl::SimpleAtoi(c, &count));
      party_var_count.push_back(count);
    }
  }

  for (size_t idx = 0; idx < lctx->WorldSize(); idx++) {
    if (idx == lctx->Rank()) {
      continue;
    }
    for (const auto &[key, value] : rows[idx]) {
      // send var key
      lctx->SendAsync(idx, key, "all2all_var_key");
      // send var meta
      lctx->SendAsync(idx, value.meta.SerializeAsString(), "all2all_var_meta");
      // send chunks count
      lctx->SendAsync(idx, std::to_string(value.chunks.size()),
                      "all2all_var_chunks_count");
      for (const auto &s : value.chunks) {
        // send chunks
        lctx->SendAsync(idx, s.SerializeAsString(), "all2all_var_chunk");
      }
    }
  }

  std::vector<SymbolTableProto> cols;
  for (size_t idx = 0; idx < lctx->WorldSize(); idx++) {
    if (idx == lctx->Rank()) {
      cols.push_back(rows[idx]);
      continue;
    }
    SymbolTableProto st_proto;
    for (size_t msg_idx = 0; msg_idx < party_var_count[idx]; msg_idx++) {
      auto key = lctx->Recv(idx, "all2all_var_key");
      ValueProto proto;
      {
        auto data = lctx->Recv(idx, "all2all_var_meta");
        SPU_ENFORCE(proto.meta.ParseFromArray(data.data(), data.size()));
      }
      size_t chunk_count = 0;
      {
        auto data = lctx->Recv(idx, "all2all_var_chunks_count");
        SPU_ENFORCE(absl::SimpleAtoi(data, &chunk_count));
      }
      proto.chunks.resize(chunk_count);
      for (size_t s_idx = 0; s_idx < chunk_count; s_idx++) {
        auto data = lctx->Recv(idx, "all2all_var_chunk");
        SPU_ENFORCE(
            proto.chunks[s_idx].ParseFromArray(data.data(), data.size()));
      }
      st_proto.insert(
          {std::string(static_cast<const char *>(key.data()), key.size()),
           std::move(proto)});
    }
    cols.push_back(std::move(st_proto));
  }

  return cols;
}

void ColocatedIo::sync() {
  // TODO: optimize this.
  //
  // Intuition, if the input-provider is colocated with runtime, we can send
  // less information to other hosts. i.e.
  //
  // For additive share, we can let the data owner's share as origin value,
  // and other parties as zero.
  //   P0  P1  P2
  //   x   0   0
  //
  // For replicated share from P0, we can let x3 to be zero, and send x1, x2
  // to P2, P1 respectively, the communication will be halved.
  //   P0  P1  P2
  //   x1      x1
  //   x2  x2
  //       0   0
  //
  // Currently, we implement the naive method, that is, using hal link context
  // to send all shares to others.

  const auto &lctx = sctx_->lctx();

  IoClient io(lctx->WorldSize(), sctx_->config());
  std::vector<SymbolTableProto> shares_per_party(lctx->WorldSize());
  for (const auto &[name, priv] : unsynced_) {
    const auto &arr = priv.arr;
    SPU_ENFORCE(arr.eltype().isa<PtTy>(), "unsupported type={}", arr.eltype());

    PtBufferView bv(arr.data(), arr.eltype().as<PtTy>()->pt_type(), arr.shape(),
                    arr.strides());

    auto shares = io.makeShares(bv, priv.vtype);
    SPU_ENFORCE(shares.size() == lctx->WorldSize());

    for (size_t idx = 0; idx < shares.size(); idx++) {
      shares_per_party[idx].insert(
          {name, shares[idx].toProto(128UL * 1024 * 1024)});
    }
  }

  std::vector<SymbolTableProto> values_per_party =
      all2all(lctx, shares_per_party);

  std::set<std::string> all_names;
  for (const auto &values : values_per_party) {
    for (const auto &[name, _] : values) {
      SPU_ENFORCE(all_names.find(name) == all_names.end(), "name duplicated {}",
                  name);
      all_names.insert(name);
    }
  }

  for (const auto &values : values_per_party) {
    for (const auto &[name, proto] : values) {
      symbols_.setVar(name, spu::Value::fromProto(proto));
    }
  }

  unsynced_.clear();
}

}  // namespace spu::device
