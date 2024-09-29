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
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/factory.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::device {

IoClient::IoClient(size_t world_size, const RuntimeConfig &config)
    : world_size_(world_size), config_(makeFullRuntimeConfig(config)) {
  base_io_ = mpc::Factory::CreateIO(config_, world_size_);
}

size_t IoClient::getShareSize(const PtBufferView &bv, Visibility vtype,
                              int owner_rank) {
  if (bv.pt_type == PT_I1 && vtype == VIS_SECRET &&
      base_io_->hasBitSecretSupport()) {
    return base_io_->getBitSecretShareSize(bv.shape.numel());
  } else {
    auto getTypeSize = [&](Visibility vtype, PtType pttype, int owner_rank) {
      if (vtype == VIS_PUBLIC) {
        return SizeOf(pttype);
      }
      return base_io_->getShareType(vtype, pttype, owner_rank).size();
    };

    // decomplex
    if (bv.pt_type == PT_CF64) {
      return 2 * getTypeSize(vtype, PT_F64, owner_rank) * bv.shape.numel();
    } else if (bv.pt_type == PT_CF32) {
      return 2 * getTypeSize(vtype, PT_F32, owner_rank) * bv.shape.numel();
    }
    return getTypeSize(vtype, bv.pt_type, owner_rank) * bv.shape.numel();
  }
}

std::vector<spu::MemRef> IoClient::makeShares(const PtBufferView &bv,
                                              Visibility vtype,
                                              int owner_rank) {
  const size_t fxp_bits = config_.fxp_fraction_bits();
  SPU_ENFORCE(fxp_bits != 0, "fxp should never be zero, please check default");

  if (bv.pt_type == PT_I1 && vtype == VIS_SECRET &&
      base_io_->hasBitSecretSupport()) {
    auto shares = base_io_->makeBitSecret(bv);
    SPU_ENFORCE(shares.size() == world_size_);

    std::vector<spu::MemRef> result;
    result.reserve(world_size_);
    for (const auto &share : shares) {
      result.emplace_back(share);
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

    std::vector<spu::MemRef> result;
    result.reserve(world_size_);
    const auto ty = r_shares.front().eltype();
    for (size_t idx = 0; idx < world_size_; ++idx) {
      // encode to ring.
      auto buf =
          std::make_shared<yacl::Buffer>(bv.shape.numel() * ty.size() * 2);
      Strides strides = makeCompactStrides(bv.shape);
      for (auto &s : strides) {
        s *= 2;
      }
      MemRef real_encoded(buf, ty, bv.shape, strides, 0);
      MemRef imag_encoded(buf, ty, bv.shape, strides, ty.size());

      for (int64_t j = 0; j < bv.shape.numel(); ++j) {
        std::memcpy((void *)&real_encoded.at(j), (void *)&r_shares[idx].at(j),
                    ty.size());
        std::memcpy((void *)&imag_encoded.at(j), (void *)&i_shares[idx].at(j),
                    ty.size());
      }

      result.emplace_back(buf, ty, bv.shape, true);
    }
    return result;
  }

  std::vector<MemRef> shares;

  if (vtype == VIS_PUBLIC) {
    auto pt_seman_type = GetPlainTextSemanticType(bv.pt_type);
    MemRef copied(std::make_shared<yacl::Buffer>(bv.copyBuffer()),
                  makeType<mpc::Pub2kTy>(pt_seman_type), bv.shape);
    shares = std::vector<MemRef>(world_size_, copied);
  } else {
    // encode to ring.
    auto encoded_type = GetEncodedType(bv.pt_type, config_.protocol().field());
    MemRef encoded(makeType<RingTy>(encoded_type, SizeOf(encoded_type) * 8),
                   bv.shape);
    encodeToRing(bv, encoded, fxp_bits);
    // make shares.
    if (!config_.experimental_enable_colocated_optimization()) {
      owner_rank = -1;
    }
    shares = base_io_->toShares(encoded, vtype, owner_rank);
  }

  return shares;
}

void IoClient::combineShares(const std::vector<MemRef> &values,
                             PtBufferView *out) {
  SPU_ENFORCE(out->pt_type != PT_INVALID);
  SPU_ENFORCE(values.size() == world_size_,
              "wrong number of shares, got={}, expect={}", values.size(),
              world_size_);
  if (values.front().isComplex()) {
    Strides ds = out->strides;
    for (auto &s : ds) {
      s *= 2;
    }

    auto s_type = out->pt_type == PT_CF32 ? PT_F32 : PT_F64;
    auto offset = s_type == PT_F32 ? sizeof(float) : sizeof(double);

    PtBufferView real_pv(out->ptr, s_type, out->shape, ds);
    PtBufferView imag_pv(static_cast<std::byte *>(out->ptr) + offset, s_type,
                         out->shape, ds);
    {
      std::vector<MemRef> reals(values.size());
      std::vector<MemRef> imags(values.size());
      for (size_t idx = 0; idx < values.size(); ++idx) {
        const auto &v = values[idx];
        auto new_strides = v.strides();
        for (auto &s : new_strides) {
          s *= 2;
        }
        reals[idx] =
            MemRef(v.buf(), v.eltype(), v.shape(), new_strides, v.offset());
        imags[idx] = MemRef(v.buf(), v.eltype(), v.shape(), new_strides,
                            v.offset() + v.eltype().size());
      }
      combineShares(reals, &real_pv);
      combineShares(imags, &imag_pv);
    }
    return;
  }

  const size_t fxp_bits = config_.fxp_fraction_bits();
  SPU_ENFORCE(fxp_bits != 0, "fxp should never be zero, please check default");

  // get all flatten shares.
  if (values.front().eltype().isa<mpc::Pub2kTy>()) {
    const auto &first_share = values.front();
    for (int64_t idx = 0; idx < out->shape.numel(); ++idx) {
      std::memcpy((void *)&out->get(idx), (void *)&first_share.at(idx),
                  SizeOf(out->pt_type));
    }
  } else {
    // reconstruct to ring buffer.
    auto encoded = base_io_->fromShares(values);
    // decode from ring.
    decodeFromRing(encoded, *out, fxp_bits);
  }
}

ColocatedIo::ColocatedIo(SPUContext *sctx) : sctx_(sctx) {}

void ColocatedIo::hostSetVar(const std::string &name, const PtBufferView &bv,
                             Visibility vtype) {
  unsynced_[name] = {bv.copyBuffer(), bv.pt_type, vtype, bv.shape,
                     static_cast<int>(sctx_->lctx()->Rank())};
}

yacl::Buffer ColocatedIo::hostGetVar(const std::string &name) const {
  const auto itr = unsynced_.find(name);
  if (itr != unsynced_.end()) {
    return itr->second.arr;
  }

  const auto &[v, pt_type] = symbols_.getVar(name);

  if (v.isPublic()) {
    yacl::Buffer dst(v.shape().numel() * SizeOf(pt_type));
    PtBufferView pv(static_cast<void *>(dst.data()), pt_type, v.shape(),
                    makeCompactStrides(v.shape()));
    if (pt_type == PT_F16 || pt_type == PT_F32 || pt_type == PT_F64) {
      kernel::hal::_decode_fp(sctx_, v, &pv, sctx_->getFxpBits());
    } else {
      kernel::hal::_decode_int(sctx_, v, &pv);
    }
    return dst;
  } else if (v.isSecret()) {
    SPU_THROW("not implemented");
    // TODO: test the secret's owner is self,
    // - if yes, reconstruct it
    // - else raise an error.
  } else {
    SPU_THROW("invalid value {}", v);
  }
}

void ColocatedIo::deviceSetVar(const std::string &name, const spu::MemRef &var,
                               PtType pt_type) {
  symbols_.setVar(name, var, pt_type);
}

spu::MemRef ColocatedIo::deviceGetVar(const std::string &name) const {
  return symbols_.getVar(name).first;
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

using SymbolTableProto =
    std::unordered_map<std::string, std::pair<ValueProto, PtType>>;

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
      lctx->SendAsync(idx, value.first.meta.SerializeAsString(),
                      "all2all_var_meta");
      // send chunks count
      lctx->SendAsync(idx, std::to_string(value.first.chunks.size()),
                      "all2all_var_chunks_count");
      // send var dtype
      lctx->SendAsync(idx, PtTypeToPyFormat(value.second), "all2all_var_dtype");
      for (const auto &s : value.first.chunks) {
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
      PtType pt_type;
      {
        auto data = lctx->Recv(idx, "all2all_var_dtype");
        pt_type = PyFormatToPtType(
            std::string(static_cast<const char *>(data.data()), data.size()));
      }
      for (size_t s_idx = 0; s_idx < chunk_count; s_idx++) {
        auto data = lctx->Recv(idx, "all2all_var_chunk");
        SPU_ENFORCE(
            proto.chunks[s_idx].ParseFromArray(data.data(), data.size()));
      }
      st_proto.insert(
          {std::string(static_cast<const char *>(key.data()), key.size()),
           {std::move(proto), pt_type}});
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

    PtBufferView bv(arr.data(), priv.pt_type, priv.shape,
                    makeCompactStrides(priv.shape));

    auto shares = io.makeShares(bv, priv.vtype, priv.owner_rank);
    SPU_ENFORCE(shares.size() == lctx->WorldSize());

    for (size_t idx = 0; idx < shares.size(); idx++) {
      shares_per_party[idx].insert(
          {name, {shares[idx].toProto(128UL * 1024 * 1024), priv.pt_type}});
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
      const auto &val = spu::MemRef::fromProto(proto.first);
      symbols_.setVar(name, val, proto.second);
    }
  }

  unsynced_.clear();
}

}  // namespace spu::device
