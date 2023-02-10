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

#include "libspu/psi/core/communication.h"

#include "spdlog/spdlog.h"

#include "libspu/core/prelude.h"

#include "interconnection/algos/psi.pb.h"

namespace spu::psi {

std::shared_ptr<yacl::link::Context> CreateP2PLinkCtx(
    const std::string& id_prefix,
    const std::shared_ptr<yacl::link::Context>& link_ctx, size_t peer_rank) {
  if (link_ctx->WorldSize() > 2) {
    // build subworld link
    auto peer_id = link_ctx->PartyIdByRank(peer_rank);
    auto self_id = link_ctx->PartyIdByRank(link_ctx->Rank());
    std::vector<std::string> party_ids({peer_id, self_id});
    std::sort(party_ids.begin(), party_ids.end());

    size_t a_rank = std::min(link_ctx->Rank(), peer_rank);
    size_t b_rank = std::max(link_ctx->Rank(), peer_rank);
    std::string sub_id = fmt::format("{}-{}-{}", id_prefix, a_rank, b_rank);

    auto ctx = link_ctx->SubWorld(sub_id, party_ids);
    SPDLOG_INFO("create p2p link, id:{}, rank:{}", ctx->Id(), ctx->Rank());

    return ctx;
  } else {
    return link_ctx;
  }
}

yacl::Buffer IcPsiBatchSerializer::Serialize(PsiDataBatch&& batch) {
  org::interconnection::algos::psi::EcdhPsiCipherBatch proto;
  proto.set_type(batch.type);
  proto.set_batch_index(batch.batch_index);
  proto.set_is_last_batch(batch.is_last_batch);

  proto.set_count(batch.item_num);
  proto.set_ciphertext(std::move(batch.flatten_bytes));

  yacl::Buffer buf(proto.ByteSizeLong());
  proto.SerializeToArray(buf.data(), buf.size());
  return buf;
}

PsiDataBatch IcPsiBatchSerializer::Deserialize(yacl::ByteContainerView buf) {
  org::interconnection::algos::psi::EcdhPsiCipherBatch proto;
  SPU_ENFORCE(proto.ParseFromArray(buf.data(), buf.size()),
              "parse EcdhPsiCipherBatch proto fail");

  PsiDataBatch batch;
  batch.item_num = proto.count();
  batch.flatten_bytes = std::move(*proto.mutable_ciphertext());
  batch.is_last_batch = proto.is_last_batch();

  batch.type = proto.type();
  batch.batch_index = proto.batch_index();
  return batch;
}

}  // namespace spu::psi
