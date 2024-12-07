#pragma once

#include "libspu/core/context.h"
#include "libspu/core/ndarray_ref.h"
#include "libspu/core/object.h"
#include "yacl/crypto/hash/hash_interface.h"
#include "yacl/link/link.h"
#include "libspu/spu.pb.h"

namespace spu::mpc::fantastic4 {

class Fantastic4MacState : public State {
    std::unique_ptr<yacl::crypto::HashInterface> hash_algo_; 
    size_t mac_len_;
    NdArrayRef send_hashes_(ring2k_t, {4, 4});
    NdArrayRef used_channels_(bool, {4, 4});

 private:
  Fantastic4MacState() = default;
 public:
    static constexpr const char* kBindName() { return "Fantastic4MacState"; }

    explicit Fantastic4MacState(const std::shared_ptr<yacl::link::Context>& lctx) {
        hash_algo_ = std::make_unique<yacl::crypto::Blake3Hash>();
        mac_len_ = 128;
        
    }


}


} // namespace spu::mpc::fantastic4