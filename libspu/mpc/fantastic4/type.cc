
#include "libspu/mpc/fantastic4/type.h"

#include "libspu/mpc/common/pv2k.h"

namespace spu::mpc::fantastic4 {

void registerTypes() {
  regPV2kTypes();

  static std::once_flag flag;
  std::call_once(flag, []() {
    TypeContext::getTypeContext()->addTypes<AShrTy, BShrTy>();
  });
}

}  // namespace spu::mpc::fantastic4