
#include "spu/mpc/spdzwisefield/type.h"

#include <mutex>

#include "spu/mpc/common/pub2k.h"

namespace spu::mpc::spdzwisefield {
    void registerTypes() {
        regPub2kTypes();

        static std::once_flag flag;
        std::call_once(flag, []() {
            TypeContext::getTypeContext()->addTypes<AShrTy, BShrTy>();
        });
    }

}  // namespace spu::mpc::spdzwisefield