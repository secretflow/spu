#include "libspu/mpc/fantastic4/jmp.h"



namespace spu::mpc::fantastic4 {
  size_t PrevRank(size_t rank, size_t world_size){
    return (rank + world_size -1) % world_size;
  }

  size_t OffsetRank(size_t myrank, size_t other, size_t world_size){
    size_t offset = (myrank + world_size -other) % world_size;
    if(offset == 3){
      offset = 1;
    }
    return offset;
  }


}