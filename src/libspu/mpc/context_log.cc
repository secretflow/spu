#include "libspu/mpc/context_log.h"

namespace spu::mpc {
void wrapStartRecorder(SPUContext* sctx) {
  auto lctx = sctx->lctx();
  if (lctx.get()->Rank() == 0) {
    OfflineRecorder::StartRecorder();
  }
}

void wrapStopRecorder(SPUContext* sctx) {
  auto lctx = sctx->lctx();
  if (lctx.get()->Rank() == 0) {
    OfflineRecorder::StopRecorder();
  }
}

void printStatus(SPUContext* sctx) {
  auto lctx = sctx->lctx();
  if (lctx.get()->Rank() == 0) {
    std::cout << "+ Message from hacked function spu::mpc::printStatus." << std::endl;
    size_t comm = sctx->getState<Communicator>()->getStats().comm;
    size_t latency = sctx->getState<Communicator>()->getStats().latency;
    std::cout << "comm: " << comm << std::endl;
    std::cout << "latency: " << latency << std::endl;
    OfflineRecorder::PrintRecord();
  }
}

void clearStatus(SPUContext* sctx) {
  auto lctx = sctx->lctx();
  if (lctx.get()->Rank() == 0) {
    std::cout << "+ Message from hacked function spu::mpc::printStatus." << std::endl;
    std::cout << "Inner status has been cleared." << std::endl;
    OfflineRecorder::ClearRecord();
    size_t comm = sctx->getState<Communicator>()->getStats().comm;
    size_t latency = sctx->getState<Communicator>()->getStats().latency;
    sctx->getState<Communicator>()->addCommStatsManually(-latency, -comm);
  }
}
}  // namespace spu::mpc