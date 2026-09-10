// Regression: stopped task handles must survive late proxy children.
#include "nccl/profiler.h"
#include <cassert>
#include <unistd.h>

extern "C" ncclProfiler_v6_t ncclProfiler_v6;
extern "C" unsigned long long fine_tag_send(unsigned long long, unsigned long long, int, int, int, int);

int main() {
  auto& p=ncclProfiler_v6;
  void* ctx=nullptr;
  int mask=0;
  p.init(&ctx,123,&mask,"test",2,2,0,nullptr);
  assert(ctx);
  void* tasks[3];
  for(int i=0;i<3;++i) {
    fine_tag_send(4096,2048,1,i+1,0,1);
    ncclProfilerEventDescr_v6_t d{};
    d.type=ncclProfileP2p; d.p2p.func="Send"; d.p2p.buff=reinterpret_cast<void*>(4096);
    d.p2p.count=1024; d.p2p.datatype="ncclBfloat16"; d.p2p.peer=1;
    p.startEvent(ctx,&tasks[i],&d);
    p.stopEvent(tasks[i]);
  }
  for(int i=0;i<3;++i) {
    ncclProfilerEventDescr_v6_t op{};
    op.type=ncclProfileProxyOp; op.parentObj=tasks[i]; op.proxyOp.peer=1;
    op.proxyOp.pid=getpid(); op.proxyOp.isSend=1;
    void* handle=nullptr;
    p.startEvent(ctx,&handle,&op);
    ncclProfilerEventDescr_v6_t chunk{};
    chunk.type=ncclProfileProxyStep; chunk.parentObj=handle;
    void* child=nullptr;
    p.startEvent(ctx,&child,&chunk);
    ncclProfilerEventStateArgs_v6_t state{};
    state.proxyStep.transSize=2048;
    p.recordEventState(child,ncclProfilerProxyStepSendWait,&state);
    p.stopEvent(child);
    p.stopEvent(handle);
  }
  p.finalize(ctx);
}
