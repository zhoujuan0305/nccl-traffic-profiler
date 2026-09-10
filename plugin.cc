// SPDX-License-Identifier: Apache-2.0
#include "nccl/profiler.h"
#include "trace_writer.h"
#include <cstring>
#include <deque>
#include <map>
#include <new>
#include <strings.h>
#include <vector>

namespace {
using U64 = unsigned long long;
fine::Writer writer;
std::atomic<U64> next_id{1};
std::atomic<int> current_step{-1};
std::atomic<int> current_microbatch{-1};
std::atomic<int> current_phase{0};

struct Tag {
  U64 id = 0;
  U64 buffer = 0;
  U64 bytes = 0;
  int peer = -1;
  int step = -1;
  int microbatch = -1;
  int phase = 0;
};
std::mutex tag_mutex;
std::map<U64, std::deque<Tag>> tags;

struct Event;
struct Context {
  U64 comm;
  int rank;
  int size;
  std::atomic<U64> live{0};
  std::mutex root_mutex;
  std::vector<Event*> roots;
};

struct Root {
  U64 id = 0;
  U64 tag_id = 0;
  int step = -1;
  int microbatch = -1;
  int phase = 0;
};

struct Event {
  U64 type = 0;
  U64 id = 0;
  U64 parent_id = 0;
  U64 proxy_id = 0;
  U64 start = 0;
  U64 submit = 0;
  U64 bytes = 0;
  U64 buffer = 0;
  U64 count = 0;
  Context* ctx = nullptr;
  Event* parent = nullptr;
  Root root;
  int peer = -1;
  int channel = -1;
  int chunk_step = -1;
  bool send = false;
  std::string function;
  std::string datatype;
  std::string algorithm;
  std::string protocol;
};

Tag take_tag(U64 buffer) {
  std::lock_guard<std::mutex> lock(tag_mutex);
  auto it = tags.find(buffer);
  if (it == tags.end() || it->second.empty()) return Tag{};
  Tag tag = it->second.front();
  it->second.pop_front();
  if (it->second.empty()) tags.erase(it);
  return tag;
}

void emit_api(const Event& e, U64 end) {
  writer.line("{\"type\":\"api\",\"id\":%llu,\"comm\":\"%llu\",\"rank\":%d,"
              "\"kind\":\"%s\",\"function\":\"%s\",\"peer_local\":%d,\"buffer\":%llu,"
              "\"count\":%llu,\"datatype\":\"%s\",\"algorithm\":\"%s\",\"protocol\":\"%s\","
              "\"step\":%d,\"microbatch\":%d,\"phase\":%d,\"tag_id\":%llu,"
              "\"start_ns\":%llu,\"end_ns\":%llu}\n",
              e.id,e.ctx->comm,writer.rank,e.type==ncclProfileP2p?"p2p":"collective",
              e.function.c_str(),e.peer,e.buffer,e.count,e.datatype.c_str(),
              e.algorithm.c_str(),e.protocol.c_str(),e.root.step,e.root.microbatch,
              e.root.phase,e.root.tag_id,e.start,end);
}

void emit_transfer(const Event& e, U64 end) {
  if (!e.send || !e.bytes) return;
  const bool chunk = e.type == ncclProfileProxyStep;
  writer.line("{\"type\":\"%s\",\"id\":%llu,\"parent_id\":%llu,\"api_id\":%llu,"
              "\"proxy_id\":%llu,\"comm\":\"%llu\",\"rank\":%d,\"peer_local\":%d,"
              "\"channel\":%d,\"chunk_step\":%d,\"bytes\":%llu,\"step\":%d,"
              "\"microbatch\":%d,\"phase\":%d,\"tag_id\":%llu,"
              "\"start_ns\":%llu,\"submit_ns\":%llu,\"end_ns\":%llu}\n",
              chunk?"chunk":"proxy",e.id,e.parent_id,e.root.id,e.proxy_id,e.ctx->comm,
              writer.rank,e.peer,e.channel,e.chunk_step,e.bytes,e.root.step,
              e.root.microbatch,e.root.phase,e.root.tag_id,e.start,e.submit,end);
}

ncclResult_t init(void** context, uint64_t comm, int* mask, const char* name,
                  int nodes, int size, int rank, ncclDebugLogger_t) {
  *context = nullptr;
  *mask = 0;
  if (!writer.open()) return ncclSuccess;
  auto* ctx = new (std::nothrow) Context;
  if (!ctx) { ++writer.allocations_failed; return ncclSuccess; }
  ctx->comm = comm; ctx->rank = rank; ctx->size = size;
  writer.line("{\"type\":\"comm_member\",\"comm\":\"%llu\",\"name\":\"%s\","
              "\"local_rank\":%d,\"rank\":%d,\"size\":%d,\"nodes\":%d}\n",
              ctx->comm,fine::escape(name).c_str(),rank,writer.rank,size,nodes);
  *context = ctx;
  *mask = ncclProfileColl | ncclProfileP2p | ncclProfileProxyOp | ncclProfileProxyStep;
  return ncclSuccess;
}

template<class Description>
ncclResult_t start(void* context, void** handle, Description* d) {
  *handle = nullptr;
  auto* ctx = static_cast<Context*>(context);
  if (!ctx || !d) return ncclSuccess;
  const U64 supported = ncclProfileColl | ncclProfileP2p | ncclProfileProxyOp | ncclProfileProxyStep;
  if (!(supported & d->type)) return ncclSuccess;
  auto* e = new (std::nothrow) Event;
  if (!e) { ++writer.allocations_failed; return ncclSuccess; }
  ++ctx->live;
  e->type=d->type; e->id=next_id++; e->ctx=ctx; e->start=fine::now_ns();
  if (d->type == ncclProfileColl || d->type == ncclProfileP2p) {
    // NCCL stops task events after enqueue, before their proxy children start.
    // Keep task handles stable until communicator finalization.
    {
      std::lock_guard<std::mutex> lock(ctx->root_mutex);
      ctx->roots.push_back(e);
    }
    e->root = Root{e->id,0,current_step.load(),current_microbatch.load(),current_phase.load()};
    if (d->type == ncclProfileColl) {
      e->function=fine::escape(d->coll.func); e->count=d->coll.count;
      e->datatype=fine::escape(d->coll.datatype);
      e->algorithm=fine::escape(d->coll.algo); e->protocol=fine::escape(d->coll.proto);
    } else {
      e->function=fine::escape(d->p2p.func); e->count=d->p2p.count;
      e->datatype=fine::escape(d->p2p.datatype); e->peer=d->p2p.peer;
      e->buffer=reinterpret_cast<U64>(d->p2p.buff);
      e->send=d->p2p.func && strcasecmp(d->p2p.func,"Send")==0;
      if (e->send) {
        const Tag tag=take_tag(e->buffer);
        if (tag.id) e->root=Root{e->id,tag.id,tag.step,tag.microbatch,tag.phase};
      }
    }
  } else {
    auto* parent=static_cast<Event*>(d->parentObj);
    if (d->type == ncclProfileProxyOp && d->proxyOp.pid != getpid()) {
      parent=nullptr;
      ++writer.anomalies;
    }
    if (parent) {
      e->parent=parent; e->parent_id=parent->id; e->root=parent->root;
    }
    if (d->type == ncclProfileProxyOp) {
      e->peer=d->proxyOp.peer; e->channel=d->proxyOp.channelId;
      e->send=d->proxyOp.isSend; e->proxy_id=e->id;
    } else if (parent) {
      e->peer=parent->peer; e->channel=parent->channel; e->send=parent->send;
      e->proxy_id=parent->proxy_id; e->chunk_step=d->proxyStep.step;
    }
  }
  *handle=e;
  return ncclSuccess;
}

ncclResult_t stop(void* handle) {
  auto* e=static_cast<Event*>(handle);
  if (!e) return ncclSuccess;
  const U64 end=fine::now_ns();
  const bool task = e->type == ncclProfileColl || e->type == ncclProfileP2p;
  if (task) emit_api(*e,end);
  else emit_transfer(*e,end);
  --e->ctx->live;
  if (!task) delete e;
  return ncclSuccess;
}

ncclResult_t state(void* handle, ncclProfilerEventState_v6_t state,
                   ncclProfilerEventStateArgs_v6_t* args) {
  auto* e=static_cast<Event*>(handle);
  if (!e || e->type!=ncclProfileProxyStep || !e->send) return ncclSuccess;
  if (state!=ncclProfilerProxyStepSendWait || !args) return ncclSuccess;
  if (e->submit) { ++writer.anomalies; return ncclSuccess; }
  e->submit=fine::now_ns(); e->bytes=args->proxyStep.transSize;
  if (e->parent) e->parent->bytes+=e->bytes;
  return ncclSuccess;
}

ncclResult_t finalize(void* context) {
  auto* ctx=static_cast<Context*>(context);
  if (!ctx) return ncclSuccess;
  writer.flush();
  writer.line("{\"type\":\"comm_finalize\",\"comm\":\"%llu\",\"rank\":%d,"
              "\"live_events\":%llu,\"write_failures\":%llu,\"allocation_failures\":%llu,"
              "\"anomalies\":%llu}\n",ctx->comm,writer.rank,ctx->live.load(),
              writer.failures.load(),writer.allocations_failed.load(),writer.anomalies.load());
  writer.flush();
  if (!ctx->live.load()) {
    for (auto* root: ctx->roots) delete root;
    delete ctx;
  }
  return ncclSuccess;
}
}  // namespace

extern "C" {
// Captured at enqueue; proxy callbacks inherit IDs instead of sampling Python state.
void fine_set_context(int step, int microbatch, int phase) {
  current_step=step; current_microbatch=microbatch; current_phase=phase;
}

unsigned long long fine_tag_send(unsigned long long buffer, unsigned long long bytes,
                                 int peer, int step, int microbatch, int phase) {
  const Tag tag{next_id++,buffer,bytes,peer,step,microbatch,phase};
  {
    std::lock_guard<std::mutex> lock(tag_mutex);
    tags[buffer].push_back(tag);
  }
  return tag.id;
}

unsigned long long fine_pending_tags() {
  std::lock_guard<std::mutex> lock(tag_mutex);
  U64 count=0;
  for (const auto& item: tags) count+=item.second.size();
  return count;
}

void fine_flush() { writer.flush(); }

ncclProfiler_v5_t ncclProfiler_v5 = {
  "ofc-fine-v1",init,start<ncclProfilerEventDescr_v5_t>,stop,state,finalize
};
ncclProfiler_v6_t ncclProfiler_v6 = {
  "ofc-fine-v1",init,start<ncclProfilerEventDescr_v6_t>,stop,state,finalize
};
}
