/*************************************************************************
 * SPDX-License-Identifier: Apache-2.0
 *************************************************************************/

#ifndef NCCL_TRAFFIC_PROFILER_H_
#define NCCL_TRAFFIC_PROFILER_H_

#include <stdint.h>
#include <stdlib.h>

#include "common.h"
#include "err.h"

enum {
  ncclProfileGroup = (1 << 0),
  ncclProfileColl = (1 << 1),
  ncclProfileP2p = (1 << 2),
  ncclProfileProxyOp = (1 << 3),
  ncclProfileProxyStep = (1 << 4),
  ncclProfileProxyCtrl = (1 << 5),
  ncclProfileKernelCh = (1 << 6),
  ncclProfileNetPlugin = (1 << 7),
  ncclProfileGroupApi = (1 << 8),
  ncclProfileCollApi = (1 << 9),
  ncclProfileP2pApi = (1 << 10),
  ncclProfileKernelLaunch = (1 << 11),
  ncclProfileCeColl = (1 << 12),
  ncclProfileCeSync = (1 << 13),
  ncclProfileCeBatch = (1 << 14),
};

typedef enum {
  ncclProfilerProxyOpSendPosted = 0,
  ncclProfilerProxyOpSendRemFifoWait = 1,
  ncclProfilerProxyOpSendTransmitted = 2,
  ncclProfilerProxyOpSendDone = 3,
  ncclProfilerProxyOpRecvPosted = 4,
  ncclProfilerProxyOpRecvReceived = 5,
  ncclProfilerProxyOpRecvTransmitted = 6,
  ncclProfilerProxyOpRecvDone = 7,
  ncclProfilerProxyStepSendGPUWait = 8,
  ncclProfilerProxyStepSendWait = 9,
  ncclProfilerProxyStepRecvWait = 10,
  ncclProfilerProxyStepRecvFlushWait = 11,
  ncclProfilerProxyStepRecvGPUWait = 12,
  ncclProfilerProxyCtrlIdle = 13,
  ncclProfilerProxyCtrlActive = 14,
  ncclProfilerProxyCtrlSleep = 15,
  ncclProfilerProxyCtrlWakeup = 16,
  ncclProfilerProxyCtrlAppend = 17,
  ncclProfilerProxyCtrlAppendEnd = 18,
  ncclProfilerProxyOpInProgress_v4 = 19,
  ncclProfilerProxyStepSendPeerWait_v4 = 20,
  ncclProfilerNetPluginUpdate = 21,
  ncclProfilerKernelChStop = 22,
  ncclProfilerGroupStartApiStop = 23,
  ncclProfilerGroupEndApiStart = 24,
  ncclProfilerCeCollStart = 25,
  ncclProfilerCeCollComplete = 26,
  ncclProfilerCeSyncStart = 27,
  ncclProfilerCeSyncComplete = 28,
  ncclProfilerCeBatchStart = 29,
  ncclProfilerCeBatchComplete = 30,
} ncclProfilerEventState_t;

typedef ncclProfilerEventState_t ncclProfilerEventState_v1_t;
typedef ncclProfilerEventState_t ncclProfilerEventState_v2_t;
typedef ncclProfilerEventState_t ncclProfilerEventState_v3_t;
typedef ncclProfilerEventState_t ncclProfilerEventState_v4_t;
typedef ncclProfilerEventState_t ncclProfilerEventState_v5_t;
typedef ncclProfilerEventState_t ncclProfilerEventState_v6_t;

#include "profiler_v6.h"

typedef ncclProfiler_v6_t ncclProfiler_t;
typedef ncclProfilerEventDescr_v6_t ncclProfilerEventDescr_t;
typedef ncclProfilerEventStateArgs_v6_t ncclProfilerEventStateArgs_t;

#endif
