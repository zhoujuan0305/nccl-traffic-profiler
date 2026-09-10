/*************************************************************************
 * SPDX-License-Identifier: Apache-2.0
 *************************************************************************/

#ifndef NCCL_TRAFFIC_ERR_H_
#define NCCL_TRAFFIC_ERR_H_

typedef enum {
  ncclSuccess = 0,
  ncclUnhandledCudaError = 1,
  ncclSystemError = 2,
  ncclInternalError = 3,
  ncclInvalidArgument = 4,
  ncclInvalidUsage = 5,
  ncclRemoteError = 6
} ncclResult_t;

#endif
