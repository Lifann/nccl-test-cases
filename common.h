#ifndef COMMON_H_
#define COMMON_H_

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <chrono>

#include "cuda_runtime.h"
#include "nccl.h"

#define now() std::chrono::high_resolution_clock::now()
#define time_diff(start, end)  \
 std::chrono::duration<double> (end - start).count()

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define COMM_VOLUMN 1024llu * 1024llu * 1024llu * 2 // 2G
#define DUTY_RATIO 20

size_t size_of_type(ncclDataType_t dtype) {
  switch (dtype) {
   case ncclInt8:
   case ncclUint8:
    return 1;
   case ncclFloat16:
   //case ncclBfloat16:
    return 2;
   case ncclInt32:
   case ncclUint32:
   case ncclFloat32:
    return 4;
   case ncclInt64:
   case ncclUint64:
   case ncclFloat64:
    return 8;
  }
  return 0;
}

void allocate_buffer(size_t size, void** host_buf, void** dev_buf) {
  *host_buf = malloc(size);
  if (!*host_buf) {
    printf("Failed to set host buf %llu\n", host_buf);
    exit(0);
  }
  memset(*host_buf, 0, size);
  CUDACHECK(cudaMalloc(dev_buf, size));
  CUDACHECK(cudaMemset(*dev_buf, 0, size));
  printf("Allocate %llu bytes on host and device.\n", size);
}

__global__ void memset_fp32_kernel(float* data, size_t size, float value) {
  int nblocks = gridDim.x * gridDim.y * gridDim.z;
  int block_id = blockIdx.x * gridDim.y * gridDim.z
               + blockIdx.y * gridDim.z
               + blockIdx.z;
  int nBlockResidue = size % nblocks;
  int nPerBlock = size / nblocks;
  int nCurBlock = nPerBlock + nBlockResidue * (block_id == nblocks - 1);

  int block_offset = block_id * nPerBlock;

  int nthreads = blockDim.x * blockDim.y * blockDim.z;
  int thread_id = threadIdx.x * blockDim.y * blockDim.z
                + threadIdx.y * blockDim.z
                + threadIdx.z;
  int thread_offset = block_offset + thread_id;

  int i = 0;
  for (; i < nCurBlock ; i += nthreads) {
    data[thread_offset + i] = value;
  }
}

__global__ void memset_i8_kernel(char* data, size_t size, char value) {
  int nblocks = gridDim.x * gridDim.y * gridDim.z;
  int block_id = blockIdx.x * gridDim.y * gridDim.z
               + blockIdx.y * gridDim.z
               + blockIdx.z;
  int nBlockResidue = size % nblocks;
  int nPerBlock = size / nblocks;
  int nCurBlock = nPerBlock + nBlockResidue * (block_id == nblocks - 1);

  int block_offset = block_id * nPerBlock;

  int nthreads = blockDim.x * blockDim.y * blockDim.z;
  int thread_id = threadIdx.x * blockDim.y * blockDim.z
                + threadIdx.y * blockDim.z
                + threadIdx.z;
  int thread_offset = block_offset + thread_id;

  int i = 0;
  for (; i < nCurBlock ; i += nthreads) {
    data[thread_offset + i] = value;
  }
}

__global__ void memset_u32_kernel(uint32_t* data, size_t size, uint32_t value) {
  int nblocks = gridDim.x * gridDim.y * gridDim.z;
  int block_id = blockIdx.x * gridDim.y * gridDim.z
               + blockIdx.y * gridDim.z
               + blockIdx.z;
  int nBlockResidue = size % nblocks;
  int nPerBlock = size / nblocks;
  int nCurBlock = nPerBlock + nBlockResidue * (block_id == nblocks - 1);

  int block_offset = block_id * nPerBlock;

  int nthreads = blockDim.x * blockDim.y * blockDim.z;
  int thread_id = threadIdx.x * blockDim.y * blockDim.z
                + threadIdx.y * blockDim.z
                + threadIdx.z;
  int thread_offset = block_offset + thread_id;

  int i = 0;
  for (; i < nCurBlock ; i += nthreads) {
    data[thread_offset + i] = value;
  }
}

#endif // COMMON_H_

