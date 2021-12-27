#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <iostream>

#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"

#include "common.h"

#define DEBUG

typedef struct AllreduceContext {
  ncclUniqueId nccl_id;
  ncclComm_t comm;
  int mpi_size;
  int mpi_rank;
  int dev;
} AllreduceContext;

void initialize_allreduce_context(AllreduceContext* context);
void finalize_allreduce_context(AllreduceContext* context);

void initialize_allreduce_context(AllreduceContext* context) {
  MPI_Comm_size(MPI_COMM_WORLD, &context->mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &context->mpi_rank);
  context->dev = context->mpi_rank % 2;
  cudaSetDevice(context->dev);
  printf("nccl device: %d\n", context->dev);

  // Create communicator by MPI. 
  if (context->mpi_rank == 0) {
    NCCLCHECK(ncclGetUniqueId(&context->nccl_id));
  }
  MPI_Bcast(&context->nccl_id, sizeof(context->nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

  NCCLCHECK(ncclCommInitRank(&context->comm, context->mpi_size, context->nccl_id, context->mpi_rank));
  printf("Init rank ok\n");
}

void finalize_allreduce_context(AllreduceContext* context) {
  NCCLCHECK(ncclCommDestroy(context->comm));
  printf("Destroy nccl context\n");
}

int main(int argc, char* argv[]) {

  int i = 0;
  size_t buffer_size = COMM_VOLUMN;
  size_t data_size = buffer_size / sizeof(uint32_t);

  MPI_Init(&argc, &argv);

  AllreduceContext context;
  initialize_allreduce_context(&context);
  printf("size: %d, rank: %d, nccl_id: %s\n", context.mpi_size, context.mpi_rank,
         context.nccl_id.internal);
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));
  //CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  void* host_buf = NULL;
  void* dev_buf = NULL;
  allocate_buffer(buffer_size, &host_buf, &dev_buf);
  dim3 grid_config(2,2,2);
  dim3 block_confg(2,2,2);
  uint32_t value = (uint32_t)(context.mpi_rank) + 4;
  memset_u32_kernel<<<grid_config, block_confg>>>(
    (uint32_t*)(dev_buf), data_size, value);
  CUDACHECK(cudaMemcpyAsync(host_buf, dev_buf, buffer_size, cudaMemcpyDeviceToHost, stream));
  CUDACHECK(cudaStreamSynchronize(stream));
  printf("Memset ok\n");

  uint32_t* host_fp = (uint32_t*) host_buf;

  cudaEvent_t event_start, event_stop;
  CUDACHECK(cudaEventCreate(&event_start));
  CUDACHECK(cudaEventCreate(&event_stop));
  CUDACHECK(cudaEventRecord(event_start, stream));
  float cost = 0;

  auto t1 = now();

  printf("Ready to allreduce\n");
  NCCLCHECK(ncclAllReduce(
      dev_buf, dev_buf, data_size / 32 / 2, ncclUint64, ncclBitOr,
      context.comm, stream));
  NCCLCHECK(ncclAllReduce(
      dev_buf, dev_buf, data_size / DUTY_RATIO, ncclUint32, ncclAvg,
      context.comm, stream));
  printf("Wait allreduce stream ...\n");

  CUDACHECK(cudaEventRecord(event_stop, stream));
  CUDACHECK(cudaEventSynchronize(event_stop));
  CUDACHECK(cudaEventElapsedTime(&cost, event_start, event_stop));
  printf("Sparse allreduce event elapse: %f\n", cost);

  CUDACHECK(cudaStreamSynchronize(stream));

  auto t2 = now();
  auto diff = time_diff(t1, t2);
  //printf("Allreduce time cost: %llf\n.", diff);
  std::cout << "Allreduce time cost: " << diff << std::endl;

  CUDACHECK(cudaMemcpyAsync(host_buf, dev_buf, buffer_size, cudaMemcpyDeviceToHost, stream));
  CUDACHECK(cudaStreamSynchronize(stream));
  printf("Allreduce done\n");

  finalize_allreduce_context(&context);
  MPI_Finalize();
  printf("done\n");
  return 0;
}
