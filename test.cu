#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"

#include "common.h"

//#define DEBUG

typedef struct AllreduceContext {
  ncclUniqueId nccl_id;
  ncclComm_t comm;
  int mpi_size;
  int mpi_rank;
  int dev;
} AllreduceContext;

void initialize_allreduce_context(AllreduceContext* context);
void finalize_allreduce_context(AllreduceContext* context);

int main(int argc, char* argv[]) {

  int i = 0;
  size_t buffer_size = 64 * 32 + 3;

  MPI_Init(&argc, &argv);

  AllreduceContext context;
  initialize_allreduce_context(&context);
  printf("size: %d, rank: %d, nccl_id: %s\n", context.mpi_size, context.mpi_rank,
         context.nccl_id.internal);

  void* host_buf = NULL;
  void* dev_buf = NULL;
  allocate_buffer(buffer_size, &host_buf, &dev_buf);
  CUDACHECK(cudaMemset(dev_buf, context.mpi_rank, buffer_size));
  dim3 grid_config(2,2,2);
  dim3 block_confg(2,2,2);
  float value = (float)(context.mpi_rank) + 1.0f;
  memset_fp32_kernel<<<grid_config, block_confg>>>(
    (float*)(dev_buf), buffer_size / 4, value);
  CUDACHECK(cudaMemcpy(host_buf, dev_buf, buffer_size, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaDeviceSynchronize());

#ifdef DBEUG
  float* host_fp = (float*) host_buf;
  if (context.mpi_rank == 0) {
    for (i = 0; i < buffer_size / 4; i++) {
      if (! (host_fp[i] == value)) {
        perror("Set cuda global mem failed.\n");
        exit(-1);
      }
      printf("%f, ", host_fp[i]);
    }
  }
  printf("check dev_buf ok.\n");
#endif // DEBUG

  size_t data_size = buffer_size / size_of_type(ncclInt32);
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));
  NCCLCHECK(ncclAllReduce(
      dev_buf, dev_buf, buffer_size, ncclFloat32, ncclSum,
      context.comm, stream));
  CUDACHECK(cudaStreamSynchronize(stream));

  CUDACHECK(cudaMemcpy(host_buf, dev_buf, buffer_size, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaDeviceSynchronize());

#ifdef DEBUG
  if (context.mpi_rank == 0) {
    printf("========================\n");
    for (i = 0; i < buffer_size / 4; i++) {
      printf("%f, ", host_fp[i]);
    }
    printf("\n");
  }
#endif // DEBUG

  finalize_allreduce_context(&context);
  MPI_Finalize();
  printf("done\n");
  return 0;
}

void initialize_allreduce_context(AllreduceContext* context) {
  MPI_Comm_size(MPI_COMM_WORLD, &context->mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &context->mpi_rank);
  context->dev = context->mpi_rank;
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
