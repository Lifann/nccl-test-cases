#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"

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

  int size = 32*1024*1024;
  int dev = 0;

  MPI_Init(&argc, &argv);

  AllreduceContext context;
  initialize_allreduce_context(&context);
  printf("size: %d, rank: %d, nccl_id: %s\n", context.mpi_size, context.mpi_rank, context.nccl_id.internal);

  MPI_Finalize();

  printf("ok\n");
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

  printf("Go init rank\n");
  NCCLCHECK(ncclCommInitRank(&context->comm, context->mpi_size, context->nccl_id, context->mpi_rank));
  printf("Init rank ok\n");
}

