BINARIES := benchmark_dense benchmark_sparse
DLINKS := -lnccl -lmpi
NCCL_INC := -I/usr/local/include
NCCL_LIB := -L/usr/local/lib
COPTS := -std=c++11 -arch=compute_70 -code=sm_70
NVCC := nvcc

all: $(BINARIES);

benchmark_dense: dense_comm_test.cu common.h
	$(NVCC) $(NCCL_INC) $(NCCL_LIB) $(COPTS) -o benchmark_dense dense_comm_test.cu $(DLINKS)

benchmark_sparse: sparse_comm_test.cu common.h
	$(NVCC) $(NCCL_INC) $(NCCL_LIB) $(COPTS) -o benchmark_sparse sparse_comm_test.cu $(DLINKS)

clean:
	rm benchmark_sparse benchmark_dense
