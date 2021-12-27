nvcc -I/usr/local/include -L/usr/local/lib -std=c++11 -arch=compute_70 -code=sm_70 -o benchmark_dense dense_comm_test.cu -lnccl -lmpi

nvcc -I/usr/local/include -L/usr/local/lib -std=c++11 -arch=compute_70 -code=sm_70 -o benchmark_sparse sparse_comm_test.cu -lnccl -lmpi
