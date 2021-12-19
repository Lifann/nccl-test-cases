nvcc -I/data/nccl/build/include/ -I/usr/local/openmpi-4.1.1/include -L/data/nccl/build/lib/ -L/usr/local/openmpi-4.1.1/lib test.cu -lnccl -lmpi
