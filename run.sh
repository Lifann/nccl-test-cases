NCCL_SOCKET_IFNAME=eth0 \
    NCCL_DEBUG=INFO \
    NCCL_DEBUG_SUBSYS=ALL \
    mpirun --allow-run-as-root -np 4 ./a.out
