#export CUDA_VISIBLE_DEVICES="0,1"

mpirun --allow-run-as-root -np 4 -H $NODE_IP_LIST \
    --report-bindings --display-map \
    --mca btl ^openib \
    --mca coll_hcoll_enable 0 --mca coll_fca_enable 0 \
    --mca plm_rsh_no_tree_spawn 1 \
    --mca pml ob1 \
    -x NCCL_IB_GID_INDEX=3 \
    -x NCCL_IB_SL=3 \
    -x NCCL_NET_GDR_READ=1 \
    -x NCCL_NET_GDR_LEVEL=5 \
    -x NCCL_SOCKET_IFNAME=eth1 \
    -x NCCL_DEBUG=INFO \
    -x NCCL_DEBUG_SUBSYS=ALL \
    -x NCCL_SOCKET_NTHREADS=4 \
    -x NCCL_NSOCKS_PERTHREAD=16 \
    -x NCCL_NTHREADS=512 \
    -x NCCL_MAX_NCHANNELS=1 \
    -x NCCL_MIN_NCHANNELS=1 \
    -x NCCL_TOPO_DUMP_FILE="topo" \
    -x NCCL_P2P_LEVEL=0 \
    -x LD_LIBRARY_PATH \
    -x PATH \
    ./benchmark_sparse
