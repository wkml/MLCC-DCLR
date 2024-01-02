# !/bin/bash

cd ..

torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 SSGRL_calibration_ddp.py dataset=COCO model=SSGRL