# !/bin/bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python dclr/train/SSGRL_calibration.py dataset=COCO model=SSGRL