#!/bin/bash

module load anaconda/2020.11
module load cuda/10.2
source activate osvr

cd ./scripts

bash SST-conf.sh