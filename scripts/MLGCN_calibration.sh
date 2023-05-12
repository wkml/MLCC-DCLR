#!/bin/bash

# Make for COCOAPI
# cd cocoapi/PythonAPI
# make -j8
# cd ../..

cd ..

printFreq=800

mode='SSGRL'
dataset='COCO2014'
prob=1.0
eps=0.05
method='MPC'

post="MLGCN-${method}-eps${eps/./_}"

pretrainedModel='/data1/2022_stu/wikim_exp/mlp-pl/data/checkpoint/resnet101.pth'

dataDir='/data1/2022_stu/COCO_2014'
dataCategoryMap='/data1/2022_stu/wikim_exp/mlp-pl/data/coco/category.json'
dataVector='/data1/2022_stu/wikim_exp/mlp-pl/data/coco/vectors.npy'
ckptDir='/data1/2022_stu/wikim_exp/mlp-pl/exp/checkpoint'

# resumeModel='None'
resumeModel='/data1/2022_stu/wikim_exp/mlp-pl/exp/mix/Checkpoint_Best_mix.pth'
evaluate='False'

epochs=30
startEpoch=0
stepEpoch=15

batchSize=16
lr=0.1
momentum=0.9
weightDecay=0

cropSize=448
scaleSize=512
workers=8

generateLabelEpoch=1

interBCEWeight=1.0
interBCEMargin=0.95
interDistanceWeight=0.05
interExampleNumber=100

interPrototypeDistanceWeight=0.05
prototypeNumber=10
seed=1
lrp=0.1

OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python MLGCN_calibration.py \
    --post ${post} \
    --printFreq ${printFreq} \
    --mode ${mode} \
    --dataset ${dataset} \
    --prob ${prob} \
    --eps ${eps} \
    --method ${method} \
    --pretrainedModel ${pretrainedModel} \
    --resumeModel ${resumeModel} \
    --evaluate ${evaluate} \
    --epochs ${epochs} \
    --startEpoch ${startEpoch} \
    --stepEpoch ${stepEpoch} \
    --batchSize ${batchSize} \
    --lr ${lr} \
    --momentum ${momentum} \
    --weightDecay ${weightDecay} \
    --cropSize ${cropSize} \
    --scaleSize ${scaleSize} \
    --workers ${workers} \
    --generateLabelEpoch ${generateLabelEpoch} \
    --interBCEWeight ${interBCEWeight} \
    --interBCEMargin ${interBCEMargin} \
    --interDistanceWeight ${interDistanceWeight} \
    --interExampleNumber ${interExampleNumber} \
    --interPrototypeDistanceWeight ${interPrototypeDistanceWeight} \
    --prototypeNumber ${prototypeNumber} \
    --dataVector ${dataVector} \
    --dataCategoryMap ${dataCategoryMap} \
    --dataDir ${dataDir} \
    --ckptDir ${ckptDir}