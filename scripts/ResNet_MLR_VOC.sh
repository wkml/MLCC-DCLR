#!/bin/bash

# Make for COCOAPI
# cd cocoapi/PythonAPI
# make -j8
# cd ../..

cd ..

printFreq=100

mode='SSGRL'
dataset='VOC2007'
prob=1.0
eps=0.02
method='WORD'

post="ResNet-VOC2007-${method}-eps${eps/./_}"

pretrainedModel='/data1/2022_stu/wikim_exp/mlp-pl/data/checkpoint/resnet101.pth'

dataDir='/data1/2022_stu/voc2007/VOCdevkit/VOC2007'
dataVector='/data1/2022_stu/wikim_exp/mlp-pl/data/voc_devkit/VOC2007/voc07_vector.npy'
ckptDir='/data1/2022_stu/wikim_exp/mlp-pl/exp/checkpoint'

resumeModel='None'
# resumeModel='/data1/2022_stu/wikim_exp/mlp-pl/exp/Loss/Checkpoint_Best.pth'
evaluate='False'

epochs=15
startEpoch=0
stepEpoch=8

batchSize=16
lr=1e-05
momentum=0.9
weightDecay=5e-4

cropSize=448
scaleSize=512
workers=8

generateLabelEpoch=3

interBCEWeight=1.0
interBCEMargin=0.95
interDistanceWeight=0.05
interExampleNumber=100

interPrototypeDistanceWeight=0.05
prototypeNumber=10
useRecomputePrototype='True'
computePrototypeEpoch=3

OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python ResNet_MLR.py \
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
    --useRecomputePrototype ${useRecomputePrototype} \
    --computePrototypeEpoch ${computePrototypeEpoch} \
    --dataVector ${dataVector} \
    --dataDir ${dataDir} \
    --ckptDir ${ckptDir}