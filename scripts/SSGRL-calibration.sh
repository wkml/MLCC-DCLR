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
method='FLSD'

post="Date0415-SSGRL-${method}-eps${eps/./_}-test"

pretrainedModel='/home/horace/workspace/Wikim/MLP-PL-master/data/checkpoint/resnet101.pth'

dataDir='/home/horace/dataset/COCO2014'
dataCategoryMap='/home/horace/workspace/Wikim/MLP-PL/data/coco/category.json'
dataVector='/home/horace/workspace/Wikim/MLP-PL-master/data/coco/vectors.npy'
resumeModel='None'
evaluate='False'

epochs=20
startEpoch=0
stepEpoch=15

batchSize=4
lr=1e-5
momentum=0.9
weightDecay=5e-4

cropSize=224
scaleSize=512
workers=8

generateLabelEpoch=5

intraBCEWeight=1.0
intraBCEMargin=0.95
intraCooccurrenceWeight=10.0

interBCEWeight=1.0
interBCEMargin=0.95
interDistanceWeight=0.05
interExampleNumber=100

interPrototypeDistanceWeight=0.05
prototypeNumber=10
useRecomputePrototype='True'
computePrototypeEpoch=5

OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python SSGRL_calibration.py \
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
    --intraBCEMargin ${intraBCEMargin} \
    --intraBCEWeight ${intraBCEWeight} \
    --intraCooccurrenceWeight ${intraCooccurrenceWeight} \
    --interBCEWeight ${interBCEWeight} \
    --interBCEMargin ${interBCEMargin} \
    --interDistanceWeight ${interDistanceWeight} \
    --interExampleNumber ${interExampleNumber} \
    --interPrototypeDistanceWeight ${interPrototypeDistanceWeight} \
    --prototypeNumber ${prototypeNumber} \
    --useRecomputePrototype ${useRecomputePrototype} \
    --computePrototypeEpoch ${computePrototypeEpoch} \
    --dataVector ${dataVector} \
    --dataCategoryMap ${dataCategoryMap} \
    --dataDir ${dataDir}