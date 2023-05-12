import sys
import time
import logging

from tensorboardX import SummaryWriter

import torch
from torch import nn
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler 

from model.SSGRL import SSGRL
from loss.SST import BCELoss, intraAsymmetricLoss, ContrastiveLoss, SeparationLoss
from loss.HST import PrototypeContrastiveLoss, computePrototype
from loss.Calibration import MDCA, FocalLoss, FLSD, DCA, MbLS, DWBL

from utils.dataloader import get_graph_and_word_file, get_data_loader
from utils.metrics import AverageMeter, AveragePrecisionMeter, Compute_mAP_VOC2012
from utils.checkpoint import load_pretrained_model, save_code_file, save_checkpoint
from utils.label_smoothing import label_smoothing_tradition, label_smoothing_dynamic_IST, label_smoothing_dynamic_CST
from config import arg_parse, logger, show_args
from tqdm import tqdm

import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = arg_parse('SSGRL')

train_loader, test_loader = get_data_loader(args)
GraphFile, WordFile = get_graph_and_word_file(args, train_loader.dataset.changedLabels)
model = SSGRL(GraphFile, WordFile, classNum=args.classNum)

if args.pretrainedModel != 'None':
        model = load_pretrained_model(model, args)

if args.resumeModel != 'None':
    print("=====load_model=====")
    checkpoint = torch.load(args.resumeModel, map_location='cpu')
    bestPrec, args.startEpoch = checkpoint['best_mAP'], checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])

model.to(device)

def dynamic_CST(posFeature):
    np.save("instance_feature_woLoss.npy", posFeature.cpu().numpy())
    posFeature = posFeature.mean(axis=1)
    posFeature /= posFeature.norm(dim=-1, keepdim=True)         # [classNum, exampleNum, c]
    probCoOccurrence = posFeature @ posFeature.T                   # [batch, classNum, classNum * exampleNum]
    # print("prococo:", probCoOccurrence.shape)

    np.save("confuse_matrix_woLoss_instance.npy", probCoOccurrence.cpu().numpy())

computePrototype(model, train_loader, args)

for batchIndex, (sampleIndex, input, target, groundTruth) in enumerate(train_loader):
    input, target = input.to(device), target.float().to(device)
    # Log time of loading data
    # Forward
    with torch.no_grad():
        output, intraCoOccurrence, feature = model(input)
    model.updateFeature(feature, target, args.interExampleNumber)

dynamic_CST(model.posFeature)


