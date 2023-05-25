import sys
import time
import logging
import random
import numpy as np
from datetime import datetime

from tensorboardX import SummaryWriter

import torch
from torch import nn
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler 

from model.ResNet_MLR import ResNet

from utils.dataloader import get_graph_and_word_file, get_data_loader
from utils.metrics import AverageMeter, AveragePrecisionMeter, Compute_mAP_VOC2012
from utils.checkpoint import load_pretrained_model, save_checkpoint
from utils.label_smoothing import label_smoothing_tradition, label_smoothing_word
from config import arg_parse, logger, show_args

global bestPrec
bestPrec = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    global bestPrec

    # Argument Parse
    args = arg_parse('SSGRL')
    args.post = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))[:10] + '-' + args.post

    if args.seed is not None:
        print ('* absolute seed: {}'.format(args.seed))
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # Bulid Logger
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    file_path = 'exp/log/{}.log'.format(args.post)
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Show Argument
    show_args(args)

    # Create dataloader
    logger.info("==> Creating dataloader...")
    train_loader, test_loader = get_data_loader(args)
    logger.info("==> Done!\n")

    # Load the network
    logger.info("==> Loading the network...")
    _, WordFile = get_graph_and_word_file(args, train_loader.dataset.changedLabels)
    model = ResNet(classNum=args.classNum, wordFeatures=WordFile)

    if args.pretrainedModel != 'None':
        logger.info("==> Loading pretrained model...")
        model = load_pretrained_model(model, args)

    if args.resumeModel != 'None':
        logger.info("==> Loading checkpoint...")
        checkpoint = torch.load(args.resumeModel, map_location='cpu')
        bestPrec, args.startEpoch = checkpoint['best_mAP'], checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        logger.info("==> Checkpoint Epoch: {0}, mAP: {1}".format(args.startEpoch, bestPrec))

    model.to(device)
    logger.info("==> Done!\n")

    criterion = {'BCEWithLogitsLoss': nn.BCEWithLogitsLoss(reduce=True, size_average=True).to(device)}
    

    for p in model.backbone.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=args.lr)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepEpoch, gamma=0.1)

    if args.evaluate:
        Validate(test_loader, model, criterion, 0, args)
        return
    
    # Running Experiment
    logger.info("Run Experiment...")
    writer = SummaryWriter('exp/summary/{}'.format(str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))[:10] + '-' + args.post))

    for epoch in range(args.startEpoch, args.startEpoch + args.epochs):

        Train(train_loader, model, criterion, optimizer, writer, epoch, args)
        mAP, ACE, ECE, MCE = Validate(test_loader, model, criterion, epoch, args)

        scheduler.step()

        writer.add_scalar('mAP', mAP, epoch)
        writer.add_scalar('ACE', ACE, epoch)
        writer.add_scalar('ECE', ECE, epoch)
        writer.add_scalar('MCE', MCE, epoch)

        isBest, bestPrec = mAP > bestPrec, max(mAP, bestPrec)
        save_checkpoint(args, {'epoch':epoch, 'state_dict':model.state_dict(), 'best_mAP':mAP}, isBest)

        if isBest:
            logger.info('[Best] [Epoch {0}]: Best mAP is {1:.3f}'.format(epoch, bestPrec))

    writer.close()

def Train(train_loader, model, criterion, optimizer, writer, epoch, args):

    model.train()

    loss, loss_base, loss_plus, loss_calibration = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    batch_time, data_time = AverageMeter(), AverageMeter()
    logger.info("=========================================")

    end = time.time()
    for batchIndex, (sampleIndex, input, target, groundTruth) in enumerate(train_loader):
        """
            target = [-1, 0, 1]
            target_ = [0, 1]
        """
        input, target = input.to(device), target.float().to(device)

        # Log time of loading data
        data_time.update(time.time() - end)

        # Forward
        output = model(input)

        # Label Smoothing
        if args.method == 'LS':
            target_ = label_smoothing_tradition(args, groundTruth)

        elif args.method == 'WORD':
            target_ = label_smoothing_word(args, groundTruth, model.wordFeatures)
        else:
            # Non Label Smoothing
            target_ = target.detach().clone().to(device)
            target_[target_ < 0] = 0

        # Loss
        loss_base_ = criterion['BCEWithLogitsLoss'](output, target_)

        loss_plus_ = torch.tensor(0.0).to(device)

        loss_calibration_ = torch.tensor(0.0).to(device)

        loss_ = loss_base_ + loss_plus_ + loss_calibration_

        loss.update(loss_.item(), input.size(0))
        loss_base.update(loss_base_.item(), input.size(0))
        loss_plus.update(loss_plus_.item(), input.size(0))
        loss_calibration.update(loss_calibration_.item(), input.size(0))

        # Backward
        loss_.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Log time of batch
        batch_time.update(time.time() - end)
        end = time.time()

        if batchIndex % args.printFreq == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(f'[Train] [Epoch {epoch}]: [{batchIndex:04d}/{len(train_loader)}] Batch Time {batch_time.avg:.3f} Data Time {data_time.avg:.3f}\n'
                        f'\t\t\t\t\t\tLearn Rate {lr:.6f}\n'
                        f'\t\t\t\t\t\tBase Loss {loss_base.val:.4f} ({loss_base.avg:.4f})\n'
                        f'\t\t\t\t\t\tPlus Loss {loss_plus.val:.4f} ({loss_plus.avg:.4f})\n'
                        f'\t\t\t\t\t\tCalibration Loss {loss_calibration.val:.4f} ({loss_calibration.avg:.4f})')
            sys.stdout.flush()

    writer.add_scalar('Loss', loss.avg, epoch)
    writer.add_scalar('Loss_Base', loss_base.avg, epoch)
    writer.add_scalar('Loss_Plus', loss_plus.avg, epoch)
    writer.add_scalar('Loss_Calibration', loss_calibration.avg, epoch)

def Validate(val_loader, model, criterion, epoch, args):

    model.eval()

    apMeter = AveragePrecisionMeter()
    pred, loss, batch_time, data_time = [], AverageMeter(), AverageMeter(), AverageMeter()
    logger.info("=========================================")

    end = time.time()
    for batchIndex, (sampleIndex, input, target, groundTruth) in enumerate(val_loader):

        input, target = input.to(device), target.float().to(device)
        
        # Log time of loading data
        data_time.update(time.time() - end)

        # Forward
        with torch.no_grad():
            output = model(input)

        target[target < 0] = 0

        # Compute loss and prediction
        loss_ = criterion['BCEWithLogitsLoss'](output, target)
        loss.update(loss_.item(), input.size(0))

        # Change target to [0, 1]
        # target[target < 0] = 0

        apMeter.add(output, target)
        pred.append(torch.cat((output, (target > 0).float()), 1))

        # Log time of batch
        batch_time.update(time.time() - end)
        end = time.time()

        # logger.info information of current batch        
        if batchIndex % args.printFreq == 0:
            logger.info('[Test] [Epoch {0}]: [{1:04d}/{2}] '
                        'Batch Time {batch_time.avg:.3f} Data Time {data_time.avg:.3f} '
                        'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, batchIndex, len(val_loader),
                batch_time=batch_time, data_time=data_time,
                loss=loss))
            sys.stdout.flush()

    pred = torch.cat(pred, 0).cpu().clone().numpy()
    mAP = Compute_mAP_VOC2012(pred, args.classNum)

    averageAP = apMeter.value().mean()
    OP, OR, OF1, CP, CR, CF1 = apMeter.overall()
    OP_K, OR_K, OF1_K, CP_K, CR_K, CF1_K = apMeter.overall_topk(3)
    ACE, ECE, MCE = apMeter.calibration()
    mACE, mECE, mMCE = apMeter.compute_classwise()

    logger.info(f'[Test] mAP: {mAP:.3f}, averageAP: {averageAP:.3f}\n'
                f'\t\t\t\t(Compute with all label) OP: {OP:.3f}, OR: {OR:.3f}, OF1: {OF1:.3f}, CP: {CP:.3f}, CR: {CR:.3f}, CF1:{CF1:.3f}\n'
                f'\t\t\t\t(Compute with top-3 label) OP: {OP_K:.3f}, OR: {OR_K:.3f}, OF1: {OF1_K:.3f}, CP: {CP_K:.3f}, CR: {CR_K:.3f}, CF1: {CF1_K:.3f}\n'
                f'\t\t\t\tACE:{ACE:.6f}, ECE:{ECE:.6f}, MCE:{MCE:.6f}\n'
                f'\t\t\t\tmACE:{mACE:.6f}, mECE:{mECE:.6f}, mMCE:{mMCE:.6f}')

    return mAP, ACE, ECE, MCE


if __name__=="__main__":
    main()