import sys
import time
import random
import numpy as np
from datetime import datetime

from loguru import logger
import hydra
from omegaconf import DictConfig, OmegaConf

from tensorboardX import SummaryWriter

import numpy as np
import torch
import torch.optim

from model.SSGRL import SSGRL, update_feature, compute_prototype
from model.MLGCN import gcn_resnet101
from loss import InstanceContrastiveLoss, PrototypeContrastiveLoss

from utils.dataloader import get_graph_and_word_file, get_data_loader
from utils.metrics import AverageMeter, AveragePrecisionMeter, Compute_mAP_VOC2012
from utils.checkpoint import save_checkpoint
from utils.label_smoothing import label_smoothing_tradition, label_smoothing_dynamic

import warnings

warnings.filterwarnings('ignore')


global best_prec
best_prec = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@hydra.main(version_base=None, config_path='./config/', config_name="config")
def main(cfg: DictConfig):
    global best_prec

    # Argument Parse
    # cfg = arg_parse('MLGCN')
    cfg.post = f"{cfg.model.name}-{cfg.dataset.name}-{cfg.model.method}-eps{cfg.model.eps}".replace('.', '_')
    cfg.post = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))[:10] + '-' + cfg.post

    if cfg.seed is not None:
        logger.info('absolute seed: {}'.format(cfg.seed))
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)

    # Bulid Logger
    logger.add('exp/log/{}.log'.format(cfg.post))


    # Show Argument
    logger.info("==========================================")
    logger.info("==========       CONFIG      =============")
    logger.info("==========================================")

    logger.info('\n{}'.format(OmegaConf.to_yaml(cfg)))

    logger.info("==========================================")
    logger.info("===========        END        ============")
    logger.info("==========================================")

    logger.info("\n")

    # Create dataloader
    logger.info("==> Creating dataloader...")
    train_loader, test_loader = get_data_loader(cfg)
    logger.info("==> Done!\n")

    # Load the network
    logger.info("==> Loading the network...")
    graph_file, word_file = get_graph_and_word_file(cfg, train_loader.dataset.changed_labels)
    model = gcn_resnet101(num_classes=cfg.dataset.class_nums, t=0.4, train_label=train_loader.dataset.changed_labels, cfg=cfg)
    aux_model = SSGRL(graph_file, word_file, class_nums=cfg.dataset.class_nums)
    
    if cfg.model.resume_model != 'None':
        logger.info("==> Loading checkpoint...")
        checkpoint = torch.load(cfg.model.resume_model, map_location='cpu')
        best_prec, cfg.start_epoch = checkpoint['best_mAP'], checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        logger.info("==> Checkpoint Epoch: {0}, mAP: {1}".format(cfg.start_epoch, best_prec))

    if cfg.model.aux_model != 'None':
        logger.info("==> Loading auxiliary model...")
        checkpoint = torch.load(cfg.model.aux_model, map_location='cpu')
        aux_model.load_state_dict(checkpoint['state_dict'])

    aux_model.cuda()
    aux_model.eval()

    for p in model.features.parameters():
        p.requires_grad = False

    for p in aux_model.parameters():
        p.requires_grad = False

    model.cuda()

    logger.info("==> Done!\n")

    criterion = {'BCEWithLogitsLoss': torch.nn.MultiLabelSoftMarginLoss(),
                 'InterInstanceDistanceLoss': InstanceContrastiveLoss(cfg.batch_size, reduce=True, size_average=True).cuda(),
                 'InterPrototypeDistanceLoss': PrototypeContrastiveLoss(reduce=True, size_average=True).cuda(),
                 }

    optimizer = torch.optim.SGD(model.get_config_optim(cfg.lr, cfg.model.lrp), 
                                lr=cfg.lr,
                                momentum=cfg.momentum,
                                weight_decay=0)

    if cfg.evaluate:
        Validate(test_loader, model, criterion, 0, cfg)
        return
    
    # Running Experiment
    logger.info("Run Experiment...")
    writer = SummaryWriter('exp/summary/{}'.format(str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))[:10] + '-' + cfg.post))

    if (cfg.model.method == 'DPCAR' or cfg.model.method == 'prototype'):
        logger.info('Compute Prototype...')
        compute_prototype(aux_model, train_loader, cfg)
        logger.info('Done!\n')

    for epoch in range(cfg.start_epoch, cfg.start_epoch + cfg.epochs):

        Train(train_loader, aux_model, model, criterion, optimizer, writer, epoch, cfg)
        mAP, ACE, ECE, MCE = Validate(test_loader, model, criterion, epoch, cfg)

        writer.add_scalar('mAP', mAP, epoch)
        writer.add_scalar('ACE', ACE, epoch)
        writer.add_scalar('ECE', ECE, epoch)
        writer.add_scalar('MCE', MCE, epoch)

        isBest, best_prec = mAP > best_prec, max(mAP, best_prec)
        save_checkpoint(cfg, {'epoch':epoch, 'state_dict':model.state_dict(), 'best_mAP':mAP}, isBest)

        if isBest:
            logger.info('[Best] [Epoch {0}]: Best mAP is {1:.3f}'.format(epoch, best_prec))

    writer.close()

def Train(train_loader, aux_model, gcn_model, criterion, optimizer, writer, epoch, cfg):
    gcn_model.train()

    loss, loss_base, loss_plus, loss_calibration = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    batch_time, data_time = AverageMeter(), AverageMeter()
    logger.info("=========================================")

    end = time.time()
    for batch_index, (sample_index, input, target, full_label, mask) in enumerate(train_loader):

        input, target = input.cuda(), target.float().cuda()

        # Log time of loading data
        data_time.update(time.time() - end)

        # Forward
        output, aux_feature = aux_model(input)
        outputs = gcn_model(input)

        # Label Smoothing
        update_feature_ddp(aux_model, aux_feature, target, cfg.inter_example_nums)
        target_instance = label_smoothing_dynamic(cfg, full_label, aux_model.pos_feature, aux_feature, epoch, 10)
        target_prototype = label_smoothing_dynamic(cfg, full_label, aux_model.prototype, aux_feature, epoch, 10)

        # Loss
        loss_instance = criterion['BCEWithLogitsLoss'](outputs, target_instance)
        loss_prototype = criterion['BCEWithLogitsLoss'](outputs, target_prototype)
        loss_base_ = (loss_instance + loss_prototype) / 2

        loss_plus_ = torch.tensor(0.0).cuda()

        loss_calibration_ = torch.tensor(0.0).cuda()

        loss_ = loss_base_ + loss_plus_ + loss_calibration_

        loss.update(loss_.item(), input.size(0))
        loss_base.update(loss_base_.item(), input.size(0))
        loss_plus.update(loss_plus_.item(), input.size(0))
        loss_calibration.update(loss_calibration_.item(), input.size(0))

        # Backward
        loss_.backward()
        torch.nn.utils.clip_grad_norm_(gcn_model.parameters(), max_norm=cfg.model.max_clip_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        # Log time of batch
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_index % cfg.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(f'[Train] [Epoch {epoch}]: [{batch_index:04d}/{len(train_loader)}] Batch Time {batch_time.avg:.3f} Data Time {data_time.avg:.3f}\n'
                        f'\t\t\t\t\t\tLearn Rate {lr:.6f}\n'
                        f'\t\t\t\t\t\tBase Loss {loss_base.val:.4f} ({loss_base.avg:.4f})\n'
                        f'\t\t\t\t\t\tPlus Loss {loss_plus.val:.4f} ({loss_plus.avg:.4f})\n'
                        f'\t\t\t\t\t\tCalibration Loss {loss_calibration.val:.4f} ({loss_calibration.avg:.4f})')
            sys.stdout.flush()

    writer.add_scalar('Loss', loss.avg, epoch)
    writer.add_scalar('Loss_Base', loss_base.avg, epoch)
    writer.add_scalar('Loss_Plus', loss_plus.avg, epoch)
    writer.add_scalar('Loss_Calibration', loss_calibration.avg, epoch)

def Validate(val_loader, gcn_model, criterion, epoch, cfg):
    gcn_model.eval()

    apMeter = AveragePrecisionMeter()
    pred, loss, batch_time, data_time = [], AverageMeter(), AverageMeter(), AverageMeter()
    logger.info("=========================================")

    end = time.time()
    for batchIndex, (sampleIndex, input, target, groundTruth, mask) in enumerate(val_loader):

        input, target = input.cuda(), target.float().cuda()
        
        # Log time of loading data
        data_time.update(time.time() - end)

        # Forward
        with torch.no_grad():
            outputs = gcn_model(input)
            
        # Compute loss and prediction
        loss_ = criterion['BCEWithLogitsLoss'](outputs, target)
        loss.update(loss_.item(), input.size(0))

        # Change target to [0, 1]
        # target[target < 0] = 0

        apMeter.add(outputs, target)
        pred.append(torch.cat((outputs, (target > 0).float()), 1))

        # Log time of batch
        batch_time.update(time.time() - end)
        end = time.time()

        # logger.info information of current batch        
        if batchIndex % cfg.print_freq == 0:
            logger.info('[Test] [Epoch {0}]: [{1:04d}/{2}] '
                        'Batch Time {batch_time.avg:.3f} Data Time {data_time.avg:.3f} '
                        'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, batchIndex, len(val_loader),
                batch_time=batch_time, data_time=data_time,
                loss=loss))
            sys.stdout.flush()

    pred = torch.cat(pred, 0).cpu().clone().numpy()
    mAP = Compute_mAP_VOC2012(pred, cfg.dataset.class_nums)

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
