import datetime
import random
import sys
import time
import logging
from datetime import datetime

from tensorboardX import SummaryWriter

import torch
import torch.optim
import torch.backends.cudnn as cudnn

from model.SSGRL import SSGRL
from model.ml_gcn import gcn_resnet101
from loss.SST import ContrastiveLoss, SeparationLoss
from loss.HST import PrototypeContrastiveLoss, computePrototype
from loss.Calibration import MDCA, FocalLoss, FLSD, DCA, MbLS, DWBL

from utils.dataloader import get_graph_and_word_file, get_data_loader
from utils.metrics import AverageMeter, AveragePrecisionMeter, Compute_mAP_VOC2012
from utils.checkpoint import load_pretrained_model, save_checkpoint
from utils.label_smoothing import label_smoothing_tradition, label_smoothing_dynamic_IST, label_smoothing_dynamic_CST
from config import arg_parse, logger, show_args

global bestPrec
bestPrec = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    cudnn.benchmark = True

def main():
    global bestPrec

    # Argument Parse
    args = arg_parse('ML_GCN')

    if args.seed is not None:
        print ('* absolute seed: {}'.format(args.seed))
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # Bulid Logger
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    file_path = 'exp/log/{}.log'.format(str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))[:10] + '-' +args.post)
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
    GraphFile, WordFile = get_graph_and_word_file(args, train_loader.dataset.changedLabels)
    ssgrl_model = SSGRL(GraphFile, WordFile, classNum=args.classNum)

    gcn_model = gcn_resnet101(num_classes=args.classNum, t=0.4, adj_file=args.adjFile, args=args)

    if args.pretrainedModel != 'None':
        logger.info("==> Loading pretrained model...")
        ssgrl_model = load_pretrained_model(ssgrl_model, args)
        # gcn_model = load_pretrained_model(gcn_model, args)

    if args.resumeModel != 'None':
        logger.info("==> Loading checkpoint...")
        checkpoint = torch.load(args.resumeModel, map_location='cpu')
        bestPrec, _ = checkpoint['best_mAP'], checkpoint['epoch']
        ssgrl_model.load_state_dict(checkpoint['state_dict'])
        logger.info("==> Checkpoint Epoch: {0}, mAP: {1}".format(args.startEpoch, bestPrec))

    ssgrl_model.to(device)
    gcn_model.to(device)
    ssgrl_model.eval()
    logger.info("==> Done!\n")

    criterion = {'BCELoss': torch.nn.MultiLabelSoftMarginLoss().to(device),
                 'InterInstanceDistanceLoss': ContrastiveLoss(args.batchSize, reduce=True, size_average=True).to(device),
                 'InterPrototypeDistanceLoss': PrototypeContrastiveLoss(reduce=True, size_average=True).to(device),
                 'BCEWithLogitsLoss': torch.nn.MultiLabelSoftMarginLoss(),
                 'SeparationLoss': SeparationLoss(reduce=True, size_average=True).to(device),
                 'MDCA': MDCA().to(device),
                 'FocalLoss': FocalLoss().to(device),
                 'FLSD': FLSD().to(device),
                 'DCA': DCA().to(device),
                 'MbLS': MbLS().to(device),
                 'DWBL': DWBL().to(device),
                 }

    for p in ssgrl_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.SGD(gcn_model.get_config_optim(args.lr, args.lrp), 
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weightDecay)

    if args.evaluate:
        Validate(test_loader, ssgrl_model, gcn_model, criterion, 0, args)
        return
    
    # Running Experiment
    logger.info("Run Experiment...")
    writer = SummaryWriter('{}/{}'.format('exp/summary/', args.post))

    if (args.method == 'MPC' or args.method == 'PROTOTYPE'):
        logger.info('Compute Prototype...')
        computePrototype(ssgrl_model, train_loader, args)
        logger.info('Done!\n')

    for epoch in range(args.startEpoch, args.startEpoch + args.epochs):

        Train(train_loader, ssgrl_model, gcn_model, criterion, optimizer, writer, epoch, args)
        mAP, ACE, ECE, MCE = Validate(test_loader, ssgrl_model, gcn_model, criterion, epoch, args)

        writer.add_scalar('mAP', mAP, epoch)
        writer.add_scalar('ACE', ACE, epoch)
        writer.add_scalar('ECE', ECE, epoch)
        writer.add_scalar('MCE', MCE, epoch)

        isBest, bestPrec = mAP > bestPrec, max(mAP, bestPrec)
        save_checkpoint(args, {'epoch':epoch, 'state_dict':gcn_model.state_dict(), 'best_mAP':mAP}, isBest)

        if isBest:
            logger.info('[Best] [Epoch {0}]: Best mAP is {1:.3f}'.format(epoch, bestPrec))

    writer.close()

def Train(train_loader, model, gcn_model, criterion, optimizer, writer, epoch, args):
    gcn_model.train()

    loss, loss_base, loss_plus, loss_calibration = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    batch_time, data_time = AverageMeter(), AverageMeter()
    logger.info("=========================================")

    end = time.time()
    for batchIndex, (sampleIndex, input, target, groundTruth) in enumerate(train_loader):

        input, target = input.to(device), target.float().to(device)

        # Log time of loading data
        data_time.update(time.time() - end)

        # Forward
        output, intraCoOccurrence, feature = model(input)
        outputs = gcn_model(input)

        # Label Smoothing
        if args.method == 'LS':
            target_ = label_smoothing_tradition(args, groundTruth)

        elif args.method == 'PROTOTYPE':
            target_ = label_smoothing_dynamic_CST(args, groundTruth, model.prototype, feature, epoch)

        elif args.method == 'INSTANCE':
            model.updateFeature(feature, target, args.interExampleNumber)
            target_ = label_smoothing_dynamic_CST(args, groundTruth, model.posFeature, feature, epoch)

        elif args.method == 'MPC':
            model.updateFeature(feature, target, args.interExampleNumber)
            target_instance = label_smoothing_dynamic_CST(args, groundTruth, model.posFeature, feature, epoch, 10)
            target_prototype = label_smoothing_dynamic_CST(args, groundTruth, model.prototype, feature, epoch, 10)

        else:
            # Non Label Smoothing
            target_ = target.detach().clone().to(device)
            target_[target_ < 0] = 0
        
        # torch.set_printoptions(sci_mode=False)
        # print(target_)

        # Loss
        if args.method == 'MPC':
            loss_instance = criterion['BCEWithLogitsLoss'](outputs, target_instance)
            loss_prototype = criterion['BCEWithLogitsLoss'](outputs, target_prototype)
            loss_base_ = (loss_instance + loss_prototype) / 2

            loss_plus_ = torch.tensor(0.0).to(device)

            loss_calibration_ = torch.tensor(0.0).to(device)

        elif args.method == 'INSTANCE' or args.method == 'PROTOTYPE':
            loss_base_ = criterion['BCEWithLogitsLoss'](outputs, target_)

            loss_plus_ = args.interDistanceWeight * criterion['InterInstanceDistanceLoss'](feature, target) if epoch >= 1 else \
                     args.interDistanceWeight * criterion['InterInstanceDistanceLoss'](feature, target) * batchIndex / float(len(train_loader))

            # loss_plus_ = torch.tensor(0.0).to(device)

            loss_calibration_ = torch.tensor(0.0).to(device)

        elif args.method == 'FL':
            loss_base_ = criterion['FocalLoss'](outputs, target_)

            loss_plus_ = torch.tensor(0.0).to(device)

            loss_calibration_ = torch.tensor(0.0).to(device)
        
        elif args.method == 'FLSD':
            loss_base_ = criterion['FLSD'](outputs, target_)

            loss_plus_ = torch.tensor(0.0).to(device)

            loss_calibration_ = torch.tensor(0.0).to(device)

        elif args.method == 'MDCA':
            loss_base_ = criterion['BCEWithLogitsLoss'](outputs, target_)

            loss_plus_ = torch.tensor(0.0).to(device)

            loss_calibration_ = criterion['MDCA'](outputs, target_)
        
        elif args.method == 'DCA':
            loss_base_ = criterion['BCEWithLogitsLoss'](outputs, target_)

            loss_plus_ = torch.tensor(0.0).to(device)

            loss_calibration_ = criterion['DCA'](outputs, target_)
        
        elif args.method == 'MbLS':
            loss_base_ = criterion['BCEWithLogitsLoss'](outputs, target_)

            loss_plus_ = torch.tensor(0.0).to(device)

            loss_calibration_ = criterion['MbLS'](outputs, target_)
        
        elif args.method == 'DWBL':
            loss_base_ = criterion['DWBL'](outputs, target_)

            loss_plus_ = torch.tensor(0.0).to(device)

            loss_calibration_ = torch.tensor(0.0).to(device)

        else:
            loss_base_ = criterion['BCEWithLogitsLoss'](outputs, target_)

            loss_plus_ = torch.tensor(0.0).to(device)

            loss_calibration_ = torch.tensor(0.0).to(device)

        loss_ = loss_base_ + loss_plus_ + loss_calibration_

        loss.update(loss_.item(), input.size(0))
        loss_base.update(loss_base_.item(), input.size(0))
        loss_plus.update(loss_plus_.item(), input.size(0))
        loss_calibration.update(loss_calibration_.item(), input.size(0))

        # Backward
        loss_.backward()
        torch.nn.utils.clip_grad_norm_(gcn_model.parameters(), max_norm=args.max_clip_grad_norm)
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

def Validate(val_loader, model, gcn_model, criterion, epoch, args):
    gcn_model.eval()

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

    logger.info(f'[Test] mAP: {mAP:.3f}, averageAP: {averageAP:.3f}\n'
                f'\t\t\t\t(Compute with all label) OP: {OP:.3f}, OR: {OR:.3f}, OF1: {OF1:.3f}, CP: {CP:.3f}, CR: {CR:.3f}, CF1:{CF1:.3f}\n'
                f'\t\t\t\t(Compute with top-3 label) OP: {OP_K:.3f}, OR: {OR_K:.3f}, OF1: {OF1_K:.3f}, CP: {CP_K:.3f}, CR: {CR_K:.3f}, CF1: {CF1_K:.3f}\n'
                f'\t\t\t\tACE:{ACE:.6f}, ECE:{ECE:.6f}, MCE:{MCE:.6f}')
    return mAP, ACE, ECE, MCE


if __name__=="__main__":
    main()
