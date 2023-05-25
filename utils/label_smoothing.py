import numpy as np
import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def label_smoothing_word(args, target, word_file):
    b, c = target.shape
    target_ = target.detach().clone().to(device).float()
    target_[target_ == -1] = 0

    epsilon = args.eps

    feature = word_file.detach().clone().to(device)
    feature /= feature.norm(dim=-1, keepdim=True)               # [batch, classNum, c]

    probCoOccurrence = feature @ feature.T                   # [batch, classNum, classNum * exampleNum]
    probCoOccurrence.fill_diagonal_(float("-inf"))
    probCoOccurrence = F.softmax(probCoOccurrence * 5, dim=1) * epsilon
    probCoOccurrence = probCoOccurrence.reshape(1, c, c).repeat(b, 1, 1)
    probCoOccurrence[target_== 0] = 0
    
    target_[target_ == 1] = 1 - epsilon
    target_ += probCoOccurrence.sum(axis=1)

    return target_

def label_smoothing_tradition(args, target):
    target_ = target.detach().clone().to(device).float()
    target_[target_ == -1] = 0

    epsilon = args.eps

    probCoOccurrence = torch.full([target_.shape[0], target_.shape[1], target_.shape[1]], epsilon / (target_.shape[1] - 1)).to(device)
    probCoOccurrence -= torch.diag_embed(torch.diag(probCoOccurrence[0]))
    probCoOccurrence[target_== 0] = 0
    target_[target_ == 1] = 1 - epsilon
    target_ += probCoOccurrence.sum(axis=1)

    return target_

def label_smoothing_dynamic_IST(args, target, CoOccurrence=None, epoch=5):
    target_ = target.detach().clone().to(device)

    epsilon = args.eps

    if epoch >= args.generateLabelEpoch:
        batchSize, classNum = target_.shape[0], target_.shape[1]
        CoOccurrence = torch.sigmoid(CoOccurrence.detach().clone())
        indexStart, indexEnd, probCoOccurrence = 0, 0, torch.zeros((batchSize, classNum, classNum)).to(device)
        for i in range(classNum):
            probCoOccurrence[:, i, i] = float("-inf")
            indexStart = indexEnd
            indexEnd += classNum - i - 1
            probCoOccurrence[:, i, i+1:] = CoOccurrence[:, indexStart:indexEnd]
            probCoOccurrence[:, i+1:, i] = CoOccurrence[:, indexStart:indexEnd]

        probCoOccurrence = F.softmax(probCoOccurrence * 5, dim=2) * epsilon
        probCoOccurrence[target_== -1] = 0
        target_[target_ == -1] = 0
        target_[target_ == 1] = 1 - epsilon
        target_ += probCoOccurrence.sum(axis=1)
    else:
        probCoOccurrence = torch.full([target_.shape[0], target_.shape[1], target_.shape[1]], epsilon / (target_.shape[1] - 1)).to(device)
        probCoOccurrence -= torch.diag_embed(torch.diag(probCoOccurrence[0]))
        probCoOccurrence[target_== -1] = 0
        target_[target_ == -1] = 0
        target_[target_ == 1] = 1 - epsilon
        target_ += probCoOccurrence.sum(axis=1)

    return target_

def label_smoothing_dynamic_CST(args, target, posFeature=None, feature=None, epoch=5, temperature=1):
    b, n, c = feature.shape
    target_ = target.detach().clone().to(device).float()
    target_[target_ == -1] = 0

    epsilon = args.eps

    if epoch >= args.generateLabelEpoch:
        exampleNum = posFeature.shape[1]
        posFeature = posFeature.detach().clone().to(device)         # [classNum, exampleNum, c]
        feature = feature.detach().clone().to(device)
        posFeature /= posFeature.norm(dim=-1, keepdim=True)         # [classNum, exampleNum, c]
        posFeature = posFeature.reshape(args.classNum * exampleNum, -1)      # [classNum * exampleNum, c]
        feature /= feature.norm(dim=-1, keepdim=True)               # [batch, classNum, c]

        probCoOccurrence = feature @ posFeature.T                   # [batch, classNum, classNum * exampleNum]
        probCoOccurrence = probCoOccurrence.reshape(b, n, args.classNum, exampleNum)   # [batch, classNum, classNum, exampleNum]
        probCoOccurrence = torch.mean(probCoOccurrence, dim=-1)     # [batch, classNum, classNum] mean pooling
        for i in range(b):
            probCoOccurrence[i].fill_diagonal_(float("-inf"))
        
        probCoOccurrence = F.softmax(probCoOccurrence * temperature, dim=2) * epsilon
        probCoOccurrence[target_== 0] = 0
        target_[target_ == 1] = 1 - epsilon
        target_ += probCoOccurrence.sum(axis=1)

    else:
        probCoOccurrence = torch.full([target_.shape[0], target_.shape[1], target_.shape[1]], epsilon / (target_.shape[1] - 1)).to(device)
        probCoOccurrence -= torch.diag_embed(torch.diag(probCoOccurrence[0]))
        probCoOccurrence[target_== 0] = 0
        target_[target_ == 1] = 1 - epsilon
        target_ += probCoOccurrence.sum(axis=1)

    return target_


def label_smoothing_partial(target, method=1, CoOccurrence=None, epoch=5):
    target_ = target.detach().clone()

    epsilon = 0.03

    if method == 1:
        # target_[target_ == 1.0] = 1.0 - epsilon
        # target_[target_ == -1.0] = -1.0 + epsilon / target_.shape[1]
        temp = torch.where(target_ == 1, (target_.sum(axis=1)-1).reshape(-1,1) * epsilon / (target_.shape[1]-1), target_.sum(axis=1).reshape(-1,1) * epsilon / (target_.shape[1]-1))
        target_[target_ == 1.0] = 1.0 - epsilon
        target_ += temp

    elif method == 2:
        CoOccurrence = F.softmax(CoOccurrence, dim=0)
        for i in range(target_.shape[0]):
            pos = torch.where(target_[i] == 1)[0]
            for j in range(target_.shape[1]):
                target_[i][j] = (1 - epsilon) if target_[i][j].data == 1 \
                                              else (-1.0 + CoOccurrence[pos, j].sum() * epsilon)
                
    else:
        if epoch >= 3:
            batchSize, classNum = target_.shape[0], target_.shape[1]
            CoOccurrence = torch.sigmoid(CoOccurrence.detach().clone())
            indexStart, indexEnd, probCoOccurrence = 0, 0, torch.zeros((batchSize, classNum, classNum)).to(device)
            for i in range(classNum):
                probCoOccurrence[:, i, i] = 1
                indexStart = indexEnd
                indexEnd += classNum - i - 1
                probCoOccurrence[:, i, i+1:] = CoOccurrence[:, indexStart:indexEnd]
                probCoOccurrence[:, i+1:, i] = CoOccurrence[:, indexStart:indexEnd]

            probCoOccurrence = F.softmax(probCoOccurrence, dim=1)
            for i in range(target_.shape[0]):
                pos = torch.where(target_[i] == 1)[0]
                for j in range(target_.shape[1]):
                    target_[i][j] = (1 - epsilon) if target_[i][j].data == 1 \
                                                  else (-1.0 + probCoOccurrence[i, pos, j].sum() * epsilon)
        else:
            temp = torch.where(target_ == 1, (target_.sum(axis=1)-1).reshape(-1,1) * epsilon / (target_.shape[1]-1), target_.sum(axis=1).reshape(-1,1) * epsilon / (target_.shape[1]-1))
            target_[target_ == 1.0] = 1.0 - epsilon
            target_ += temp

    return target_.to(device)