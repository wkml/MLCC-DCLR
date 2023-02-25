import numpy as np
import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def label_smoothing(args, target, method=1, CoOccurrence=None, epoch=5):
    target_ = target.detach().clone().to(device)

    epsilon = args.eps

    if method == 1:
        probCoOccurrence = torch.full([target_.shape[0], target_.shape[1], target_.shape[1]], epsilon / (target_.shape[1] - 1)).to(device)
        probCoOccurrence -= torch.diag_embed(torch.diag(probCoOccurrence[0]))
        probCoOccurrence[target_== -1] = 0
        target_[target_ == -1] = 0
        target_[target_ == 1] = 1 - epsilon
        target_ += probCoOccurrence.sum(axis=1)

        # target_[target_ == 1.0] = 1 - epsilon
        # target_[target_ == -1.0] = epsilon / target_.shape[1]

        # target_[target_ == -1] = 0
        # temp = torch.where(target_ == 1, (target_.sum(axis=1)-1).reshape(-1,1) * epsilon / (target_.shape[1]-1), target_.sum(axis=1).reshape(-1,1) * epsilon / (target_.shape[1]-1))
        # target_[target_ == 1] = 1 - epsilon
        # target_ += temp

    elif method == 2:
        CoOccurrence = F.softmax(CoOccurrence, dim=0)
        for i in range(target_.shape[0]):
            pos = torch.where(target_[i] == 1)[0]
            for j in range(target_.shape[1]):
                target_[i][j] = (1 - epsilon) if target_[i][j].data == 1 else (CoOccurrence[pos, j].sum() * epsilon)
                
    else:
        if epoch >= 5:
            batchSize, classNum = target_.shape[0], target_.shape[1]
            CoOccurrence = torch.sigmoid(CoOccurrence.detach().clone())
            indexStart, indexEnd, probCoOccurrence = 0, 0, torch.zeros((batchSize, classNum, classNum)).to(device)
            for i in range(classNum):
                probCoOccurrence[:, i, i] = float("-inf")
                indexStart = indexEnd
                indexEnd += classNum - i - 1
                probCoOccurrence[:, i, i+1:] = CoOccurrence[:, indexStart:indexEnd]
                probCoOccurrence[:, i+1:, i] = CoOccurrence[:, indexStart:indexEnd]

            probCoOccurrence = F.softmax(probCoOccurrence, dim=1) * epsilon
            probCoOccurrence[target_== -1] = 0
            target_[target_ == -1] = 0
            target_[target_ == 1] = 1 - epsilon
            target_ += probCoOccurrence.sum(axis=1)
            # for i in range(target_.shape[0]):
            #     pos = torch.where(target_[i] == 1)[0]
            #     for j in range(target_.shape[1]):
            #         target_[i][j] = (1 - epsilon) if target_[i][j].data == 1 \
            #                                       else (probCoOccurrence[i, pos, j].sum() * epsilon)
        else:
            probCoOccurrence = torch.full([target_.shape[0], target_.shape[1], target_.shape[1]], epsilon / (target_.shape[1] - 1)).to(device)
            probCoOccurrence -= torch.diag_embed(torch.diag(probCoOccurrence[0]))
            probCoOccurrence[target_== -1] = 0
            target_[target_ == -1] = 0
            target_[target_ == 1] = 1 - epsilon
            target_ += probCoOccurrence.sum(axis=1)
            # target_[target_ == 1.0] = 1 - epsilon
            # target_[target_ == -1.0] = epsilon / target_.shape[1]

    return target_.to(device)

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