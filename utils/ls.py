import numpy as np
import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def label_smoothing(target, method='1', CoOccurrence=None, epoch=5):
    target_ = target.float().cpu()

    if method == '1':
        target_[target_ == 1.0] = 0.9
        target_[target_ == -1.0] = 0.1 / target_.shape[1]
    elif method == '2':
        CoOccurrence = F.softmax(CoOccurrence, dim=0)
        for i in range(target_.shape[0]):
            pos = np.where(target_[i] == 1)
            for j in range(target_.shape[1]):
                target_[i][j] = 0.9 if target_[i][j].data == 1 else (CoOccurrence[pos, j].sum() * 0.1)
    else:
        if epoch >= 5:
            batchSize, classNum = target_.shape[0], target_.shape[1]
            CoOccurrence = torch.sigmoid(CoOccurrence)
            indexStart, indexEnd, probCoOccurrence = 0, 0, torch.zeros((batchSize, classNum, classNum)).to(device)
            for i in range(classNum):
                probCoOccurrence[:, i, i] = 1
                indexStart = indexEnd
                indexEnd += classNum-i-1
                probCoOccurrence[:, i, i+1:] = CoOccurrence[:, indexStart:indexEnd]
                probCoOccurrence[:, i+1:, i] = CoOccurrence[:, indexStart:indexEnd]

            probCoOccurrence = F.softmax(probCoOccurrence, dim=1)
            for i in range(target_.shape[0]):
                pos = np.where(target_[i] == 1)
                for j in range(target_.shape[1]):
                    target_[i][j] = 0.9 if target_[i][j].data == 1 else (probCoOccurrence[i, pos, j].sum() * 0.1)
        else:
            target_[target_ == 1.0] = 0.9
            target_[target_ == -1.0] = 0.1 / target_.shape[1]

    return target_.to(device)