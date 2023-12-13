import numpy as np
import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def label_smoothing_tradition(cfg, target):
    target_ = target.detach().clone().to(device).float()
    target_[target_ == -1] = 0

    epsilon = cfg.model.eps

    probCoOccurrence = torch.full([target_.shape[0], target_.shape[1], target_.shape[1]], epsilon / (target_.shape[1] - 1)).to(device)
    probCoOccurrence -= torch.diag_embed(torch.diag(probCoOccurrence[0]))
    probCoOccurrence[target_== 0] = 0
    target_[target_ == 1] = 1 - epsilon
    target_ += probCoOccurrence.sum(axis=1)

    return target_

def label_smoothing_dynamic(cfg, target, pos_feature=None, feature=None, epoch=5, temperature=1):
    b, n, c = feature.shape
    target_ = target.detach().clone().to(device).float()
    target_[target_ == -1] = 0

    epsilon = cfg.model.eps

    if epoch >= cfg.model.generate_label_epoch:
        example_num = pos_feature.shape[1]
        pos_feature = pos_feature.detach().clone().to(device)         # [classNum, example_num, c]
        feature = feature.detach().clone().to(device)
        pos_feature /= pos_feature.norm(dim=-1, keepdim=True)         # [classNum, example_num, c]
        pos_feature = pos_feature.reshape(cfg.dataset.class_nums * example_num, -1)      # [classNum * example_num, c]
        feature /= feature.norm(dim=-1, keepdim=True)               # [batch, classNum, c]

        probCoOccurrence = feature @ pos_feature.T                   # [batch, classNum, classNum * example_num]
        probCoOccurrence = probCoOccurrence.reshape(b, n, cfg.dataset.class_nums, example_num)   # [batch, classNum, classNum, example_num]
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
