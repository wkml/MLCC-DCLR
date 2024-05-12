import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def label_smoothing_dynamic(cfg, target, pos_feature=None, feature=None, epoch=5, temperature=1):
    b, n, c = feature.shape
    target_ = target.detach().clone().cuda().float()
    target_[target_ == -1] = 0

    epsilon = cfg.model.eps

    if epoch >= cfg.model.generate_label_epoch:
        example_num = pos_feature.shape[1]
        pos_feature = pos_feature.detach().clone().cuda()         # [classNum, example_num, c]
        feature = feature.detach().clone().cuda()
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
        probCoOccurrence = torch.full([target_.shape[0], target_.shape[1], target_.shape[1]], epsilon / (target_.shape[1] - 1)).cuda()
        probCoOccurrence -= torch.diag_embed(torch.diag(probCoOccurrence[0]))
        probCoOccurrence[target_== 0] = 0
        target_[target_ == 1] = 1 - epsilon
        target_ += probCoOccurrence.sum(axis=1)

    return target_
