import torch
import torch.nn as nn
import logging

from .MMCE import MMCE_weighted
from .FLSD import FocalLossAdaptive
from torch.nn import functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# from https://github.com/torrvision/focal_calibration/blob/main/Losses/focal_loss.py
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, **kwargs):
        super(FocalLoss, self).__init__()

        self.gamma = gamma

    def forward(self, input, target):
        input = torch.sigmoid(input)
        
        neg_input = 1 - input
        pt = torch.where(target == 1, input, neg_input)
        loss = -1 * (1 - pt) ** 2 * torch.log(pt)

        return loss.mean()

class CrossEntropy(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(CrossEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input, target):
        return self.criterion(input, target)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, alpha=0.0, dim=-1, **kwargs):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - alpha
        self.alpha = alpha
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        num_classes = pred.shape[self.dim]
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.alpha / (num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class MDCA(torch.nn.Module):
    def __init__(self):
        super(MDCA,self).__init__()

    def forward(self, output, target):
        output = torch.sigmoid(output)
        avg_count = torch.mean(target, dim=0)
        avg_conf = torch.mean(output, dim=0)
        loss = torch.abs(avg_conf - avg_count)
        return loss.mean()

class MbLS(nn.Module):
    def __init__(self, margin=5):
        super(MbLS, self).__init__()
        self.margin = margin

    def forward(self, logits, targets):
        max_values = logits.max(dim=1) # [batch_size]
        max_values = max_values.values.unsqueeze(dim=1).repeat(1, logits.shape[1]) # [batch_size, class_num]
        diff = max_values - logits
        loss_margin = F.relu(diff - self.margin).mean()
        return 0.1 * loss_margin
    
class DWBL(nn.Module):
    def __init__(self, margin=5):
        super(DWBL, self).__init__()
        self.margin = margin
        weight = [  45164, 2286, 8606,  2442,  2243,  2789,  2463,  4321,  2098,  2893,
                    1205,  1214,  481,  3844,  2240,  2817,  3041,  2068,  1105,  1388,
                    1518,   668, 1324,  1798,  3923,  2748,  4860,  2666,  1631,  1511,
                    2209,  1170, 2986,  1625,  1804,  1884,  2510,  2343,  2368,  5967,
                    1771,  6516, 2536,  3096,  2493,  5027,  1618,  1171,  1645,  1216,
                    1339,  1185,  821,  2202,  1062,  2080,  8949,  3169,  3084,  2539,
                    8378,  2316, 3191,  2474,  1290,  2179,  1471,  3321,  1088,  2002,
                    151,   3288, 1671,  3732,  3158,  2530,   673,  1510,   128,   700,]
        self.weight = torch.log(torch.tensor(max(weight) / weight).to(device)) + 1

    def forward(self, input, target):

        input = torch.sigmoid(input)
        neg_input = 1 - input

        pt = torch.where(target == 1, input, neg_input)
    
        loss = -1 * self.weight ** (1 - pt) * torch.log(pt) - pt * (1- pt)

        return loss.mean()


class ClassficationAndMDCA(nn.Module):
    def __init__(self, loss="NLL+MDCA", alpha=0.1, beta=1.0, gamma=1.0, **kwargs):
        super(ClassficationAndMDCA, self).__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        if "NLL" in loss:
            self.classification_loss = nn.CrossEntropyLoss()
        elif "FL" in loss:
            self.classification_loss = FocalLoss(gamma=self.gamma)
        else:
            self.classification_loss = LabelSmoothingLoss(alpha=self.alpha) 
        self.MDCA = MDCA()

    def forward(self, logits, targets):
        loss_cls = self.classification_loss(logits, targets)
        loss_cal = self.MDCA(logits, targets)
        return loss_cls + self.beta * loss_cal

class BrierScore(nn.Module):
    def __init__(self, **kwargs):
        super(BrierScore, self).__init__()

    def forward(self, logits, target):
        
        target = target.view(-1,1)
        target_one_hot = torch.FloatTensor(logits.shape).to(target.get_device())
        target_one_hot.zero_()
        target_one_hot.scatter_(1, target, 1)

        pt = torch.softmax(logits, dim=1)
        squared_diff = (target_one_hot - pt) ** 2

        loss = torch.sum(squared_diff) / float(logits.shape[0])
        return loss

class DCA(nn.Module):
    def __init__(self, beta=1.0, **kwargs):
        super().__init__()

    def forward(self, output, target):
        conf = torch.sigmoid(output)
        calib_loss = torch.abs(conf.mean() - target.mean())
        return calib_loss

class MMCE(nn.Module):
    def __init__(self, beta=2.0, **kwargs):
        super().__init__()
        self.beta = beta
        self.mmce = MMCE_weighted()
        self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        cls = self.cls_loss(logits, targets)
        calib = self.mmce(logits, targets)
        return cls + self.beta * calib

class FLSD(nn.Module):
    def __init__(self, gamma=3.0, **kwargs):
        super().__init__()
        self.gamma = gamma
        self.criterion = FocalLossAdaptive(gamma=self.gamma)

    def forward(self, logits, targets):
        return self.criterion.forward(logits, targets)


loss_dict = {
    "focal_loss" : FocalLoss,
    "cross_entropy" : CrossEntropy,
    "LS" : LabelSmoothingLoss,
    "NLL+MDCA" : ClassficationAndMDCA,
    "LS+MDCA" : ClassficationAndMDCA,
    "FL+MDCA" : ClassficationAndMDCA,
    "brier_loss" : BrierScore,
    "NLL+DCA" : DCA,
    "MMCE" : MMCE,
    "FLSD" : FLSD
}