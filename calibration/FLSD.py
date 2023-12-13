import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from scipy.special import lambertw
import numpy as np


def get_gamma(p=0.2):
    '''
    Get the gamma for a given pt where the function g(p, gamma) = 1
    '''
    y = ((1-p)**(1-(1-p)/(p*np.log(p)))/(p*np.log(p)))*np.log(1-p)
    gamma_complex = (1-p)/(p*np.log(p)) + lambertw(-y + 1e-12, k=-1)/np.log(1-p)
    gamma = np.real(gamma_complex) #gamma for which p_t > p results in g(p_t,gamma)<1
    return gamma

class FocalLossAdaptive(nn.Module):
    def __init__(self, gamma=0):
        super(FocalLossAdaptive, self).__init__()
        self.gamma = 1

    def get_gamma_list(self, pt):
        gamma = torch.full([pt.size(0), pt.size(1)], self.gamma, device=pt.device)
        gamma[pt > 0.2] = 0.5
        return gamma

    def forward(self, input, target):
        input = torch.sigmoid(input)
        
        neg_input = 1 - input
        pt = torch.where(target == 1, input, neg_input)
        pt = torch.clamp(pt, 0.001, 0.999)
        gamma = self.get_gamma_list(pt)
        loss = -1 * (1 - pt) ** gamma * torch.log(pt)
        
        return loss.mean()
