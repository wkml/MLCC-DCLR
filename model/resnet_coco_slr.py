import numpy as np

import torch
import torch.nn as nn

from .backbone.resnet import resnet101
from .GraphNeuralNetwork import GatedGNN
from .SemanticDecoupling import SemanticDecoupling
from .Element_Wise_Layer import Element_Wise_Layer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResNet_COCOSLR(nn.Module):
    def __init__(self, classNum=80):

        super(ResNet_COCOSLR, self).__init__()

        self.backbone = resnet101()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(2048, classNum)


    def forward(self, input):
        batchSize = input.size(0)

        featureMap = self.backbone(input)                                            # (BatchSize, Channel, imgSize, imgSize)

        featureMap = self.avgpool(featureMap)                                        # (BatchSize, Channel, 1, 1)

        featureMap = featureMap.view(batchSize, -1)                                  # (BatchSize, Channel)

        output = self.fc(featureMap)                                                 # (BatchSize, classNum)

        return output
        