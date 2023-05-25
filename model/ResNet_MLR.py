import numpy as np

import torch
import torch.nn as nn

from .backbone.resnet import resnet101

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResNet(nn.Module):
    def __init__(self, classNum=80, wordFeatures=None):

        super(ResNet, self).__init__()

        self.backbone = resnet101()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.wordFeatures = self.load_features(wordFeatures)

        self.fc = nn.Linear(2048, classNum)


    def forward(self, input):
        batchSize = input.size(0)

        featureMap = self.backbone(input)                                            # (BatchSize, Channel, imgSize, imgSize)

        featureMap = self.avgpool(featureMap)                                        # (BatchSize, Channel, 1, 1)

        featureMap = featureMap.reshape(batchSize, -1)                               # (BatchSize, Channel)

        output = self.fc(featureMap)    

        return output
    
    def load_features(self, wordFeatures):
        return nn.Parameter(torch.from_numpy(wordFeatures.astype(np.float32)), requires_grad=False)
