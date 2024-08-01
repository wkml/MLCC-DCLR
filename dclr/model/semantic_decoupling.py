import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticDecoupling(nn.Module):

    def __init__(self, num_classes, image_dim, word_dim, semantic_dim=1024):
        
        super(SemanticDecoupling, self).__init__()

        self.num_classes = num_classes
        self.img_feat_dim = image_dim
        self.word_feat_dim = word_dim
        self.inter_dim = semantic_dim

        self.fc1 = nn.Linear(self.img_feat_dim, self.inter_dim, bias=False)
        self.fc2 = nn.Linear(self.word_feat_dim, self.inter_dim, bias=False)
        self.fc3 = nn.Linear(self.inter_dim, self.inter_dim)
        self.fc4 = nn.Linear(self.inter_dim, 1)

    def forward(self, img_feat, word_feat, visualize=False):
        '''
        Shape of imgFeaturemap : (BatchSize, Channel, imgSize, imgSize)
        Shape of wordFeatures : (classNum, wordFeatureDim)
        '''

        BatchSize, imgSize = img_feat.size()[0], img_feat.size()[3]
        img_feat = torch.transpose(torch.transpose(img_feat, 1, 2), 2, 3) # BatchSize * imgSize * imgSize * Channel
        
        imgFeature = img_feat.reshape(BatchSize * imgSize * imgSize, -1)                                             # (BatchSize * imgSize * imgSize) * Channel
        imgFeature = self.fc1(imgFeature).reshape(BatchSize * imgSize * imgSize, 1, -1).repeat(1, self.num_classes, 1)                    # (BatchSize * imgSize * imgSize) * classNum * intermediaDim
        wordFeature = self.fc2(word_feat).reshape(1, self.num_classes, self.inter_dim).repeat(BatchSize * imgSize * imgSize, 1, 1) # (BatchSize * imgSize * imgSize) * classNum * intermediaDim
        feature = self.fc3(torch.tanh(imgFeature * wordFeature).reshape(-1, self.inter_dim))                                       # (BatchSize * imgSize * imgSize * classNum) * intermediaDim
        
        Coefficient = self.fc4(feature)                                                                                             # (BatchSize * imgSize * imgSize * classNum) * 1
        Coefficient = torch.transpose(torch.transpose(Coefficient.reshape(BatchSize, imgSize, imgSize, self.num_classes), 2, 3), 1, 2).reshape(BatchSize, self.num_classes, -1)
        Coefficient = F.softmax(Coefficient, dim=2)                                                                                 # BatchSize * classNum * (imgSize * imgSize))
        Coefficient = Coefficient.reshape(BatchSize, self.num_classes, imgSize, imgSize)                                                  # BatchSize * classNum * imgSize * imgSize
        Coefficient = torch.transpose(torch.transpose(Coefficient, 1, 2), 2, 3)                                                     # BatchSize * imgSize * imgSize * classNum
        Coefficient = Coefficient.reshape(BatchSize, imgSize, imgSize, self.num_classes, 1).repeat(1, 1, 1, 1, self.img_feat_dim)        # BatchSize * imgSize * imgSize * classNum * imgFeatureDim

        featuremapWithCoefficient = img_feat.reshape(BatchSize, imgSize, imgSize, 1, self.img_feat_dim).repeat(1, 1, 1, self.num_classes, 1) * Coefficient # BatchSize * imgSize * imgSize * classNum * imgFeatureDim
        semanticFeature = torch.sum(torch.sum(featuremapWithCoefficient, 1), 1)                                                                            # BatchSize * classNum * imgFeatureDim

        if visualize:
            return semanticFeature, torch.sum(torch.abs(featuremapWithCoefficient), 4), Coefficient[:,:,:,:,0]
        return semanticFeature, featuremapWithCoefficient, Coefficient[:,:,:,:,0]


