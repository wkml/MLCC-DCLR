import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticDecouplingOriginal(nn.Module):

    def __init__(self, num_classes, image_dim, word_dim, semantic_dim=1024):
        
        super(SemanticDecouplingOriginal, self).__init__()

        self.num_classes = num_classes
        self.image_dim = image_dim
        self.word_dim = word_dim
        self.semantic_dim = semantic_dim

        self.fc1 = nn.Linear(self.image_dim, self.semantic_dim, bias=False)
        self.fc2 = nn.Linear(self.word_dim, self.semantic_dim, bias=False)
        self.fc3 = nn.Linear(self.semantic_dim, self.semantic_dim)
        self.fc4 = nn.Linear(self.semantic_dim, 1)

    def forward(self, imgFeaturemap, wordFeatures, visualize=False):
        BatchSize, imgSize = imgFeaturemap.size()[0], imgFeaturemap.size()[3]
        imgFeaturemap = torch.transpose(torch.transpose(imgFeaturemap, 1, 2), 2, 3) # BatchSize * imgSize * imgSize * Channel
        
        imgFeature = imgFeaturemap.contiguous().view(BatchSize * imgSize * imgSize, -1)                                             # (BatchSize * imgSize * imgSize) * Channel
        imgFeature = self.fc1(imgFeature).view(BatchSize * imgSize * imgSize, 1, -1).repeat(1, self.num_classes, 1)                    # (BatchSize * imgSize * imgSize) * classNum * intermediaDim
        wordFeature = self.fc2(wordFeatures).view(1, self.num_classes, self.semantic_dim).repeat(BatchSize * imgSize * imgSize, 1, 1) # (BatchSize * imgSize * imgSize) * classNum * intermediaDim
        feature = self.fc3(torch.tanh(imgFeature * wordFeature).view(-1, self.semantic_dim))                                       # (BatchSize * imgSize * imgSize * classNum) * intermediaDim
        
        Coefficient = self.fc4(feature)                                                                                             # (BatchSize * imgSize * imgSize * classNum) * 1
        Coefficient = torch.transpose(torch.transpose(Coefficient.view(BatchSize, imgSize, imgSize, self.num_classes), 2, 3), 1, 2).view(BatchSize, self.num_classes, -1)
        Coefficient = F.softmax(Coefficient, dim=2)                                                                                 # BatchSize * classNum * (imgSize * imgSize))
        Coefficient = Coefficient.view(BatchSize, self.num_classes, imgSize, imgSize)                                                  # BatchSize * classNum * imgSize * imgSize
        Coefficient = torch.transpose(torch.transpose(Coefficient, 1, 2), 2, 3)                                                     # BatchSize * imgSize * imgSize * classNum
        Coefficient = Coefficient.view(BatchSize, imgSize, imgSize, self.num_classes, 1).repeat(1, 1, 1, 1, self.image_dim)        # BatchSize * imgSize * imgSize * classNum * imgFeatureDim

        featuremapWithCoefficient = imgFeaturemap.view(BatchSize, imgSize, imgSize, 1, self.image_dim).repeat(1, 1, 1, self.num_classes, 1) * Coefficient # BatchSize * imgSize * imgSize * classNum * imgFeatureDim
        semanticFeature = torch.sum(torch.sum(featuremapWithCoefficient, 1), 1)                                                                            # BatchSize * classNum * imgFeatureDim

        if visualize:
            return semanticFeature, torch.sum(torch.abs(featuremapWithCoefficient), 4), Coefficient[:,:,:,:,0]
        return semanticFeature, featuremapWithCoefficient, Coefficient[:,:,:,:,0]

class SemanticDecoupling(nn.Module):

    def __init__(self, num_classes, image_dim, word_dim, semantic_dim=1024):
        super(SemanticDecoupling, self).__init__()

        self.num_classes = num_classes
        self.image_dim = image_dim
        self.word_dim = word_dim
        self.semantic_dim = semantic_dim

        self.fc_img = nn.Linear(image_dim, semantic_dim, bias=False)
        self.fc_word = nn.Linear(word_dim, semantic_dim, bias=False)
        self.fc_inter1 = nn.Linear(semantic_dim, semantic_dim)
        self.fc_inter2 = nn.Linear(semantic_dim, 1)

    def forward(self, image_map: torch.Tensor, word_features: torch.Tensor, visualize=False):
        batch_size, img_size = image_map.size(0), image_map.size(3)
        
        # (batch_size, img_size * img_size * num_classes, semantic_dim)
        image_map = (image_map.permute(0, 3, 1, 2)
                     .reshape(batch_size, img_size * img_size, self.image_dim)
                     .unsqueeze(2)
                     .expand(-1, -1, self.num_classes, -1)
                     .reshape(batch_size, img_size * img_size * self.num_classes, -1))
        
        word_feature = (word_features.unsqueeze(0).unsqueeze(1)
                        .expand(batch_size, img_size * img_size, -1, -1)
                        .reshape(batch_size, img_size * img_size * self.num_classes, -1))
        
        # (batch_size, img_size * img_size * num_classes, semantic_dim)
        img_feature = self.fc_img(image_map)
        word_feature = self.fc_word(word_feature)
        fusion_feature = torch.tanh(img_feature * word_feature)

        fusion_feature = self.fc_inter1(fusion_feature)
        attn_map = self.fc_inter2(fusion_feature).reshape(batch_size, img_size, img_size, self.num_classes)

        attn_map = F.softmax(attn_map, dim=3)
        feature_map_with_coefficient = image_map.reshape(batch_size, img_size, img_size, self.num_classes, -1).expand(-1, -1, -1, self.num_classes, -1) * attn_map.unsqueeze(4)

        semantic_feature = feature_map_with_coefficient.sum(1).sum(1)

        if visualize:
            return semantic_feature, torch.sum(torch.abs(feature_map_with_coefficient), 4), attn_map
        return semantic_feature, feature_map_with_coefficient, attn_map

if __name__ == '__main__':
    batch_size = 2
    img_size = 64
    num_classes = 10
    image_dim = 1024
    word_dim = 128

    # 生成随机输入
    img_featuremap = torch.rand(batch_size, image_dim, img_size, img_size).cuda()
    word_features = torch.rand(num_classes, word_dim).cuda()

    # 原始代码测试
    # model_original = SemanticDecoupling(num_classes, image_dim, word_dim).cuda()
    # output_original, _, _ = model_original(img_featuremap, word_features)

    # 重写后的代码测试
    model_rewritten = SemanticDecoupling(num_classes, image_dim, word_dim).cuda()
    output_rewritten, _, _ = model_rewritten(img_featuremap, word_features)
    print(output_rewritten.shape)