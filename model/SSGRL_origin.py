import torch
import torch.nn as nn
import numpy as np

from .backbone.resnet import resnet101
from .GGNN import GGNN
from .semantic import semantic
from .Element_Wise_Layer import Element_Wise_Layer

class SSGRL(nn.Module):
    def __init__(self, adjacency_matrix, word_features, 
                image_feature_dim=2048, output_dim=2048, time_step=3,
                classNum=80, word_feature_dim = 300):
        super(SSGRL, self).__init__()
        self.backbone = resnet101()

        self.num_classes = classNum
        self.word_feature_dim = word_feature_dim
        self.image_feature_dim = image_feature_dim
        
        self.word_semantic = semantic(num_classes= self.num_classes,
                                      image_feature_dim = self.image_feature_dim,
                                      word_feature_dim=self.word_feature_dim)

        self.word_features = self.load_features(word_features)
        self._in_matrix, self._out_matrix = self.load_matrix(adjacency_matrix)
        self.time_step = time_step
        
        self.graph_net = GGNN(input_dim=self.image_feature_dim,
                              time_step=self.time_step,
                              in_matrix=self._in_matrix,
                              out_matrix=self._out_matrix)

        self.output_dim = output_dim
        self.fc_output = nn.Linear(2 * self.image_feature_dim, self.output_dim)
        self.classifiers = Element_Wise_Layer(self.num_classes, self.output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        # 特征抽取
        img_feature_map = self.backbone(x)
        # 特征融合（SD）
        graph_net_input = self.word_semantic(img_feature_map,
                                             self.word_features)
        graph_net_feature = self.graph_net(graph_net_input)

        output = torch.cat((graph_net_feature.view(batch_size*self.num_classes,-1), graph_net_input.view(-1, self.image_feature_dim)), 1)
        output = self.fc_output(output)
        output = torch.tanh(output)
        output = output.contiguous().view(batch_size, self.num_classes, self.output_dim)
        result = self.classifiers(output)
        return result 

    def load_features(self, wordFeatures):
        return nn.Parameter(torch.from_numpy(wordFeatures.astype(np.float32)), requires_grad=False)

    def load_matrix(self, mat):
        _in_matrix, _out_matrix = mat.astype(np.float32), mat.T.astype(np.float32)
        _in_matrix, _out_matrix = nn.Parameter(torch.from_numpy(_in_matrix), requires_grad=False), nn.Parameter(torch.from_numpy(_out_matrix), requires_grad=False)
        return _in_matrix, _out_matrix