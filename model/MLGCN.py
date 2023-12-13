import pickle
import torchvision.models as models
from torch.nn import Parameter
import math
import torch
import numpy as np
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNResnet(nn.Module):
    def __init__(self, model, num_classes, in_channel=300, t=0, train_label=None, cfg=None):
        super(GCNResnet, self).__init__()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.num_classes = num_classes
        self.pooling = nn.MaxPool2d(14, 14)

        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)

        adj_file = "datasets/coco_adj.pkl"
        _adj = gen_A(num_classes, t, adj_file)
        # _adj = gen_graph(num_classes, t, train_label)
        self.A = Parameter(torch.from_numpy(_adj).float())

        # shape = [class_num, emb_size]
        # label_embedding = np.load(cfg.dataset.data_vector)
        # self.label_embedding = Parameter(torch.Tensor(label_embedding).float(), requires_grad=False)
        
        label_embedding = 'datasets/coco_glove_word2vec.pkl'
        with open(label_embedding, 'rb') as f:
            self.label_embedding = pickle.load(f)
        # [class_num=80, emb_size=300]
        self.label_embedding = Parameter(torch.Tensor(self.label_embedding).float(), requires_grad=False)
        # [class_num, emb_size] -> [batch_size, class_num, emb_size]
        # self.label_embedding = torch.unsqueeze(self.label_embedding, 0).repeat(cfg.batch_size, 1, 1).to(device)

    def forward(self, feature):
        feature = self.features(feature)
        feature = self.pooling(feature)
        feature = feature.reshape(feature.size(0), -1)

        adj = gen_adj(self.A).detach()
        x = self.gc1(self.label_embedding, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)

        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)
        return x

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                ]


def gen_graph(num_classes, t, labels):

    graph = np.zeros((labels.shape[1], labels.shape[1]), dtype=float)

    for index in range(labels.shape[0]):
        indexs = np.where(labels[index] == 1)[0]
        for i in indexs:
            for j in indexs:
                graph[i, j] += 1

    for i in range(labels.shape[1]):
        graph[i] /= graph[i, i]

    np.nan_to_num(graph)
    t = 0.5
    graph[graph < t] = 0
    graph[graph >= t] = 1

    # graph = graph * 0.25 / (graph.sum(0, keepdims=True) + 1e-6)
    # graph = graph + np.identity(num_classes, np.int)

    return graph

def gen_A(num_classes, t, adj_file):
    import pickle
    result = pickle.load(open(adj_file, 'rb'))
    
    _adj = result['adj']
    _nums = result['nums']
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes, np.int)
    return _adj


def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


def gcn_resnet101(num_classes, t, train_label=None, in_channel=300, cfg=None):
    model = models.resnet101(pretrained=True)
    return GCNResnet(model, num_classes, t=t, train_label=train_label, in_channel=in_channel, cfg=cfg)