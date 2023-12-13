import numpy as np

import torch
import torch.nn as nn

import torchvision.models as models
from kmeans_pytorch import kmeans
from sklearn.cluster import KMeans

from .graph_neural_network import GatedGNN
from .semantic_decoupling import SemanticDecoupling
from .element_wise_layer import ElementWiseLayer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SSGRL(nn.Module):
    def __init__(self, adjacency_matrix, word_features,
                 image_feature_dim=2048, inter_media_dim=1024, output_dim=2048,
                 class_nums=80, word_feature_dim=300, time_step=3):

        super(SSGRL, self).__init__()

        resnet = models.resnet101(pretrained=True)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            nn.AvgPool2d(2, stride=2)
        )

        self.class_nums = class_nums
        self.time_step = time_step

        self.output_dim = output_dim
        self.inter_media_dim = inter_media_dim
        self.word_feature_dim = word_feature_dim
        self.image_feature_dim = image_feature_dim
        
        self.word_features = self.load_features(word_features)
        self.in_matrix, self.out_matrix = self.load_matrix(adjacency_matrix)

        self.semantic_decoupling = SemanticDecoupling(self.class_nums, self.image_feature_dim, self.word_feature_dim, intermediaDim=self.inter_media_dim)
        self.graph_neural_network = GatedGNN(self.image_feature_dim, self.time_step, self.in_matrix, self.out_matrix)

        self.fc = nn.Linear(2 * self.image_feature_dim, self.output_dim)
        self.classifiers = ElementWiseLayer(self.class_nums, self.output_dim)

        self.relu = nn.ReLU(inplace=True)

        self.inter_fc_1 = nn.Linear(self.image_feature_dim, self.inter_media_dim)
        self.inter_fc_2 = nn.Linear(self.inter_media_dim, self.image_feature_dim)

        # self.pos_feature = None
        # self.prototype = None

        self.pos_feature = nn.Parameter(torch.zeros(self.class_nums, 100, image_feature_dim), requires_grad=False)
        self.prototype = nn.Parameter(torch.zeros(self.class_nums, 10, image_feature_dim), requires_grad=False)

    def forward(self, input, only_feature=False):
        batchSize = input.size(0)

        # (BatchSize, Channel, imgSize, imgSize)
        feature_map = self.backbone(input)                                            

        # (BatchSize, classNum, imgFeatureDim)
        semantic_feature = self.semantic_decoupling(feature_map, self.word_features)[0]  

        cos_semantic_feature = self.inter_fc_1(semantic_feature)
        cos_semantic_feature = self.inter_fc_2(self.relu(cos_semantic_feature))

        if only_feature:
            return cos_semantic_feature
        
        # (BatchSize, classNum, imgFeatureDim)
        feature = self.graph_neural_network(semantic_feature)                           
        
        # Predict Category
        output = torch.tanh(self.fc(torch.cat((feature.view(batchSize * self.class_nums, -1),
                                               semantic_feature.view(-1, self.image_feature_dim)),1)))

        output = output.contiguous().view(batchSize, self.class_nums, self.output_dim)
        # (BatchSize, classNum)
        result = self.classifiers(output)                                            

        return result, cos_semantic_feature

    def load_features(self, word_features):
        return nn.Parameter(torch.from_numpy(word_features.astype(np.float32)), requires_grad=False)

    def load_matrix(self, mat):
        _in_matrix, _out_matrix = mat.astype(np.float32), mat.T.astype(np.float32)
        _in_matrix, _out_matrix = nn.Parameter(torch.from_numpy(_in_matrix), requires_grad=False), nn.Parameter(torch.from_numpy(_out_matrix), requires_grad=False)
        return _in_matrix, _out_matrix
    

# =============================================================================
# Help Functions
# =============================================================================

def update_feature(model, feature, target, example_num):
    if model.pos_feature is None:
        model.pos_feature = torch.zeros((model.class_nums, example_num, feature.size(-1))).to(device)

    feature = feature.detach().clone()
    for c in range(model.class_nums):
        pos_feature = feature[:, c][target[:, c] == 1]
        model.pos_feature[c] = torch.cat((pos_feature, model.pos_feature[c]), dim=0)[:example_num]

def compute_prototype(model, train_loader, cfg):
    model.eval()

    prototypes, features = [], [torch.zeros(10, model.output_dim) for i in range(cfg.dataset.class_nums)]

    for _, (_, input, target, groundTruth, _) in enumerate(train_loader):

        input, target, groundTruth = input.to(device), target.to(device), groundTruth.to(device)

        with torch.no_grad():
            feature = model(input, only_feature=True).cpu()

            for i in range(cfg.dataset.class_nums):
                features[i] = torch.cat((features[i], feature[target[:, i] == 1, i]), dim=0)

    for i in range(cfg.dataset.class_nums):
        kmeans = KMeans(n_clusters=cfg.model.prototype_nums).fit(features[i][10:].numpy())
        # prototypes.append(torch.tensor(kmeans.cluster_centers_).to(device))
        model.prototype[i] = torch.tensor(kmeans.cluster_centers_).to(device)
    # model.prototype = torch.stack(prototypes, dim=0)

# def compute_prototype(model, train_loader, cfg):
#     model.eval()

#     prototypes, features = [], [torch.zeros(10, model.output_dim).to(device) for _ in range(cfg.dataset.class_nums)]

#     for _, (_, input, target, full_labels, _) in enumerate(train_loader):

#         input, target, full_labels = input.to(device), target.to(device), full_labels.to(device)

#         with torch.no_grad():
#             semantic_feature = model(input, only_feature=True)

#             for i in range(cfg.dataset.class_nums):
#                 features[i] = torch.cat((features[i], semantic_feature[target[:, i] == 1, i]), dim=0)

#     for i in range(cfg.dataset.class_nums):
#         _, cluster_centers = kmeans(X=features[i][10:], num_clusters=cfg.model.prototype_nums, device=device)
#         model.prototype[i] = cluster_centers