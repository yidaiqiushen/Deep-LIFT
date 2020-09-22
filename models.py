from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from util import *
from torch.nn import Parameter
import torchvision.models as models


class AttentionAggregator(nn.Module):
    def __init__(self, in_dim):
        super(AttentionAggregator, self).__init__()
        self.weight = nn.Parameter(
                        torch.FloatTensor(in_dim*2))
        self.f1 = nn.Conv1d(in_dim, 1, kernel_size=1, stride=1)
        self.f2 = nn.Conv1d(in_dim, 1, kernel_size=1, stride=1)
        self.prelu = nn.PReLU(1)

    def forward(self, features, A):
        seq = torch.transpose(features, 1, 2)
        f1 = self.f1(seq)
        f2 = self.f2(seq)
        logits = torch.transpose(f1,2,1) + f2
        att = torch.where(A>0, self.prelu(logits), -9e15*torch.ones_like(A))
        att = F.softmax(att, dim=2)
        x = torch.bmm(att, features)
        return x


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, agg):
        super(GraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(
                torch.FloatTensor(in_dim *2, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        init.xavier_uniform_(self.weight)
        init.constant_(self.bias, 0)
        self.agg = agg(in_dim)

    def forward(self, features, A):
        b, n, d = features.shape
        assert(d==self.in_dim)
        agg_feats = self.agg(features,A)
        cat_feats = torch.cat([features, agg_feats], dim=2)
        out = torch.einsum('bnd,df->bnf', (cat_feats, self.weight))
        out = F.relu(out + self.bias)
        return out


class GCNResnet(nn.Module):
    def __init__(self, model, num_classes, in_channel=300, t=0, dropout=0.6, adj_file=None):
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
        
        self.relu = nn.LeakyReLU(0.2)
        self.mlp_1 = nn.Sequential(
            Flatten(),
            nn.Linear(num_classes, num_classes),
            self.relu,
            nn.Linear(num_classes, num_classes)
        )

        self.mlp_2 = nn.Sequential(
            Flatten(),
            nn.Linear(num_classes, num_classes),
            self.relu,
            nn.Linear(num_classes, num_classes)
        )

        self.num_classes = num_classes
        self.dropout = dropout

        self.W = nn.Parameter(torch.zeros(size=(2048, 2048)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.bn0 = nn.BatchNorm1d(in_channel, affine=False)
        self.gc1 = GraphConv(in_channel, 1024, AttentionAggregator)
        self.gc2 = GraphConv(1024, 2048, AttentionAggregator)

        #_adj = gen_A(num_classes, t, adj_file)
        _adj = torch.ones(num_classes, num_classes)
        self.A = Parameter(_adj.float())
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

        self.pooling_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.pooling_max = nn.MaxPool2d(14, 14)

        # inp:B*C*d:32*80*300  feature:32*2048*14*14
    def forward(self, feature, inp):
        feature = self.features(feature)
        feature = feature.view(feature.size(0), feature.size(1), -1)  # [32,2048,196]

        adj = gen_adj(self.A).detach()
        B, N, D = inp.shape
        x = inp.view(-1, D)
        x = self.bn0(x)
        x = x.view(B, N, D)
        
        x1 = self.gc1(x, adj)   # [80,1024]
        # x1 = self.relu(x1)
        x2 = self.gc2(x1, adj)  # [80,2048]
        x2 = self.relu(x2)
        # x = F.dropout(x, self.dropout, training=self.training)

        A = torch.matmul(x2, self.W)
        A = torch.matmul(A, feature)  # [32,80,14**2]
        A_map = A.view(A.size(0), A.size(1), 14, 14)  # [32,80,14,14]

        # Optional: channel attention
        #avg_pool_1 = self.pooling_avg(A_map)  # [32,80,1,1]
        #channel_att1 = self.mlp_1(avg_pool_1)  # [32,80]
        #max_pool_1 = self.pooling_max(A_map)  # [32,80,1,1]
        #channel_att2 = self.mlp_1(max_pool_1)  # [32,80]
        #channel_att_sum = channel_att1 + channel_att2        # [32,80]
        #scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(A_map)
        #A_map_ = A_map * scale                               # [32,80,14,14]

        avg_pool_2 = self.pooling_avg(A_map)  # [32,80,1,1]
        max_pool_2 = self.pooling_max(A_map)  # [32,80,1,1]
        output = avg_pool_2 + max_pool_2      # [32,80,1,1]

        output = self.mlp_2(output)           # [32,80]
        return output

    def get_config_optim(self, model, lr, lrp):
        pretrained_params = list(map(id, self.features.parameters()))
        classify_params = filter(lambda p: id(p) not in pretrained_params, model.parameters())
        return [
            {'params': self.features.parameters(), 'lr': lr * lrp},
            {'params': classify_params, 'lr': lr},
        ]


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def gcn_resnet101(num_classes, t, pretrained=True, adj_file=None, in_channel=300):
    model = models.resnet101(pretrained=pretrained)
    return GCNResnet(model, num_classes, t=t, adj_file=adj_file, in_channel=in_channel)
