from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
# ------------------------------------------------------------------------------
class CausalPredictor(nn.Module):
    def __init__(self, DIC, in_channels):
        super(CausalPredictor, self).__init__()
        num_classes = 2
        self.embedding_size = 1
        representation_size = in_channels

        self.causal_score = nn.Linear(2*representation_size, num_classes)
        self.Wy = nn.Linear(representation_size, self.embedding_size)
        self.Wz = nn.Linear(representation_size, self.embedding_size)

        nn.init.normal_(self.causal_score.weight, std=0.01)
        nn.init.normal_(self.Wy.weight, std=0.02)
        nn.init.normal_(self.Wz.weight, std=0.02)
        nn.init.constant_(self.Wy.bias, 0)
        nn.init.constant_(self.Wz.bias, 0)
        nn.init.constant_(self.causal_score.bias, 0)

        self.feature_size = representation_size
        self.dic = torch.tensor(np.load(DIC, allow_pickle=True)[1:], dtype=torch.float)
        self.prior = torch.ones(1) * 1 / 1

    def forward(self, x, xm):
        device = x.get_device()
        dic_z = self.dic.view(1, -1).to(device)
        prior = self.prior.to(device)
        # print(x.shape)

        #box_size_list = [proposal.bbox.size(0) for proposal in proposals]
        #feature_split = x.split(box_size_list)
        #xzs = [self.z_dic(feature_pre_obj, dic_z, prior) for feature_pre_obj in feature_split]
        z = self.z_dic(xm, dic_z, prior)

        #causal_logits_list = [self.causal_score(xz) for xz in xzs]

        return z.view(z.shape[0], self.dic.shape[1], self.dic.shape[2])
        
# ------------------------------------------------------------------------------
    def z_dic(self, y, dic_z, prior):
        length = y.size(0)
        # if length == 1:
        #     print('debug')
        # print(y.shape)
        # print(dic_z.shape)
        # test1 = self.Wy(y)
        # test2 = self.Wz(dic_z).t()
        attention = torch.mm(self.Wy(y), self.Wz(dic_z).t()) / (self.embedding_size ** 0.5)
        attention = F.softmax(attention, 1)
        z_hat = attention.unsqueeze(2) * dic_z.unsqueeze(0)
        z = torch.matmul(prior.unsqueeze(0), z_hat).squeeze(1)
        #xz = torch.cat((y.unsqueeze(1).repeat(1, length, 1), z.unsqueeze(0).repeat(length, 1, 1)), 2).view(-1, 2*y.size(1))

        # detect if encounter nan
        #if torch.isnan(xz).sum():
        #    print(xz)
        return z

