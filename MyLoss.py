# coding=utf-8

import torch.nn as nn
import torch.nn.functional as F
import torch


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.6, gamma=2, focus=2, logits=False, reduce=False):
        assert focus > 1, 'focus must bigger than 1 !'
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.focus = focus
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets, tags=None):

        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        alpha_t = torch.tensor([self.alpha if i == 1 else 1-self.alpha for i in targets], device=tags.device)

        # 根据tags对部分数据增加loss权重
        tags = torch.tensor(tags, device=targets.device).long()
        tags = tags*(self.focus-1) + torch.ones_like(tags, device=tags.device)
        # print(alpha_t.device, BCE_loss.device, tags_weight.device, pt.device)
        F_loss = alpha_t * (1-pt)**self.gamma * BCE_loss * tags
        del alpha_t, pt, tags, BCE_loss
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return torch.sum(F_loss)
