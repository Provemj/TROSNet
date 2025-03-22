#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2019/12/5 15:03 
# @Software: PyCharm
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


# Cascaded Multi Modality Fusion Attention
class CMMFA(nn.Module):
    def __init__(self, pre_channel, cur_channel):
        super(CMMFA, self).__init__()
        self.rho = Parameter(torch.Tensor(1, 1, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, 1, 1, 1))
        self.rho.data.fill_(1.0)
        self.gamma.data.fill_(0.5)
        self.conv_p_to_c = nn.Conv2d(pre_channel, cur_channel, kernel_size = 3, stride = 2, padding = 1, bias = True)
        self.conv_merge = nn.Conv2d(cur_channel, cur_channel, kernel_size = 1, stride = 1, bias = True)
        self.conv_c1 = nn.Conv2d(cur_channel, cur_channel, kernel_size = 1, stride = 1, bias = True)
        self.conv_c2 = nn.Conv2d(cur_channel, cur_channel, kernel_size = 1, stride = 1, bias = True)

        self.act = nn.Sigmoid()

    def forward(self, rgb, depth, pre_cmmfa):
        pre_cmmfa = (1 - self.rho) * self.conv_p_to_c(pre_cmmfa) + self.rho * self.conv_merge(self.gamma * rgb + (1 - self.gamma) * depth)
        rgb = rgb + self.act(self.conv_c1(pre_cmmfa)) * (rgb + depth)
        depth = depth + self.act(self.conv_c2(pre_cmmfa)) * (depth + rgb)

        return rgb, depth, pre_cmmfa
