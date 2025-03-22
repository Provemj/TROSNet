#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2019/12/12 19:31 
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.sync_batchnorm import SynchronizedBatchNorm2d

TRAIN_BN_MOM = 0.0003

class BasicNonLocalBlock(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''
    def __init__(self, in_channels, key_channels, value_channels, out_channels = None):
        super(BasicNonLocalBlock, self).__init__()

        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels

        if not out_channels:
            self.out_channels = self.in_channels
        else:
            self.out_channels = out_channels

        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            SynchronizedBatchNorm2d(self.key_channels, TRAIN_BN_MOM),
        )
        self.f_query = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            SynchronizedBatchNorm2d(self.key_channels, TRAIN_BN_MOM),
        )
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
            kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
            kernel_size=1, stride=1, padding=0)

        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)

        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)

        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)

        key = self.f_key(x).view(batch_size, self.key_channels, -1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        # 按行进行softmax
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)

        return context

# with different input source
# with PSP

class PSPModule(nn.Module):
    # (1, 2, 3, 6)
    # 输出长度 1 + 4 + 9 + 36 = 50
    def __init__(self, sizes=(1, 3, 6, 8), dimension=2):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats):
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        center = torch.cat(priors, -1)
        return center

class TwoSourceDowmSampleNonLocalBlock(nn.Module):
    def __init__(self, in_channels, key_channels, value_channels, out_channels = None, psp_size=(1,3,6,8)):
        super(TwoSourceDowmSampleNonLocalBlock, self).__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels

        if not out_channels:
            self.out_channels = self.in_channels
        else:
            self.out_channels = out_channels

        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            SynchronizedBatchNorm2d(self.key_channels, TRAIN_BN_MOM),
        )
        self.f_query = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            SynchronizedBatchNorm2d(self.key_channels, TRAIN_BN_MOM),
        )
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
            kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
            kernel_size=1, stride=1, padding=0)
        self.psp = PSPModule(psp_size)

        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x1, x2):
        batch_size = x1.size(0)

        value = self.psp(self.f_value(x1))
        value = value.view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)

        query = self.f_query(x2).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)

        key = self.psp(self.f_value(x1))
        key = key.view(batch_size, self.key_channels, -1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        # 按行进行softmax
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x2.size()[2:])
        context = self.W(context)

        return context

if __name__ == "__main__":
    a = torch.randn(1, 10, 40, 50)
    b = torch.randn(1, 10, 40, 50)
    net = TwoSourceDowmSampleNonLocalBlock(10, 5, 5)
    net.eval()
    b = net(a, b)
    print(b.shape)