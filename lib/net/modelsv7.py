#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/12/5 14:56
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.resnet import resnet_50
from net.res_adiln import ILN, adaILN, ResnetAdaILNBlock
from net.cascaded_attention import CMMFA
from net.sync_batchnorm import SynchronizedBatchNorm2d
TRAIN_BN_MOM = 0.0003

"""
CFM RGB + Depth
"""

class myNet(nn.Module):
    def __init__(self, num_classes = 37, hha = False):
        super(myNet, self).__init__()

        n_blocks = 4
        self.hha = hha

        # rgb image branch
        self.rgb_backbone = None

        # depth image branch
        if not self.hha:
            self.conv1_d = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                     bias=False)

        self.depth_backbone = None

        self.conv_init = nn.Conv2d(64, 64, kernel_size = 1, stride = 1, bias = True)
        self.cmmfa1 = CMMFA(64, 256)
        self.cmmfa2 = CMMFA(256, 512)
        self.cmmfa3 = CMMFA(512, 1024)
        self.cmmfa4 = CMMFA(1024, 2048)

        # merge branch
        n_features = 2048
        self.merge_conv1 = nn.Conv2d(n_features, n_features // 2, 1)
        self.merge_bn1 = SynchronizedBatchNorm2d(n_features // 2, TRAIN_BN_MOM)
        self.merge_conv2  = nn.Conv2d(n_features // 2, n_features // 2, kernel_size = 1, stride = 1, bias = True)

        self.merge_conv1_d = nn.Conv2d(n_features, n_features // 2, 1)
        self.merge_bn1_d = SynchronizedBatchNorm2d(n_features // 2, TRAIN_BN_MOM)
        self.merge_conv2_d = nn.Conv2d(n_features // 2, n_features // 2, kernel_size = 1, stride = 1, bias = True)

        self.merge_avg = nn.AdaptiveMaxPool2d(1)
        self.merge_act = nn.Sigmoid()
        self.merge_conv1x1 = nn.Conv2d(n_features, n_features , kernel_size=1, stride=1, bias=True)
        self.merge_relu = nn.ReLU(True)

        self.merge_conv3 = nn.Conv2d(n_features, n_features, kernel_size=1, stride=1)
        self.merge_conv4 = nn.Conv2d(n_features, n_features // 2, kernel_size=1, stride = 1, bias=True)

        # decoder bottleneck
        self.gamma = nn.Linear(n_features // 2, n_features // 2, bias=False)
        self.beta = nn.Linear(n_features // 2, n_features // 2, bias=False)
        self.up_avg = nn.AdaptiveMaxPool2d(1)

        for i in range(n_blocks):
            setattr(self, 'decoder_t' + str(i+1), ResnetAdaILNBlock(n_features // 2, use_bias=False))


        # upsampling
        for i in range(n_blocks):
            upblocks = [nn.ConvTranspose2d(n_features // (2 * (2 ** i)), n_features // (2 * (2 ** (i+1))), kernel_size=2, stride=2, padding=0, bias=False),
                         ILN(n_features // (2 * (2 ** (i+1)))),
                         nn.ReLU(True)]

            setattr(self, 'decoder_up' + str(i+1), nn.Sequential(*upblocks))

        # final conv
        for i in range(n_blocks):
            setattr(self, 'final_conv' + str(i+1), nn.Conv2d(n_features // (4 * (2 ** i)), num_classes, kernel_size=1, stride=1, padding=0, bias=True))

        # weight initial
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.rgb_backbone = resnet_50()
        self.depth_backbone = resnet_50()

    def encoder(self, rgb, depth):
        # rgb features
        rgb = self.rgb_backbone.conv1(rgb)
        rgb = self.rgb_backbone.bn1(rgb)
        rgb = self.rgb_backbone.relu(rgb)

        # depth features
        if not self.hha:
            depth = self.conv1_d(depth)
        else:
            depth = self.depth_backbone.conv1(depth)

        depth = self.depth_backbone.bn1(depth)
        depth = self.depth_backbone.relu(depth)

        pre_attention = self.conv_init(rgb + depth)

        rgb = self.rgb_backbone.maxpool(rgb)
        depth = self.depth_backbone.maxpool(depth)

        rgb = self.rgb_backbone.layer1(rgb)
        depth = self.depth_backbone.layer1(depth)

        rgb, depth, pre_attention = self.cmmfa1(rgb, depth, pre_attention)

        rgb = self.rgb_backbone.layer2(rgb)
        depth = self.depth_backbone.layer2(depth)
        rgb, depth, pre_attention = self.cmmfa2(rgb, depth, pre_attention)

        rgb = self.rgb_backbone.layer3(rgb)
        depth = self.depth_backbone.layer3(depth)
        rgb, depth, pre_attention = self.cmmfa3(rgb, depth, pre_attention)

        rgb = self.rgb_backbone.layer4(rgb)
        depth = self.depth_backbone.layer4(depth)
        rgb, depth, pre_attention = self.cmmfa4(rgb, depth, pre_attention)

        return rgb, depth, pre_attention

    def merge(self, f_rgb, f_depth, attention):
        f_rgb = self.merge_conv1(f_rgb)
        # [1024, 15, 20]
        f_rgb = self.merge_bn1(f_rgb)
        a_rgb = self.merge_act(self.merge_conv2(self.merge_avg(f_rgb)))
        f_rgb = f_rgb * a_rgb
        # [1024, 15, 20]

        f_depth = self.merge_conv1_d(f_depth)
        # [1024, 15, 20]
        f_depth = self.merge_bn1_d(f_depth)
        a_depth = self.merge_act(self.merge_conv2_d(self.merge_avg(f_depth)))
        f_depth = f_depth * a_depth
        # [1024, 15, 20]

        f_merge = self.merge_relu(self.merge_conv1x1(torch.cat([f_rgb, f_depth], dim = 1)) + attention)
        # [2048, 15, 20]
        a_merge = self.merge_act(self.merge_conv3(self.merge_avg(f_merge)))
        f_merge = f_merge * a_merge
        f_merge = self.merge_relu(self.merge_conv4(f_merge))
        # [1024, 15, 20]
        return f_merge

    def decoder(self, f_merge, phase = 'train'):
        f_merge_avg = self.up_avg(f_merge)
        gamma, beta = self.gamma(f_merge_avg.view(f_merge.shape[0], -1)), self.beta(f_merge_avg.view(f_merge.shape[0], -1))
        # transblocks
        f_merge = self.decoder_t1(f_merge, gamma, beta)
        # [1024, 15, 20]
        f_merge = self.decoder_t2(f_merge, gamma, beta)
        # [1024, 15, 20]
        f_merge = self.decoder_t3(f_merge, gamma, beta)
        # [1024, 15, 20]
        f_merge = self.decoder_t4(f_merge, gamma, beta)
        # [1024, 15, 20]
        # upsample blocks
        features = []

        f_merge = self.decoder_up1(f_merge)
        # [512, 30, 40]
        if phase in ['train','val']:
            features.append(f_merge)

        f_merge = self.decoder_up2(f_merge)
        # [256, 60, 80]
        if phase in ['train','val']:
            features.append(f_merge)

        f_merge = self.decoder_up3(f_merge)
        # [128, 120, 160]
        if phase in ['train','val']:
            features.append(f_merge)

        f_merge = self.decoder_up4(f_merge)
        # [64, 240, 320]
        features.append(f_merge)

        return features

    def forward(self, rgb, depth, phase = 'train'):
        f_rgb, f_depth, attention = self.encoder(rgb, depth)
        features = self.decoder(self.merge(f_rgb, f_depth, attention), phase = phase)

        if phase in ['train', 'val']:
            f = F.interpolate(features[0], size=[rgb.size(2) // 8, rgb.size(3) // 8], mode='bilinear', align_corners=True)
            # [512, 60, 80]
            out1 = self.final_conv1(f)
            # [num_classes, 60, 80]
            f = F.interpolate(features[1], size=[rgb.size(2) // 4, rgb.size(3) // 4], mode='bilinear', align_corners=True)
            # [256, 120, 160]
            out2 = self.final_conv2(f)
            # [num_classes, 120, 160]
            f = F.interpolate(features[2], size=[rgb.size(2) // 2, rgb.size(3) // 2], mode='bilinear', align_corners=True)
            # [128, 240, 320]
            out3 = self.final_conv3(f)
            # [num_classes, 240, 320]
            f = F.interpolate(features[3], size=[rgb.size(2), rgb.size(3)], mode='bilinear', align_corners=True)
            # [64, 480, 640]
            out4 = self.final_conv4(f)
            # [num_classes, 480, 640]
            return [out4, out3, out2, out1]
        else:
            f = F.interpolate(features[0], size=[rgb.size(2), rgb.size(3)], mode='bilinear', align_corners=True)
            out4 = self.final_conv4(f)
            return out4

if __name__ == "__main__":
    net = resnet_50()
    net.eval()
    a = torch.randn(1, 3, 400, 600)
    b = net(a)
    print(b.shape)