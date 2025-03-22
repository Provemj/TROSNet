#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2019/10/31 13:40
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F

from net.resnet import resnet_50
from net.res_adiln import ILN, adaILN, ResnetAdaILNBlock
from net.sync_batchnorm import SynchronizedBatchNorm2d
from net.MMFF import MMFF

TRAIN_BN_MOM = 0.0003

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

        # merge branch
        n_features = 2048
        self.MMFF = MMFF(in_channels = n_features, key_channels = n_features, value_channels = n_features, out_channels = n_features, psp_size=(1,3,6,8))

        # decoder bottleneck
        self.gamma = nn.Linear(n_features // 2, n_features // 2, bias=False)
        self.beta = nn.Linear(n_features // 2, n_features // 2, bias=False)
        self.up_avg = nn.AdaptiveAvgPool2d(1)

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
        # input rgb, depth [3, 480, 640]
        rgb = self.rgb_backbone.conv1(rgb)
        # [64, 240, 320]
        rgb = self.rgb_backbone.bn1(rgb)
        rgb = self.rgb_backbone.relu(rgb)
        rgb = self.rgb_backbone.maxpool(rgb)
        # [64, 120, 160]
        rgb = self.rgb_backbone.layer1(rgb)
        # [256, 120, 160]
        rgb = self.rgb_backbone.layer2(rgb)
        # [512, 60, 80]
        rgb = self.rgb_backbone.layer3(rgb)
        # [1024, 30, 40]
        rgb = self.rgb_backbone.layer4(rgb)
        # [2048, 15, 20]

        # depth features
        if not self.hha:
            depth = self.conv1_d(depth)
        else:
            depth = self.depth_backbone.conv1(depth)
        # [64, 240, 320]
        depth = self.depth_backbone.bn1(depth)
        depth = self.depth_backbone.relu(depth)
        depth = self.depth_backbone.maxpool(depth)
        # [64, 120, 160]
        depth = self.depth_backbone.layer1(depth)
        # [256, 120, 160]
        depth = self.depth_backbone.layer2(depth)
        # [512, 60, 80]
        depth = self.depth_backbone.layer3(depth)
        # [1024, 30, 40]
        depth = self.depth_backbone.layer4(depth)
        # [2048, 15, 20]

        return rgb, depth

    def merge(self, f_rgb, f_depth):
        f_merge = self.MMFF(f_rgb, f_depth)
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
        f_rgb, f_depth = self.encoder(rgb, depth)
        features = self.decoder(self.merge(f_rgb, f_depth), phase = phase)

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