#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2020/2/16 13:14 
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
from net.resnet import resnet_101

BatchNorm2d = nn.BatchNorm2d

class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x

'''
[description]

'''
class MultiResolutionFusion(nn.Module):
    def __init__(self, out_feats, *shapes):
        super().__init__()

        _, max_size = max(shapes, key=lambda x: x[1])       # get the maxer shape of several input feture maps
        self.max_size = (max_size, max_size)

        self.scale_factors = []
        for i, shape in enumerate(shapes):
            feat, size = shape
            # if max_size % size != 0:
            #     raise ValueError("max_size not divisble by shape {}".format(i))

            # self.scale_factors.append(max_size // size)
            self.add_module(
                "resolve{}".format(i),
                nn.Conv2d(
                    feat,
                    out_feats,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False))

    def forward(self, *xs):
        # print(self.max_size)
        max_size = self.max_size#xs[-1].size()[-2:]     # max size of these feature, in default situation, the last data in the data-array has the biggest shape
        output = self.resolve0(xs[0])
        if xs[0].size()[-2] != max_size[0]:
            output = nn.functional.interpolate(
                output,
                size=max_size,
                mode='bilinear',
                align_corners=True)

        for i, x in enumerate(xs[1:], 1):
            this_feature = self.__getattr__("resolve{}".format(i))(x)
            # upsamples all (smaller) feature maps to the largest resolution of the inputs
            if xs[i].size()[-2] != max_size[0]:
                this_feature = nn.functional.interpolate(
                    this_feature,
                    size=max_size,
                    mode='bilinear',
                    align_corners=True)
            output += this_feature

        return output


'''
[description]
chained residual pool
'''
class ChainedResidualPool(nn.Module):
    def __init__(self, feats):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        # two pool-block
        for i in range(1, 3):
            self.add_module(
                "block{}".format(i),
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=5, stride=1, padding=2),   # obtain the raw feature map size
                    nn.Conv2d(
                        feats,
                        feats,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False)))

    def forward(self, x):
        x = self.relu(x)
        path = x

        for i in range(1, 3):
            path = self.__getattr__("block{}".format(i))(path)
            x = x + path

        return x


class ChainedResidualPoolImproved(nn.Module):
    def __init__(self, feats):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        for i in range(1, 5):
            self.add_module(
                "block{}".format(i),
                nn.Sequential(
                    nn.Conv2d(
                        feats,
                        feats,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False),
                    nn.MaxPool2d(kernel_size=5, stride=1, padding=2)))

    def forward(self, x):
        x = self.relu(x)
        path = x

        for i in range(1, 5):
            path = self.__getattr__("block{}".format(i))(path)
            x += path

        return x


class BaseRefineNetBlock(nn.Module):
    def __init__(self, features, residual_conv_unit, multi_resolution_fusion,
                 chained_residual_pool, *shapes):
        super().__init__()

        for i, shape in enumerate(shapes):
            feats = shape[0]        # channel-num of this stage's output feature map
            self.add_module(
                "rcu{}".format(i),
                nn.Sequential(
                    residual_conv_unit(feats), residual_conv_unit(feats)))

        # stage-4 of ResNet needn't have to use 'multi_resolution_fusion'
        if len(shapes) != 1:
            self.mrf = multi_resolution_fusion(features, *shapes)
        else:
            self.mrf = None

        self.crp = chained_residual_pool(features)
        self.output_conv = residual_conv_unit(features)

    def forward(self, *xs):
        rcu_xs = []

        # multi-resolution input fusion
        for i, x in enumerate(xs):
            rcu_xs.append(self.__getattr__("rcu{}".format(i))(x))

        # Multi-resolution Fusion
        if self.mrf is not None:
            out = self.mrf(*rcu_xs)
        else:
            out = rcu_xs[0]

        # Chained Residual Pooling
        out = self.crp(out)

        # Output Conv.
        return self.output_conv(out)


class RefineNetBlock(BaseRefineNetBlock):
    def __init__(self, features, *shapes):
        super().__init__(features, ResidualConvUnit, MultiResolutionFusion,
                         ChainedResidualPool, *shapes)


class RefineNetBlockImprovedPooling(nn.Module):
    def __init__(self, features, *shapes):
        super().__init__(features, ResidualConvUnit, MultiResolutionFusion,
                         ChainedResidualPoolImproved, *shapes)

class MMFBlock(nn.Module):
    def __init__(self, features):
        super(MMFBlock, self).__init__()
        self.downchannel = features // 2

        self.relu = nn.ReLU(inplace=True)

        self.rgb_feature = nn.Sequential(
            nn.Conv2d(features, self.downchannel, kernel_size=1, stride=1, padding=0, bias=False),      # downsample

            # nonlinear_transformations
            ResidualConvUnit(self.downchannel),
            ResidualConvUnit(self.downchannel),

            nn.Conv2d(self.downchannel, features, kernel_size=3, stride=1, padding=1, bias=False)       # upsample
        )
        self.hha_feature = nn.Sequential(
            nn.Conv2d(features, self.downchannel, kernel_size=1, stride=1, padding=0, bias=False),      # downsample

            # nonlinear_transformations
            ResidualConvUnit(self.downchannel),
            ResidualConvUnit(self.downchannel),

            nn.Conv2d(self.downchannel, features, kernel_size=3, stride=1, padding=1, bias=False)       # upsample
        )

        self.ResidualPool = nn.Sequential(
                    nn.MaxPool2d(kernel_size=5, stride=1, padding=2),   # obtain the raw feature map size
                    nn.Conv2d(
                        features,
                        features,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False))

    def forward(self, rgb, hha):
        rgb_fea = self.rgb_feature(rgb)
        hha_fea = self.hha_feature(hha)
        fusion = self.relu(rgb_fea + hha_fea)
        x = self.ResidualPool(fusion)
        return fusion + x


class RDFNet(nn.Module):
    def __init__(self, num_classes = 3, features=256):
        super(RDFNet, self).__init__()
        self.rgb_backbone = None
        self.depth_backbone = None

        # MMF Block
        self.mmf1 = MMFBlock(256)
        self.mmf2 = MMFBlock(512)
        self.mmf3 = MMFBlock(1024)
        self.mmf4 = MMFBlock(2048)

        # modify the feature maps from each stage of RenNet, modify their channels
        self.layer1_rn = nn.Conv2d(
            256, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(
            512, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(
            1024, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(
            2048, 2 * features, kernel_size=3, stride=1, padding=1,
            bias=False)  # here, 2*fetures means we use two same stage-4 features as input

        self.refinenet4 = RefineNetBlock(2 * features,
                                         (2 * features, 2 * features))
        self.refinenet3 = RefineNetBlock(features,
                                         (2 * features, features),
                                         (features, features))
        self.refinenet2 = RefineNetBlock(features,
                                         (features, features),
                                         (features, features))
        self.refinenet1 = RefineNetBlock(features, (features, features),
                                         (features, features))

        self.output_conv = nn.Sequential(
            ResidualConvUnit(features), ResidualConvUnit(features),
            nn.Conv2d(
                features,
                num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True))


        # weight initial
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


        self.rgb_backbone = resnet_101()
        self.depth_backbone = resnet_101()

    def forward(self, rgb, depth):
        imsize = rgb.size()[2:]

        rgb = self.rgb_backbone.conv1(rgb)
        rgb = self.rgb_backbone.bn1(rgb)
        rgb = self.rgb_backbone.relu(rgb)
        rgb = self.rgb_backbone.maxpool(rgb)

        depth = self.depth_backbone.conv1(depth)
        depth = self.depth_backbone.bn1(depth)
        depth = self.depth_backbone.relu(depth)
        depth = self.depth_backbone.maxpool(depth)

        rgb1 = self.rgb_backbone.layer1(rgb)
        rgb2 = self.rgb_backbone.layer2(rgb1)
        rgb3 = self.rgb_backbone.layer3(rgb2)
        rgb4 = self.rgb_backbone.layer4(rgb3)

        depth1 = self.depth_backbone.layer1(depth)
        depth2 = self.depth_backbone.layer2(depth1)
        depth3 = self.depth_backbone.layer3(depth2)
        depth4 = self.depth_backbone.layer4(depth3)

        fusion1 = self.mmf1(rgb1, depth1)
        fusion2 = self.mmf2(rgb2, depth2)
        fusion3 = self.mmf3(rgb3, depth3)
        fusion4 = self.mmf4(rgb4, depth4)

        # print(fusion1.shape, fusion2.shape, fusion3.shape, fusion4.shape)

        # modify the number of channel
        layer_1_rn = self.layer1_rn(fusion1)
        layer_2_rn = self.layer2_rn(fusion2)
        layer_3_rn = self.layer3_rn(fusion3)
        layer_4_rn = self.layer4_rn(fusion4)

        path_4 = self.refinenet4(layer_4_rn)
        path_3 = self.refinenet3(path_4, layer_3_rn)
        path_2 = self.refinenet2(path_3, layer_2_rn)
        path_1 = self.refinenet1(path_2, layer_1_rn)
        out = self.output_conv(path_1)
        out = nn.functional.interpolate(out, size=imsize, mode='bilinear', align_corners=True)
        return out