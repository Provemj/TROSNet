#！/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time : 2020/2/20 20:36
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.resnet import resnet_50

class SSMA(nn.Module):
    def __init__(self, inchannel):
        super(SSMA, self).__init__()
        self.conv1 = nn.Conv2d(2*inchannel, inchannel // 16,kernel_size=3,stride=1,padding=1)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(inchannel // 16, 2*inchannel,kernel_size=3,stride=1,padding=1)
        self.sigmoid = nn.Sigmoid()
        self.conv3 = nn.Conv2d(2*inchannel, inchannel,kernel_size=3,stride=1,padding=1)
        self.bn = nn.BatchNorm2d(inchannel)

    def forward(self, rgb, depth):
        rgb = torch.cat([rgb, depth], dim=1)
        depth = self.conv1(rgb)
        depth = self.relu(depth)
        depth = self.conv2(depth)
        depth = self.sigmoid(depth)
        depth = self.conv3(depth)
        depth = self.bn(depth)

        return depth * rgb

class Encoder_easpp(nn.Module):
    def __init__(self, isd = False):
        super(Encoder_easpp, self).__init__()
        self.eAspp_rate = [3, 6, 8]

        self.encoder = None

        self.isd = isd

        if self.isd:
            self.conv1_d = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                     bias=False)

        self.skip1 = nn.Sequential(
            nn.Conv2d(256, 24, kernel_size=1,stride=1),
            nn.BatchNorm2d(24)
        )

        self.skip2 = nn.Sequential(
            nn.Conv2d(512, 24, kernel_size=1,stride=1),
            nn.BatchNorm2d(24)
        )
        
        #  eAspp
        self.IA = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1,stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.IB = nn.Sequential(
            nn.Conv2d(2048, 64, kernel_size=1,stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding =self.eAspp_rate[0], dilation=self.eAspp_rate[0]),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=self.eAspp_rate[0], dilation=self.eAspp_rate[0]),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.IC = nn.Sequential(
            nn.Conv2d(2048, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=self.eAspp_rate[1], dilation=self.eAspp_rate[1]),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=self.eAspp_rate[1], dilation=self.eAspp_rate[1]),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.ID = nn.Sequential(
            nn.Conv2d(2048, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=self.eAspp_rate[2], dilation=self.eAspp_rate[2]),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=self.eAspp_rate[2], dilation=self.eAspp_rate[2]),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.IE = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        self.out = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
        )

        # weight initial
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.ecoder = resnet_50()

    def forward(self, x):
        if self.isd:
            x = self.conv1_d(x)
        else:
            x = self.ecoder.conv1(x)
        x = self.ecoder.bn1(x)
        x = self.ecoder.relu(x)
        x = self.ecoder.maxpool(x)
        x = self.ecoder.layer1(x)
        s1 = self.skip1(x)
        x = self.ecoder.layer2(x)
        s2 = self.skip2(x)
        x = self.ecoder.layer3(x)
        x = self.ecoder.layer4(x)

        a = self.IA(x)
        b = self.IB(x)
        c = self.IC(x)
        d = self.ID(x)
        e = self.avg(x)
        e = e.expand(-1, -1, x.shape[2], x.shape[3])
        e = self.IE(e)

        x = self.out(torch.cat([a,b,c,d,e], dim=1))

        return s1, s2, x

class net(nn.Module):
    def __init__(self, num_classes = 3):
        super(net, self).__init__()

        self.ssma_red = nn.Sequential(
            nn.Conv2d(512, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(32, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid(),
        )

        self.ssma_s1_red = nn.Sequential(
            nn.Conv2d(48, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(8, 48, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid(),
        )

        self.ssma_s2_red = nn.Sequential(
            nn.Conv2d(48, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(8, 48, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
        )

        self.s1_out = nn.Conv2d(48, 24, kernel_size=3, stride=1, padding=1, bias=True)
        self.s2_out = nn.Conv2d(48, 24, kernel_size=3, stride=1, padding=1, bias=True)

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(256),
        )

        self.dea1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 24, kernel_size=1, stride=1),
            nn.BatchNorm2d(24),
            nn.ReLU(True),
        )

        self.up1 = nn.Sequential(
            nn.Conv2d(280, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
        )

        self.dea2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 24, kernel_size=1, stride=1),
            nn.BatchNorm2d(24),
            nn.ReLU(True),
        )

        self.out1 = nn.Sequential(
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_classes),
        )

        self.out2 = nn.Sequential(
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_classes),
        )

        self.final = nn.Sequential(
            nn.Conv2d(280, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(num_classes),
        )

        # weight initial
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.rgb_block = Encoder_easpp()
        self.depth_block = Encoder_easpp(isd = True)


    def forward(self, rgb, depth):
        # 24-120-160
        # 24-60-80
        # 256-15-20
        rgb_s1, rgb_s2, rgb_e = self.rgb_block(rgb)
        depth_s1, depth_s2, depth_e = self.depth_block(depth)

        in1 = torch.cat([rgb_e, depth_e], dim = 1)
        s1_in1 = torch.cat([rgb_s2, depth_s2], dim = 1)
        s2_in1 = torch.cat([rgb_s1, depth_s1], dim = 1)
        in1 = self.ssma_red(in1) * in1
        s1_in1 = self.ssma_s1_red(s1_in1) * s1_in1
        s2_in1 = self.ssma_s2_red(s2_in1) * s2_in1

        in1 = self.conv3(in1)
        s1_in1 = self.s1_out(s1_in1)
        s2_in1 = self.s2_out(s2_in1)

        in1 = self.deconv1(in1)
        out1 = self.out1(in1)
        out1 = F.interpolate(out1, size=[rgb.size(2), rgb.size(3)], mode='bilinear', align_corners=True)
        s1_in1 = self.dea1(in1) * s1_in1

        in1 = self.up1(torch.cat([in1, s1_in1], dim = 1))
        in1 = self.deconv2(in1)
        out2 = self.out2(in1)
        out2 = F.interpolate(out2, size=[rgb.size(2), rgb.size(3)], mode='bilinear', align_corners=True)
        s2_in1 = self.dea2(in1) * s2_in1

        out3 = self.final(torch.cat([in1, s2_in1], dim = 1))

        return [out1, out2, out3]


if __name__ == "__main__":
    a = torch.randn(1, 3, 480, 640)
    b = torch.randn(1, 3, 480, 640)
    my = net()
    my.eval()
    s1, s2, b = my(a, b)
    print(s1.shape)
    print(s2.shape)
    print(b.shape)