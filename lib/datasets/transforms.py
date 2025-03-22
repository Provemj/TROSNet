#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2019/10/29 21:09 
# @Site :  
# @File : transforms.py 
# @Software: PyCharm
import random

import torch
import numpy as np
from PIL import Image

class RandomResize:
    def __init__(self, resize_factor):
        self.factor = resize_factor

    def __call__(self, sample):
        random_scale = random.uniform(1.0, self.factor)
        rgb = sample['rgb']
        depth = sample['depth']
        label = sample['label']

        W, H = rgb.size
        W, H = int(round(random_scale * W)), int(round(random_scale * H))
        rgb = rgb.resize((W, H), Image.BILINEAR)
        depth = depth.resize((W, H), Image.NEAREST)
        label = label.resize((W, H), Image.NEAREST)

        return {'rgb': rgb, 'depth': depth, 'label': label, 'size':[H, W]}

class RandomCrop:
    def __init__(self, size = (640, 480)):
        self.size = size

    def __call__(self, sample):
        rgb = sample['rgb']
        depth = sample['depth']
        label = sample['label']

        W, H = rgb.size
        nW, nH = self.size

        if W > nW and H > nH:
            i, j = random.randint(0, W - nW), random.randint(0, H - nH)
            rgb = rgb.crop((i, j, i + nW, j + nH))
            depth = depth.crop((i, j, i + nW, j + nH))
            label = label.crop((i, j, i + nW, j + nH))
        else:
            rgb = rgb.resize(self.size, Image.BILINEAR)
            depth = depth.resize(self.size, Image.NEAREST)
            label = label.resize(self.size, Image.NEAREST)

        return {'rgb': rgb, 'depth': depth, 'label': label, 'size':sample['size']}

class FixResize:
    def __init__(self, size = None):
        self.size = size

    def __call__(self, sample):
        if self.size:
            rgb = sample['rgb']
            depth = sample['depth']
            label = sample['label']

            rgb = rgb.resize(self.size, Image.BILINEAR)
            depth = depth.resize(self.size, Image.NEAREST)
            label = label.resize(self.size, Image.NEAREST)

            if 'size' in sample:
                return {'rgb': rgb, 'depth': depth, 'label': label, 'size':sample['size']}
            else:
                W, H = rgb.size
                return {'rgb': rgb, 'depth': depth, 'label': label, 'size': [H, W]}
        else:
            return  sample

class RandomRotate:
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        rgb = sample['rgb']
        depth = sample['depth']
        label = sample['label']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        rgb = rgb.rotate(rotate_degree, Image.BILINEAR, fillcolor = 0)
        depth = depth.rotate(rotate_degree, Image.NEAREST, fillcolor = 0)
        label = label.rotate(rotate_degree, Image.NEAREST, fillcolor = 255)

        return {'rgb': rgb, 'depth': depth, 'label': label, 'size':sample['size']}

class RandomFlip:
    def __call__(self, sample):
        rgb, depth, label = sample['rgb'], sample['depth'], sample['label']
        if random.random() > 0.5:
            rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        return {'rgb': rgb, 'depth': depth, 'label': label, 'size':sample['size']}

class Multiscale_input:
    def __init__(self, mulscale_rate=None):
        self.rate = mulscale_rate

    def __call__(self, sample):
        if self.rate:
            rgb, depth = sample['rgb'], sample['depth']
            W, H = depth.size
            for r in self.rate:
                sample['depth_%f' % r] = depth.resize((int(W * r), int(H * r)), Image.NEAREST)
                sample['rgb_%f' % r] = rgb.resize((int(W * r), int(H * r)), Image.BILINEAR)

        return sample

class Multiscale_output:
    def __init__(self, mulscale_rate = None):
        self.rate = mulscale_rate

    def __call__(self, sample):
        if self.rate:
            label = sample['label']
            W, H = label.size
            for r in self.rate:
                sample['label_%f'%r] = label.resize((int(W * r), int(H * r)), Image.NEAREST)

        return sample


class Normalize:
    def __init__(self, hha = False, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.hha = hha
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        key_list = sample.keys()

        for k in key_list:
            if 'rgb' in k:
                sample[k] = np.array(sample[k]).astype(np.float32)
                sample[k] /= 255.0
                sample[k] -= self.mean
                sample[k] /= self.std
            elif 'depth' in k:
                sample[k] = np.array(sample[k]).astype(np.float32)
                if self.hha:
                    sample[k] /= 255.0
                # else:
                #     sample[k] /= 1000.0
            elif 'label' in k:
                sample[k] = np.array(sample[k]).astype(np.float32)

        return sample

class ToTensor:
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        key_list = sample.keys()

        for k in key_list:
            if 'rgb' in k:
                sample[k] = sample[k].transpose((2, 0, 1))
                sample[k] = torch.from_numpy(sample[k]).float()
            elif 'depth' in k:
                if len(sample[k].shape) > 2:
                    sample[k] = sample[k].transpose(2,0,1)
                    sample[k] = torch.from_numpy(sample[k]).float()
                    # sample[k] = sample[k].unsqueeze(0)
                else:
                    sample[k] = torch.from_numpy(sample[k]).float() / 1000.0
                    sample[k] = sample[k].unsqueeze(0)
            elif 'label' in k:
                sample[k] = torch.from_numpy(sample[k]).float()

        return sample