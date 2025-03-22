#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/10/29 20:24
# @Site :
# @File : datasets.py
# @Software: PyCharm
import os

import torchvision
from imageio import imread
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader

from datasets import transforms

TROS_rootdir = "/data1/zhangguodong/cord"

class TROSDataset(Dataset):
    def __init__(self, cfg, type = "train"):
        super(TROSDataset, self).__init__()

        self.rootdir = TROS_rootdir

        if type == "train":
            self.transform = self.transform_train
            self.datalist = os.listdir(os.path.join(TROS_rootdir,"train","mask"))
            self.datalist = [i.split(".")[0] for i in self.datalist]
            self.rootdir = os.path.join(TROS_rootdir,"train")
        elif type == "val":
            self.transform = self.transform_val
            self.datalist = os.listdir(os.path.join(TROS_rootdir,"test","mask"))
            self.datalist  = [i.split(".")[0] for i in self.datalist]
            self.rootdir = os.path.join(TROS_rootdir, "test")
        elif type == "test":
            self.transform = self.transform_test
            self.datalist = os.listdir(os.path.join(TROS_rootdir,"test","mask"))
            self.datalist  = sorted([i.split(".")[0] for i in self.datalist])
            self.rootdir = os.path.join(TROS_rootdir, "test")
        else:
            raise NotImplementedError

        self.cfg = cfg

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        sample, rgb, depth, label = {}, None, None, None
        if not self.cfg.DATA_HHA:
            rgb = Image.open(os.path.join(self.rootdir, "rgb", self.datalist[index] + ".jpg")).convert('RGB')
            depth = Image.open(os.path.join(self.rootdir, "depth", self.datalist[index] + ".png"))
            label = Image.open(os.path.join(self.rootdir, "mask", self.datalist[index] + ".png"))
        else:
            rgb = Image.open(os.path.join(self.rootdir, "rgb", self.datalist[index] + ".jpg")).convert('RGB')
            depth = Image.open(os.path.join(self.rootdir, "hha", self.datalist[index] + ".png")).convert('RGB')
            label = Image.open(os.path.join(self.rootdir, "mask", self.datalist[index] + ".png"))

        sample = {"rgb": rgb, "depth": depth, "label": label}
        return self.transform(sample)

    def transform_train(self, sample):
        composed_transforms = torchvision.transforms.Compose([transforms.RandomResize(self.cfg.DATA_RESCALE),
                                                              transforms.RandomCrop(self.cfg.DATA_RANDOM_CROP),
                                                              transforms.RandomRotate(self.cfg.DATA_RANDOM_ROTATE),
                                                              transforms.RandomFlip(),
                                                              transforms.Multiscale_output(self.cfg.TRAIN_MULTI_SCALE_OUTPUT),
                                                              transforms.Normalize(hha = self.cfg.DATA_HHA),
                                                              transforms.ToTensor()])
        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = torchvision.transforms.Compose([transforms.FixResize(self.cfg.DATA_FIX_RESIZE),
                                                              transforms.Multiscale_output(self.cfg.TRAIN_MULTI_SCALE_OUTPUT),
                                                              transforms.Normalize(hha = self.cfg.DATA_HHA),
                                                              transforms.ToTensor()])
        return composed_transforms(sample)

    def transform_test(self, sample):
        composed_transforms = torchvision.transforms.Compose([transforms.Multiscale_input(self.cfg.TEST_MULTI_SCALE_INPUT),
                                                              transforms.Normalize(hha = self.cfg.DATA_HHA),
                                                              transforms.ToTensor()])
        return composed_transforms(sample)