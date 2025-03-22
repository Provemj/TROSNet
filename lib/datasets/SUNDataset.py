#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2019/12/3 21:56 
# @Site :  
# @File : SUNDataset.py 
# @Software: PyCharm

import os

import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from datasets import transforms

SUNRGBD_rootdir = "/data1/zhangguodong/RGBDsegmentation/mySunRGBD/"

class SUNDataset(Dataset):
    def __init__(self, cfg, type = "train"):
        super(SUNDataset, self).__init__()

        self.rootdir = SUNRGBD_rootdir

        if type == "train":
            self.transform = self.transform_train
            self.datalist = []
            with open(os.path.join(self.rootdir, "train.txt"), "r") as f :
                self.datalist = f.readlines()
        elif type == "val":
            self.transform = self.transform_val
            self.datalist = []
            with open(os.path.join(self.rootdir, "test.txt"), "r") as f :
                self.datalist = f.readlines()
        elif type == "test":
            self.transform = self.transform_test
            self.datalist = []
            with open(os.path.join(self.rootdir, "test.txt"), "r") as f :
                self.datalist = f.readlines()
        else:
            raise NotImplementedError

        self.cfg = cfg

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        sample, rgb, depth, label = {}, None, None, None
        if not self.cfg.DATA_HHA:
            rgb = Image.open(os.path.join(self.rootdir, self.datalist[index][:-1], "image", "rgb.png")).convert('RGB')
            depth = Image.open(os.path.join(self.rootdir, self.datalist[index][:-1], "depth_bfx", "depth.png"))
            label = Image.open(os.path.join(self.rootdir, self.datalist[index][:-1], "label", "label.png"))
        else:
            rgb = Image.open(os.path.join(self.rootdir, self.datalist[index][:-1], "image","rgb.png")).convert('RGB')
            depth = Image.open(os.path.join(self.rootdir, self.datalist[index][:-1], "hha", "hha.png")).convert('RGB')
            label = Image.open(os.path.join(self.rootdir, self.datalist[index][:-1], "label", "label.png"))

        sample = {"rgb": rgb, "depth": depth, "label": label}
        return self.transform(sample)

    def transform_train(self, sample):
        composed_transforms = torchvision.transforms.Compose([transforms.RandomResize(self.cfg.DATA_RESCALE),
                                                              transforms.RandomCrop(self.cfg.DATA_RANDOM_CROP),
                                                              transforms.RandomRotate(self.cfg.DATA_RANDOM_ROTATE),
                                                              transforms.RandomFlip(),
                                                              transforms.Multiscale_output(self.cfg.TRAIN_MULTI_SCALE_OUTPUT),
                                                              transforms.Normalize(hha = self.cfg.DATA_HHA,
                                                                                   mean = (0.485, 0.456, 0.406),
                                                                                   std = (0.229, 0.224, 0.225)),
                                                              transforms.ToTensor()])
        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = torchvision.transforms.Compose([transforms.FixResize(self.cfg.DATA_FIX_RESIZE),
                                                              transforms.Multiscale_output(self.cfg.TRAIN_MULTI_SCALE_OUTPUT),
                                                              transforms.Normalize(hha = self.cfg.DATA_HHA),
                                                              transforms.ToTensor()])
        return composed_transforms(sample)

    def transform_test(self, sample):
        composed_transforms = torchvision.transforms.Compose([transforms.FixResize(self.cfg.DATA_FIX_RESIZE),
                                                              transforms.Multiscale_input(self.cfg.TEST_MULTI_SCALE_INPUT),
                                                              transforms.Normalize(hha = self.cfg.DATA_HHA),
                                                              transforms.ToTensor()])
        return composed_transforms(sample)