#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/12/3 19:46
# @Site :
# @File : config.py
# @Software: PyCharm
import os
import sys

"""
baseline 6,7   my
baseline_ca 4,5 my2
baseline_cmf 2,3 my3
"""

class Configuration:
    def __init__(self):
        self.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', '..'))
        self.EXP_NAME = 'cosnet_baseline'

        self.DATA_NAME = 'COS'
        self.DATA_HHA = False
        self.DATA_WORKERS = 4
        self.DATA_RESCALE = 1.5
        self.DATA_RANDOM_CROP = (640, 480)
        self.DATA_FIX_RESIZE = (640, 480)
        self.DATA_RANDOM_ROTATE = 5.0

        self.MODEL_BACKBONE = 'resnet50'                    # ['resnet50', 'drn']
        self.MODEL_NUM_CLASSES = 3
        self.MODEL_SAVE_DIR = os.path.join(self.ROOT_DIR, 'model', self.EXP_NAME)
        self.MODEL_SAVE_INTERVAL = 1

        self.TRAIN_START_EPOCH = 0
        self.TRAIN_EPOCHS = 500
        self.TRAIN_MULTI_SCALE_OUTPUT = [1/2,1/4,1/8]   # 数据处理时默认包含了 factor == 1
        self.TRAIN_LOSS_TYPE = 'ce'                          # ['ce', 'focal']
        self.TRAIN_LOSS_BALANCE = True
        self.TRAIN_LR = 7e-4
        self.TRAIN_LR_POWER = 0.9
        self.TRAIN_MOMENTUM = 0.9
        self.TRAIN_BN_MOM = 0.0003
        self.TRAIN_WEIGHT_DECAY = 1e-4
        self.TRAIN_BATCHES = 8
        self.TRAIN_CKPT = None
        self.TRAIN_FINETUNE = False

        self.GPU_IDS = [4,5]
        self.LOG_DIR = os.path.join(self.ROOT_DIR, 'log', self.EXP_NAME)
        self.IMAGE_DIR = os.path.join(self.ROOT_DIR, 'image', self.EXP_NAME)
        self.WEIGHTS_DIR = os.path.join(self.ROOT_DIR, 'weights')

        self.TEST_MULTI_SCALE_INPUT = [0.5, 0.75, 1.0, 1.25]
        self.TEST_FLIP = True
        self.TEST_CKPT = os.path.join(self.MODEL_SAVE_DIR, 'model_best.pth.tar')
        self.TEST_BATCHES = 16
        self.TEST_SAVE_IMAGE = False

        self.USE_CRF = False

        self.__check()
        self.__add_path(os.path.join(self.ROOT_DIR, 'lib'))

    def __check(self):
        if not os.path.isdir(self.LOG_DIR):
            os.makedirs(self.LOG_DIR)

        if not os.path.isdir(self.MODEL_SAVE_DIR):
            os.makedirs(self.MODEL_SAVE_DIR)

        if not os.path.isdir(self.IMAGE_DIR):
            os.makedirs(self.IMAGE_DIR)

    def __add_path(self, path):
        if path not in sys.path:
            sys.path.insert(0, path)

cfg = Configuration()