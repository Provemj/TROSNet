#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2019/12/3 22:29 
# @Software: PyCharm
import os
import glob
import shutil
from collections import  OrderedDict

import torch
from tensorboardX import SummaryWriter

class TensorboardSummary:
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

class Saver:
    def __init__(self, cfg):
        self.cfg = cfg

        self.log_dir = self.cfg.LOG_DIR
        self.model_dir = self.cfg.MODEL_SAVE_DIR

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def save_checkpoint(self, state):
        """Saves checkpoint to disk"""

        best_pred_Acc = state['best_pred_Acc']
        best_pred_mAcc = state['best_pred_mAcc']
        best_pred_mIoU = state['best_pred_mIoU']
        best_pred_FWIoU = state['best_pred_FWIoU']
        epoch = state["epoch"]

        with open(os.path.join(self.log_dir, 'best_pred.txt'), 'w') as f:
            f.write("mIoU: " + str(best_pred_mIoU) + "\n")
            f.write("Acc: " + str(best_pred_Acc) + "\n")
            f.write("mAcc: " + str(best_pred_mAcc) + "\n")
            f.write("FWIoU: " + str(best_pred_FWIoU) + "\n")
            f.write("Epoch: " + str(epoch) + "\n")

        torch.save(state, os.path.join(self.model_dir, 'model_best.pth.tar'))

    def save_checkpoint_last(self, state):
        """Saves checkpoint to disk"""

        best_pred_Acc = state['best_pred_Acc']
        best_pred_mAcc = state['best_pred_mAcc']
        best_pred_mIoU = state['best_pred_mIoU']
        best_pred_FWIoU = state['best_pred_FWIoU']
        epoch = state["epoch"]

        with open(os.path.join(self.log_dir, 'last_pred.txt'), 'w') as f:
            f.write("mIoU: " + str(best_pred_mIoU) + "\n")
            f.write("Acc: " + str(best_pred_Acc) + "\n")
            f.write("mAcc: " + str(best_pred_mAcc) + "\n")
            f.write("FWIoU: " + str(best_pred_FWIoU) + "\n")
            f.write("Epoch: " + str(epoch) + "\n")

        torch.save(state, os.path.join(self.model_dir, 'model_last.pth.tar'))

    def save_experiment_config(self):
        logfile = os.path.join(self.log_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['datset'] = self.cfg.DATA_NAME
        p['hha'] = self.cfg.DATA_HHA
        p['backbone'] = self.cfg.MODEL_BACKBONE
        p['lr'] = self.cfg.TRAIN_LR
        p['momentum'] = self.cfg.TRAIN_MOMENTUM
        p['lr_power'] = self.cfg.TRAIN_LR_POWER
        p['weight_decay'] = self.cfg.TRAIN_WEIGHT_DECAY
        p['loss_type'] = self.cfg.TRAIN_LOSS_TYPE
        p['num_epochs'] = self.cfg.TRAIN_EPOCHS
        p['batch_size'] = self.cfg.TRAIN_BATCHES
        p['train_output_scale'] = self.cfg.TRAIN_MULTI_SCALE_OUTPUT
        p['test_input_scale'] = self.cfg.TEST_MULTI_SCALE_INPUT
        p['test_flip'] = self.cfg.TEST_FLIP

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()