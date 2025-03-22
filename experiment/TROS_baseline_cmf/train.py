#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/12/3 19:45
# @Site :
# @File : train.py
# @Software: PyCharm
import  os
from math import ceil

import warnings
warnings.filterwarnings(action='ignore')

import torch
import numpy as np

from config import cfg
from net.baseline_cmf import myNet
from net.sync_batchnorm.replicate import patch_replication_callback
from utils.metrics import Evaluator
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver, TensorboardSummary
from utils.losses import SegmentationLosses, calculate_weigths_labels
from datasets.dataLoader import trainval_dataloader

class Trainer:
    def __init__(self):
        # Init
        print('=> Initializing ...')
        print('=> ' + cfg.DATA_NAME)
        self.device = torch.device("cuda:" + str(cfg.GPU_IDS[0]) if torch.cuda.is_available() else "cpu")
        self.iteration = 0

        if cfg.TRAIN_MULTI_SCALE_OUTPUT:
            self.label_list = ['label'] + ['label_%f'%r for r in cfg.TRAIN_MULTI_SCALE_OUTPUT]
        else:
            self.label_list = ['label']

        # Saver
        self.saver = Saver(cfg)
        self.saver.save_experiment_config()

        # Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.log_dir)
        self.writer = self.summary.create_summary()

        # Dataloader
        self.train_loader, self.val_loader = trainval_dataloader(cfg)

        # Model
        model = myNet(num_classes = cfg.MODEL_NUM_CLASSES, hha = cfg.DATA_HHA)

        # Optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr = cfg.TRAIN_LR, momentum = cfg.TRAIN_MOMENTUM,
                                    weight_decay = cfg.TRAIN_WEIGHT_DECAY)

        # Losses
        if cfg.TRAIN_LOSS_BALANCE:
            classes_weights_path = os.path.join(cfg.ROOT_DIR, 'weights', cfg.DATA_NAME + '_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                print("=> Loading loss balanced weights ...")
                weight = np.load(classes_weights_path)
            else:
                print("=> Caculating loss balanced weights ...")
                weight = calculate_weigths_labels(cfg.MODEL_NUM_CLASSES, self.train_loader, cfg.DATA_NAME, os.path.join(cfg.ROOT_DIR, 'weights'))
            weight = torch.from_numpy(weight.astype(np.float32))
            weight = weight.to(self.device)
        else:
            weight = None

        self.criterion = SegmentationLosses(weight = weight, device = self.device).build_loss(mode = cfg.TRAIN_LOSS_TYPE, mulout = cfg.TRAIN_MULTI_SCALE_OUTPUT)
        self.model, self.optimizer = model, optimizer

        # Metrics
        self.evaluator = Evaluator(cfg.MODEL_NUM_CLASSES)

        # Lr schedular
        self.scheduler = LR_Scheduler(cfg.TRAIN_LR, cfg.TRAIN_EPOCHS * len(self.train_loader), cfg.TRAIN_LR_POWER)

        # Gpu
        if cfg.GPU_IDS:
            self.model = torch.nn.DataParallel(self.model, device_ids = cfg.GPU_IDS)
            patch_replication_callback(self.model)
            self.model = self.model.to(self.device)

        # Load checkpoint
        self.best_pred_Acc = 0.0
        self.best_pred_mAcc = 0.0
        self.best_pred_mIoU = 0.0
        self.best_pred_FWIoU = 0.0

        if cfg.TRAIN_CKPT is not None:
            if not os.path.isfile(cfg.TRAIN_CKPT):
                raise RuntimeError("=> no checkpoint found at '{}'".format(cfg.TRAIN_CKPT))
            checkpoint = torch.load(cfg.TRAIN_CKPT)
            cfg.TRAIN_START_EPOCH = checkpoint['epoch']
            if cfg.GPU_IDS:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])

            if not cfg.TRAIN_FINETUNE:
                self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.best_pred_mIoU = checkpoint['best_pred_mIoU']
            print("=> loaded checkpoint '{}' (epoch {})".format(cfg.TRAIN_CKPT, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if cfg.TRAIN_FINETUNE:
            cfg.TRAIN_START_EPOCH = 0

    def training(self, epoch):
        self.model.train()
        train_loss = 0.0
        num_img_tr = len(self.train_loader)

        for i, sample in enumerate(self.train_loader):
            self.iteration += 1
            # self.scheduler(self.optimizer, self.iteration)
            image, depth = sample['rgb'].to(self.device), sample['depth'].to(self.device)
            target = [sample[s].to(self.device) for s in self.label_list]

            self.optimizer.zero_grad()
            output = self.model(rgb = image, depth = depth, phase = 'train')
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

            print('Epoch: %d / %d,iteration: %d / %d, avgloss: %.5f, lr: %.8f' % (epoch + 1, cfg.TRAIN_EPOCHS, i + 1,
                                                                                  num_img_tr, train_loss / (i+1),
                                                                                  self.optimizer.param_groups[0]['lr']))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

        self.writer.add_scalar('train/average_loss_epoch', train_loss / num_img_tr, epoch + 1)
        print('=> Epoch %d / %d, average_loss: %.5f' %(epoch + 1, cfg.TRAIN_EPOCHS, train_loss / num_img_tr))

    def val(self, epoch):
        self.model.eval()
        val_loss = 0.0
        num_img_te = len(self.val_loader)

        self.evaluator.reset()
        for i, sample in enumerate(self.val_loader):
            image, depth= sample['rgb'].to(self.device), sample['depth'].to(self.device)
            target = [sample[s].to(self.device) for s in self.label_list]

            with torch.no_grad():
                output = self.model(rgb = image, depth = depth, phase = 'val')

            loss = self.criterion(output, target)
            val_loss += loss.item()
            print('Epoch: %d / %d,iteration: %d / %d, val loss: %.5f' % (epoch + 1, cfg.TRAIN_EPOCHS, i + 1, num_img_te, val_loss / (i + 1)))

            pred = torch.max(output[0], dim = 1)[1]
            pred = pred.data.cpu().numpy()
            target = target[0].cpu().numpy()
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast val during the training
        Acc = self.evaluator.Pixel_Accuracy()
        mAcc = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/average_loss_epoch', val_loss / num_img_te, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch + 1)
        self.writer.add_scalar('val/mAcc', mAcc, epoch + 1)
        self.writer.add_scalar('val/mIoU', mIoU, epoch + 1)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch + 1)
        print('=> val:')
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, mAcc, mIoU, FWIoU))
        print('Epoch %d / %d, average_loss: %.5f' %(epoch + 1, cfg.TRAIN_EPOCHS, val_loss / num_img_te))

        if mIoU > self.best_pred_mIoU:
            self.best_pred_Acc = Acc
            self.best_pred_mAcc = mAcc
            self.best_pred_mIoU = mIoU
            self.best_pred_FWIoU = FWIoU
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred_Acc': self.best_pred_Acc,
                'best_pred_mAcc': self.best_pred_mAcc,
                'best_pred_mIoU': self.best_pred_mIoU,
                'best_pred_FWIoU': self.best_pred_FWIoU,
            })

        if (epoch + 1 == cfg.TRAIN_EPOCHS):
            self.best_pred_Acc = Acc
            self.best_pred_mAcc = mAcc
            self.best_pred_mIoU = mIoU
            self.best_pred_FWIoU = FWIoU
            self.saver.save_checkpoint_last({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred_Acc': self.best_pred_Acc,
                'best_pred_mAcc': self.best_pred_mAcc,
                'best_pred_mIoU': self.best_pred_mIoU,
                'best_pred_FWIoU': self.best_pred_FWIoU,
            })

def main():

    trainer = Trainer()

    for epoch in range(cfg.TRAIN_START_EPOCH, cfg.TRAIN_EPOCHS):
        trainer.training(epoch)
        trainer.val(epoch)

    print('=> Train finished!')
    trainer.writer.close()


if __name__ == "__main__":
   main()