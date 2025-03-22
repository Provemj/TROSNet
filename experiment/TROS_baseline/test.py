#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/12/4 16:09
# @Site :
# @File : test.py
# @Software: PyCharm

import os

import warnings
warnings.filterwarnings(action='ignore')

import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F

from config import cfg
from net.baseline import myNet
from net.sync_batchnorm.replicate import patch_replication_callback
from net.crf import GaussCRF, get_sun_conf
from utils.metrics import Evaluator
from utils.visualize import colored_segmap_cos
from datasets.dataLoader import test_dataloader

class Test:
    def __init__(self, args):
        self.mean = torch.FloatTensor([0.485, 0.456, 0.406])
        self.mean = self.mean.unsqueeze(1).unsqueeze(2)
        self.std = torch.FloatTensor([0.229, 0.224, 0.225])
        self.std = self.std.unsqueeze(1).unsqueeze(2)

        # Define Dataloader
        self.test_loader = test_dataloader(cfg)

        # Define network
        self.model = myNet(num_classes = cfg.MODEL_NUM_CLASSES, hha = cfg.DATA_HHA)
        if cfg.USE_CRF:
            crf_cfg = get_sun_conf()
            self.crf = GaussCRF(crf_cfg, [cfg.DATA_FIX_RESIZE[1], cfg.DATA_FIX_RESIZE[0]], cfg.MODEL_NUM_CLASSES)

        # Define Evaluator
        self.evaluator = Evaluator(cfg.MODEL_NUM_CLASSES)

        # Using cuda
        self.device = torch.device("cuda:" + str(cfg.GPU_IDS[0]) if torch.cuda.is_available() else "cpu")
        if cfg.GPU_IDS:
            self.model = torch.nn.DataParallel(self.model, device_ids = cfg.GPU_IDS)
            if cfg.USE_CRF:
                self.crf = torch.nn.DataParallel(self.crf, device_ids = cfg.GPU_IDS)
            patch_replication_callback(self.model)
            self.model = self.model.to(self.device)
            if cfg.USE_CRF:
                self.crf = self.crf.to(self.device)

        # Resuming checkpoint
        if cfg.TEST_CKPT is not None:
            if not os.path.isfile( cfg.TEST_CKPT):
                raise RuntimeError("=> no checkpoint found at '{}'".format( cfg.TEST_CKPT))

            checkpoint = torch.load(cfg.TEST_CKPT)

            if cfg.GPU_IDS:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])

            print("=> Loading checkpoint '{}'".format(cfg.TEST_CKPT))

    def test(self):
        self.model.eval()
        if cfg.USE_CRF:
            self.crf.eval()
        self.evaluator.reset()

        for i, sample in enumerate(self.test_loader):
            [batch, height, width] = sample['label'].size()
            print('iteration: %d / %d' %(i+1, len(self.test_loader)))
            with torch.no_grad():

                image = sample['rgb_%f'%(1.0)].to(self.device)
                depth = sample['depth_%f'%(1.0)].to(self.device)

                output = self.model(rgb=image, depth=depth, phase='test')
                # output = self.crf(output, image, depth)

            output = F.interpolate(output, size=(height, width), scale_factor=None, mode='bilinear', align_corners=True)

            pred = torch.max(output, dim = 1)[1]
            pred = pred.data.cpu().numpy()
            target = sample['label'].cpu().numpy()

            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

            # Save some images
            rgb = sample['rgb_%f'%(1.0)][0, :, :, :].data.cpu()
            rgb = ((rgb * self.std) + self.mean) * 255.0
            rgb = rgb.permute(1, 2, 0).numpy()

            pred_label = pred[0,:,:]
            gt_label = target[0,:,:]


            if cfg.TEST_SAVE_IMAGE:
                if i % 1 == 0:
                    # merge = np.hstack([rgb, colored_segmap_cos(gt_label, cfg.MODEL_NUM_CLASSES), colored_segmap_cos(pred_label, cfg.MODEL_NUM_CLASSES)])
                    merge = colored_segmap_cos(pred_label, cfg.MODEL_NUM_CLASSES)
                    merge = np.uint8(merge)
                    merge = Image.fromarray(merge)
                    merge.save(os.path.join(cfg.IMAGE_DIR, str(i) + ".png"))

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        mAcc = self.evaluator.Pixel_Accuracy_Class()
        Acc_class = self.evaluator.Class_Accuracy()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        IoU = self.evaluator.Intersection_over_Union()

        print('=> Result :')
        print("Acc:{}, Acc_class:{}, IoU:{}, fwIoU: {}".format(Acc, mAcc, mIoU, FWIoU))

        # save result
        with open(os.path.join(cfg.LOG_DIR, 'evaluation_result.txt'), 'w') as f:
            f.write("Acc: " + str(Acc) + "\n")
            f.write("mAcc: " + str(mAcc) + "\n")
            f.write("Acc_class: " + str(Acc_class) + "\n")
            f.write("mIoU: " + str(mIoU) + "\n")
            f.write("FWIoU: " + str(FWIoU) + "\n")
            f.write("IoU: " + str(IoU) + "\n")




def main():
    test = Test(cfg)
    print("=> Evaluation ...")
    test.test()
    print("=> Test finished!")

if __name__ == "__main__":
    main()