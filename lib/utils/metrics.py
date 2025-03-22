#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2019/10/31 15:42 
# @Software: PyCharm
import torch
import math
import numpy as np


class Evaluator:
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,self.num_class))

    def Pixel_Accuracy(self):
        # 取混淆矩阵对角线的总和，除以所有样本
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Class_Accuracy(self):
        mAcc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        return mAcc

    def Pixel_Accuracy_Class(self):
        # 每一类的准确率
        mAcc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        # 防止有np.nan的值出现, 求均值分母也要去掉nan类的数量
        mAcc = np.nanmean(mAcc)
        return mAcc

    def Mean_Intersection_over_Union(self):
        mIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        mIoU = np.nanmean(mIoU)
        return mIoU

    def Intersection_over_Union(self):
        mIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        return mIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        # 混淆矩阵 行为gt 列为 pred
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        # pre_image[mask] shape变为（m,）
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength = self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)

        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)