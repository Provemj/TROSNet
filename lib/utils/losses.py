#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2019/10/31 14:14 
# @Software: PyCharm
import os

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse

# size_average 将所得的Loss / H * W * B
class SegmentationLosses:
    def __init__(self, weight = None, ignore_index = 255, device = None):
        self.ignore_index = ignore_index
        self.weight = weight
        self.device = device

    def build_loss(self, mode='ce', mulout = None):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            if not mulout:
                return self.CrossEntropyLoss
            else:
                return self.mulCrossEntropyLoss
        elif mode == 'focal':
            if not mulout:
                return self.FocalLoss
            else:
                return self.mulFocalLoss
        elif mode == 'lovasz':
            if not mulout:
                return self.Lovasz
            else:
                return self.mulLovasz
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        criterion = nn.CrossEntropyLoss(weight = self.weight, ignore_index = self.ignore_index, reduction='mean')
        if self.device:
            criterion = criterion.to(self.device)

        loss = criterion(logit, target.long())

        return loss

    def FocalLoss(self, logit, target, gamma = 2, alpha = 0.5):
        criterion = nn.CrossEntropyLoss(weight = self.weight, ignore_index = self.ignore_index, reduction='mean')
        if self.device:
            criterion = criterion.to(self.device)

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        return loss

    def Lovasz(self, logits, targets):

        return lovasz_softmax(F.softmax(logits, dim=1), targets, ignore=self.ignore_index)

    def mulCrossEntropyLoss(self, logits, targets, phase = 'train'):
        # reduce = False 返回所有元素对应的loss
        if phase == 'train':
            losses = []
            for logit, target in zip(logits, targets):
                loss_scale = self.CrossEntropyLoss(logit, target)
                losses.append(loss_scale)
            return sum(losses)
        else:
            loss_scale = self.CrossEntropyLoss(logits, targets)
            return loss_scale

    def mulFocalLoss(self, logits, targets, gamma = 2, alpha = 0.5, phase = 'train'):
        if phase == 'train':
            losses = []
            for logit, target in zip(logits, targets):
                loss_scale = self.FocalLoss(logit, target, gamma = gamma, alpha = alpha)
                losses.append(loss_scale)
            return sum(losses)
        else:
            loss_scale = self.FocalLoss(logits, targets, gamma = gamma, alpha = alpha)
            return loss_scale

    def mulLovasz(self, logits, targets, phase = 'train'):
        if phase == 'train':
            losses = []
            for logit, target in zip(logits, targets):
                loss_scale = self.Lovasz(logit, target)
                losses.append(loss_scale)
            return sum(losses)
        else:
            loss_scale = self.Lovasz(logits, targets)
            return loss_scale

# lovasz_loss-----------------------------------------------------------------------------------------------------------
def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n

# class labels weights--------------------------------------------------------------------------------------------------
def calculate_weigths_labels(num_classes, dataloader, dataset, dir):
    # Create an instance from the data loader
    z = np.zeros((num_classes,))

    print('=> Calculating classes weights...')
    print(dataset)
    for _, sample in enumerate(dataloader):
        y = sample['label']
        y = y.detach().cpu().numpy()
        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength = num_classes)
        z += count_l

    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)

    ret = np.array(class_weights)
    np.save(os.path.join(dir, dataset + '_classes_weights.npy'), ret)

    print("=> Finished!")
    return ret

#  depth assisted loss

class depthassistedloss(nn.Module):
    def __init__(self, tao = 0.09):
        super(depthassistedloss, self).__init__()
        self.tao = tao

    def forward(self, depth, label):
        h = depth.shape[2]
        depth = depth.squeeze(1)
        label = label.float()
        d_depth = depth[:,1:,:] - depth[:,:h-1,:]
        d_label = label[:,1:,:] - label[:,:h-1,:]

        th = torch.pow(d_depth,2) - self.tao
        mask = th > 0
        d_label = torch.pow(d_label,2)
        loss = F.tanh(5 * d_label + 1) + F.tanh(-5 * d_label + 1)
        loss = loss * mask

        return torch.mean(loss)

if __name__ == "__main__":
    d = torch.randn(3, 1, 480, 640)
    l = torch.randn(3, 1, 480, 640)
    net = depthassistedloss()
    net.eval()
    loss = net(d, l)
    print(loss)