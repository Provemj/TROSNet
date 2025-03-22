#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2019/12/3 22:35 
# @Software: PyCharm

import torch.nn as nn

class LR_Scheduler:
    def __init__(self, base_lr, max_iteration, decay_weights):
        self.lr = base_lr
        self.weights = decay_weights
        self.max_iteration = max_iteration

    def __call__(self, optimizer, i):
        lr = self.lr * pow((1 - i / self.max_iteration), 0.9)
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            optimizer.param_groups[0]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10

def get_params(model, key):
    for m in model.named_modules():
        if key == '1x':
            if 'backbone' in m[0] and isinstance(m[1], nn.Conv2d):
                for p in m[1].parameters():
                    yield p
        elif key == '10x':
            if 'backbone' not in m[0] and isinstance(m[1], nn.Conv2d):
                for p in m[1].parameters():
                    yield p