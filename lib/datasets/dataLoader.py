#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2019/10/29 20:24 
# @Site :  
# @File : datasets.py 
# @Software: PyCharm

from torch.utils.data import DataLoader
from datasets.NYUDataset import NYUDataset
from datasets.SUNDataset import SUNDataset
from datasets.TROSDataset import TROSDataset

def trainval_dataloader(cfg):
    if cfg.DATA_NAME == "SUNRGBD":
        train_set = SUNDataset(cfg, type = "train")
        val_set = SUNDataset(cfg, type  = "val")
    elif cfg.DATA_NAME == "NYUV2":
        train_set = NYUDataset(cfg, type = "train")
        val_set = NYUDataset(cfg, type  = "val")
    elif cfg.DATA_NAME == "TROS":
        train_set = TROSDataset(cfg, type = "train")
        val_set = TROSDataset(cfg, type  = "val")
    else:
        raise ValueError('dataLoader.py: dataset %s is not support yet' % cfg.DATA_NAME)

    train_loader = DataLoader(train_set, batch_size = cfg.TRAIN_BATCHES, shuffle = True, num_workers = cfg.DATA_WORKERS)
    val_loader = DataLoader(val_set, batch_size = cfg.TRAIN_BATCHES, shuffle = False, num_workers = cfg.DATA_WORKERS)

    return train_loader, val_loader

def test_dataloader(cfg):
    if cfg.DATA_NAME == "SUNRGBD":
        test_set = SUNDataset(cfg, type="test")
    elif cfg.DATA_NAME == "NYUV2":
        test_set = NYUDataset(cfg, type="test")
    elif cfg.DATA_NAME == "TROS":
        test_set = TROSDataset(cfg, type="test")
    else:
        raise ValueError('dataLoader.py: dataset %s is not support yet' % cfg.DATA_NAME)

    test_loader = DataLoader(test_set, batch_size = cfg.TEST_BATCHES, shuffle = False, num_workers = cfg.DATA_WORKERS)
    return test_loader