#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2019/12/3 22:32 
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt

label_colours = np.array([
                 (148, 65, 137), (255, 116, 69), (86, 156, 137),
                 (202, 179, 158), (155, 99, 235), (161, 107, 108),
                 (133, 160, 103), (76, 152, 126), (84, 62, 35),
                 (44, 80, 130), (31, 184, 157), (101, 144, 77),
                 (23, 197, 62), (141, 168, 145), (142, 151, 136),
                 (115, 201, 77), (100, 216, 255), (57, 156, 36),
                 (88, 108, 129), (105, 129, 112), (42, 137, 126),
                 (155, 108, 249), (166, 148, 143), (81, 91, 87),
                 (100, 124, 51), (73, 131, 121), (157, 210, 220),
                 (134, 181, 60), (221, 223, 147), (123, 108, 131),
                 (161, 66, 179), (163, 221, 160), (31, 146, 98),
                 (99, 121, 30), (49, 89, 240), (116, 108, 9),
                 (161, 176, 169), (80, 29, 135), (177, 105, 197),
                 (139, 110, 246)])

def colored_segmap(label, n_classes):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    r = label.copy()
    g = label.copy()
    b = label.copy()

    for ll in range(0, n_classes):
        r[label == ll] = label_colours[ll, 0]
        g[label == ll] = label_colours[ll, 1]
        b[label == ll] = label_colours[ll, 2]

    # 255 is the backgorund
    r[label == 255] = 0
    g[label == 255] = 0
    b[label == 255] = 0

    rgb = np.zeros((label.shape[0], label.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b

    return rgb


def colored_segmap_tros(label, n_classes):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    r = label.copy()
    g = label.copy()
    b = label.copy()

    label_colours_tros = np.array([(0,0,0),(255,0,0),(0,255,0)])

    for ll in range(0, n_classes):
        r[label == ll] = label_colours_tros[ll, 0]
        g[label == ll] = label_colours_tros[ll, 1]
        b[label == ll] = label_colours_tros[ll, 2]

    # 255 is the backgorund
    r[label == 255] = 0
    g[label == 255] = 0
    b[label == 255] = 0

    rgb = np.zeros((label.shape[0], label.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b

    return rgb

def decode_seg_map_sequence(label_masks, num_classes):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = colored_segmap(label_mask, num_classes)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks

cmap_d = plt.cm.jet
def colored_depthmap(depth, d_min = None, d_max = None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    # H, W, C
    return np.uint8(255 * cmap_d(depth_relative)[:, :, :3])

cmap_f = plt.cm.gray
def colored_heatmap(map):
    map_min = np.min(map)
    map_max = np.max(map)
    map_relative = (map - map_min) / (map_max - map_min)
    # H, W, C
    return np.uint8(255 * cmap_f(map_relative)[:, :, :3])

# merge rgb, depth, label
def merge_into_row(rgb, depth, label, dataset):
    mean = torch.FloatTensor([0.485, 0.456, 0.406])
    mean = mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    std = torch.FloatTensor([0.229, 0.224, 0.225])
    std = std.unsqueeze(0).unsqueeze(2).unsqueeze(3)

    rgb_ = rgb.data.cpu()
    rgb_ = ((rgb_ * std) + mean) * 255.0
    rgb = rgb.permute(1, 2, 0).numpy()

    depth_ = depth.cpu().data
    depth_ = depth_.squeeze(0).numpy()

    label_ = label.cpu().data
    label_ = label_.numpy()

    depth_ = colored_depthmap(depth_)
    label_ = colored_segmap(label_, dataset)

    return np.uint8(np.hstack([rgb_, depth_, label_]))
