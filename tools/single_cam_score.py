#!/usr/bin/env python3
# encoding: utf-8
"""
@Project: causal_graph_semi_seg_auto
@author: Zane-Chen
@contact: 1900938761@qq.com
@file: single_cam_score.py
@time: 8/6/21 5:00 PM
@IDE: PyCharm
@desc: define python functions to be called by cpp
"""
import numpy as np
import os
import imageio
import cv2
from PIL import Image
import torch
import torch.nn.functional as F


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


if __name__ == '__main__':
    n_cl = 2
    round_out_dir = '/home/tzq-cz/code/causal_graph_semi_seg_auto/myExperiment/results/EM_rounds_prostate/' + 'round_' + str(
        1) + '/'
    cam_dir = os.path.join(round_out_dir, 'prediction/cam_ll')
    rw_dir = os.path.join(round_out_dir, 'prediction/rw')
    gt_dir = '/home/tzq-cz/data/prostate_MR/DL_Label_valid'
    id = 'DL_Image0001_0016'
    cam_eval_thres = 0.5
    rw_eval_thres = 0.75
    cam_dict = np.load(os.path.join(cam_dir, id + '.npy'), allow_pickle=True).item()
    rw = np.load(os.path.join(rw_dir, id + '.npy'), allow_pickle=True)
    cams = cam_dict['high_res']
    cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=cam_eval_thres)
    rw = F.interpolate(torch.from_numpy(rw).unsqueeze(0), (cams.shape[1], cams.shape[2]), mode='bilinear', align_corners=False)
    rw = rw/torch.max(rw)
    rw = np.pad(rw.squeeze(0), ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=rw_eval_thres)
    keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
    # cls_labels = np.argmax(cams, axis=0)
    cls_labels = np.argmax(rw, axis=0)
    cls_labels = keys[cls_labels]
    gt = Image.open(os.path.join(gt_dir, id.replace('Image', 'Label') + '.png'))
    gt = np.array(gt)
    gt[gt > 0] = 1
    hh, ww = np.shape(gt)
    pred = cv2.resize(cls_labels, (hh, ww), interpolation=cv2.INTER_NEAREST)
    hist = np.zeros((n_cl, n_cl))
    hist += fast_hist(gt.flatten(), cls_labels.flatten(), n_cl)
    acc = np.diag(hist).sum() / hist.sum()
    print('>>>', 'overall accuracy', acc)
    # per-class accuracy
    acc = np.diag(hist) / hist.sum(1)
    print('>>>', 'mean accuracy', np.nanmean(acc))
    # per-class IU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print('>>>', 'per class IOU', iu)
    mean_iou = np.nanmean(iu)
    print('>>>', 'mean IU', mean_iou)
    freq = hist.sum(1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    print('>>>', 'fwavacc', fwavacc)
    dice = 2 * np.diag(hist) / (hist.sum(1) + hist.sum(0))
    print('>>>', 'per-class dice', dice)


