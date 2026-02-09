#!/usr/bin/env python3
# encoding: utf-8
"""
@author: Zane-Chen
@contact: 1900938761@qq.com
@file: score.py
@time: 2021/4/1 15:52
@desc:
"""

import numpy as np
from PIL import Image
import os
import glob
import cv2


def score(prediction_dir, ground_truth_dir, class_num=2):
    hist = compute_hist(prediction_dir, ground_truth_dir, class_num)
    # overall accuracy
    # acc = np.diag(hist).sum() / hist.sum()
    # print('>>>', 'overall accuracy', acc)
    # # per-class accuracy
    # acc = np.diag(hist) / hist.sum(1)
    # print('>>>', 'mean accuracy', np.nanmean(acc))
    # # per-class IU
    # iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    # print('>>>', 'per class IOU', iu)
    # mean_iou = np.nanmean(iu)
    # print('>>>', 'mean IU', mean_iou)
    # freq = hist.sum(1) / hist.sum()
    # fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    # print('>>>', 'fwavacc', fwavacc)
    dice = 2 * np.diag(hist) / (hist.sum(1) + hist.sum(0))
    print('>>>', 'per-class dice', dice)
    return dice[1]


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def compute_hist(prediction_dir, ground_truth_dir, class_num=21):
    n_cl = class_num
    hist = np.zeros((n_cl, n_cl))
    for pred_img in glob.glob(os.path.join(prediction_dir, '*.png')):
        pred = Image.open(pred_img)
        pred = np.array(pred)
        # pred = pred[:,:,0]
        pred[pred > 0] = 1
        label_name = pred_img.split('/')[-1].replace('Image', 'Label')
        #label_name = pred_img[-17:]
        #label_name = pred_img[-15:]
        # print(label_name)
        label_path = os.path.join(ground_truth_dir, label_name)
        #label_path = os.path.join(ground_truth_dir, label_name)
        gt = Image.open(label_path)
        gt = np.array(gt)
        gt[gt > 0] = 1
        # gt[gt < 2] = 0
        # gt[gt == 2] = 1
        # gt[gt > 1] = 0
        hh, ww = np.shape(gt)
        pred = cv2.resize(pred, (hh, ww), interpolation=cv2.INTER_NEAREST)
        hist += fast_hist(gt.flatten(),pred.flatten(),n_cl)
    return hist


if __name__ == '__main__':
    # round_out_dir = '/home/tzq-cz/code/causal_graph_semi_seg_auto/myExperiment/results/EM_rounds_prostate/' + 'round_' + str(
    #     0) + '/'
    # pred_dir = os.path.join(round_out_dir, 'prediction/cam_png')
    pred_dir = '/home/chenz/data5/code/Volume-C-CAM/Volume-C-CAM-for-ACDC/myExperiment/results/EM_rounds_acdc/round_1/prediction/lpcam_filter_png_post'
    # pred_dir = '/home/tzq-cz/data/VOCdevkit/VOC2012/SegmentationClassAug_weak_t0'
    # pred_dir = '/home/tzq-cz/code/causal_graph_semi_seg_auto/myExperiment/results/layercam'
    # gt_dir = '/home/tzq-cz/data/VOCdevkit/VOC2012/SegmentationClassAug'
    gt_dir = '/data2/CZ/data/CZ_ACDC/DL_Label'
    score(pred_dir, gt_dir)