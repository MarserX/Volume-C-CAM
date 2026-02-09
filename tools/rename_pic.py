#!/usr/bin/env python3
# encoding: utf-8
"""
@Project: causal_graph_semi_seg_auto
@author: Zane-Chen
@contact: 1900938761@qq.com
@file: rename_pic.py
@time: 7/22/21 11:24 AM
@IDE: PyCharm
@desc: define python functions to be called by cpp
"""
import cv2
import os
from PIL import Image
import numpy as np

root_dir = '/home/tzq-cz/data/prostate_MR'
img_list = os.listdir(os.path.join(root_dir, 'Label'))

for img_name1 in img_list:
    img_name2 = img_name1.replace('Merge', 'Label')
    img = cv2.imread(os.path.join(root_dir, 'Label', img_name1), cv2.IMREAD_GRAYSCALE)
    # img2 = cv2.imread(os.path.join(root_dir, 'DL_Label_valid', img_name2), cv2.IMREAD_GRAYSCALE)
    img_save = img[:, :, 0]
    cv2.imwrite(os.path.join(root_dir, 'DL_Label_valid', img_name2), img_save)
