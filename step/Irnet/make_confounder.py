import numpy as np
import os
import cv2
import torch


def make_one_hot(input, num_classes):
    """input: [N, 1, *]
       output: [N, C, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, torch.LongTensor(input), 1)
    return result


def run(args):
    root = '/home/tzq-cz/code/causal_graph_semi_seg_auto'
    seg_pred_dir = os.path.join('results/prediction_T%s' % args.roundNum)
    h, w = args.crop_size, args.crop_size
    confounder = torch.zeros((2, h, w))
    one_hot_sum = torch.ones((2, h, w))
    # seg_pred_dir = os.path.join(root, 'segmentation/data/prediction/voc12/val')
    # seg_pred_dir = os.path.join(seg_pred_dir, 'prediction_trainaug_weak_centercrop')
    img_list = os.listdir(seg_pred_dir)
    count = 0
    for img_name in img_list:
        img_path = os.path.join(seg_pred_dir, img_name)
        seg_pred = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        seg_pred_resize = cv2.resize(seg_pred, (h, w), interpolation=cv2.INTER_NEAREST)
        seg_pred_resize_ = seg_pred_resize[np.newaxis, np.newaxis, :]
        seg_pred_one_hot = make_one_hot(seg_pred_resize_, 2)
        one_hot_sum += seg_pred_one_hot.squeeze()
        count += 1
        print(count)
    one_hot_avg = one_hot_sum / count
    np.save(os.path.join(args.round_out_dir, 'confounder.npy'), np.array(one_hot_avg))
    print(one_hot_sum.shape)


if __name__ == "__main__":
    root = '/home/tzq-cz/code/code_rep/CONTA/'
    seg_pred_dir = '/home/tzq-cz/code/code_rep/DeepLabV3Plus-Pytorch/results/t0'
    h, w = 513, 513
    confounder = torch.zeros((21, h, w))
    one_hot_sum = torch.ones((21, h, w))
    # seg_pred_dir = os.path.join(root, 'segmentation/data/prediction/voc12/val')
    seg_pred_dir = os.path.join(seg_pred_dir, 'prediction_trainaug_weak_centercrop')
    img_list = os.listdir(seg_pred_dir)
    count = 0
    for img_name in img_list:
        img_path = os.path.join(seg_pred_dir, img_name)
        seg_pred = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        seg_pred_resize = cv2.resize(seg_pred, (h, w), interpolation=cv2.INTER_NEAREST)
        seg_pred_resize_ = seg_pred_resize[np.newaxis, np.newaxis, :]
        seg_pred_one_hot = make_one_hot(seg_pred_resize_, 21)
        one_hot_sum += seg_pred_one_hot.squeeze()
        count += 1
        print(count)
    one_hot_avg = one_hot_sum / count
    np.save(os.path.join(root, 'pseudo_mask/voc12/confounder_t0.npy'), np.array(one_hot_avg))
    print(one_hot_sum.shape)
