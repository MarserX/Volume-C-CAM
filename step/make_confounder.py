import numpy as np
import os
import cv2
import torch


def make_one_hot(input, num_classes):
    """input: [N, 1, *]
       output: [N, C, *]
    """
    input[input == 255] = 0
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, torch.LongTensor(input), 1)
    return result


def run(args):
    root = '/data3/masx/code/Volume-C-CAM'
    # seg_pred_dir = os.path.join(root, 'myExperiment', args.round_in_dir, 'prediction/sem_seg_label')
    # seg_pred_dir = os.path.join(root, 'myExperiment', args.round_out_dir, 'prediction/sem_seg_label')
    seg_pred_dir = os.path.join(root,  args.round_out_dir, 'prediction/cam_png')
    h, w = 512, 512
    # confounder = torch.zeros((1, h, w))
    one_hot_sum = torch.ones((2, h, w))
    # seg_pred_dir = os.path.join(root, 'segmentation/data/prediction/voc12/val')
    # seg_pred_dir = os.path.join(seg_pred_dir, 'prediction_trainaug_weak_centercrop')
    img_list = os.listdir(seg_pred_dir)
    count = 0
    for img_name in img_list:
        img_path = os.path.join(seg_pred_dir, img_name)
        seg_pred = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        seg_pred_resize = cv2.resize(seg_pred, (h, w), interpolation=cv2.INTER_NEAREST)
        seg_pred_resize[seg_pred_resize > 0] = 1
        seg_pred_resize_ = seg_pred_resize[np.newaxis, np.newaxis, :]
        seg_pred_one_hot = make_one_hot(seg_pred_resize_, 2)
        one_hot_sum += seg_pred_one_hot.squeeze()
        count += 1
        # print(count)
    one_hot_avg = one_hot_sum / count
    # np.save(os.path.join(args.round_out_dir, 'confounder.npy'), np.array(one_hot_avg))
    np.save(os.path.join('/data3/masx/code/Volume-C-CAM', args.round_out_dir, 'confounder.npy'),
            np.array(one_hot_avg))
    print(one_hot_sum.shape)


if __name__ == "__main__":
    root = '/data3/masx/code/Volume-C-CAM/myExperiment/results/EM_rounds_acdc/round_0'
    # seg_pred_dir = '/home/tzq-cz/code/code_rep/DeepLabV3Plus-Pytorch/results/t0'
    data_dir = '/data3/masx/data/CZ_ACDC/'
    idx_file = os.path.join('/data3/masx/code/Volume-C-CAM/data', 'train_all_valid_acdc.txt')
    seg_pred_dir = os.path.join(data_dir, 'DL_Label')
    h, w = 512, 512
    # confounder = torch.zeros((1, h, w))
    one_hot_sum = torch.ones((2, h, w))
    # seg_pred_dir = os.path.join(root, 'segmentation/data/prediction/voc12/val')
    # seg_pred_dir = os.path.join(seg_pred_dir, 'prediction_trainaug_weak_centercrop')
    # img_list = os.listdir(seg_pred_dir)
    img_list = []
    with open(idx_file, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if lines:
                img_list.append(lines)
            else:
                break
    count = 0
    for img_name in img_list:
        img_path = os.path.join(seg_pred_dir, img_name.split(' ')[0].strip()+'.png')
        seg_pred = cv2.imread(img_path.replace('Image', 'Label'), cv2.IMREAD_GRAYSCALE)
        seg_pred_resize = cv2.resize(seg_pred, (h, w), interpolation=cv2.INTER_NEAREST)
        seg_pred_resize[seg_pred_resize > 0] = 1
        seg_pred_resize_ = seg_pred_resize[np.newaxis, np.newaxis, :]
        seg_pred_one_hot = make_one_hot(seg_pred_resize_, 2)
        one_hot_sum += seg_pred_one_hot.squeeze()
        # one_hot_sum += torch.from_numpy(seg_pred_resize_.squeeze(0)).to(dtype=torch.float32)
        count += 1
        print(count)
    one_hot_avg = one_hot_sum / count
    np.save(os.path.join(root, 'confounder_full.npy'), np.array(one_hot_avg))
    print(one_hot_sum.shape)
