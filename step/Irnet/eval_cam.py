import sys
sys.path.append("/data3/masx/code/Volume-C-CAM")
import numpy as np
import os

from tools import score
import imageio


def run(args):
    pred_dir = os.path.join( args.round_out_dir, 'prediction/cam_filter_png')
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    cam_dir = os.path.join(args.round_out_dir, 'prediction/cam_filter')
    cam_list = os.listdir(cam_dir)
    for id in cam_list:
        cam_dict = np.load(os.path.join(cam_dir, id), allow_pickle=True).item()
        cams = cam_dict['high_res']
        cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thres)
        keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
        cls_labels = np.argmax(cams, axis=0)
        cls_labels = keys[cls_labels]
        imageio.imsave(os.path.join(args.round_out_dir, pred_dir, id[0:-4].replace('Image', 'Label') + '.png'),
                       (cls_labels*255).astype(np.uint8))
    gt_dir = '/data3/masx/data/prostate_MR/DL_Label'
    score.score(pred_dir, gt_dir)


if __name__ == '__main__':
    round_out_dir = '/data3/masx/code/Volume-C-CAM/myExperiment/results/EM_rounds_acdc/layercam'

    pred_dir = os.path.join(round_out_dir, 'prediction/lpcam_png')

    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    
    cam_dir = os.path.join(round_out_dir, 'prediction/lpcam')

    cam_list = os.listdir(cam_dir)
    yuzhi = 0.05
    # yuzhi = 0.9
    best_dice = 0.
    while yuzhi < 0.99:
        print('yuzhi: ', yuzhi)
        for id in cam_list:
            # img_ = np.asarray(imageio.imread(os.path.join(img_dir, id[0:-4] + '.png')))
            # img = np.zeros((img_.shape[0], img_.shape[1], 3))
            # img[:, :, 0] = img_
            # img[:, :, 1] = img_
            # img[:, :, 2] = img_
            # img = np.uint8(img)
            cam_dict = np.load(os.path.join(cam_dir, id), allow_pickle=True).item()
            # cam_dict = np.load(os.path.join(cam_dir, id), allow_pickle=True)
            cams = cam_dict['high_res']
            # cams = cam_dict['filt_cam']
            # cams = cam_dict[0]
            # cams = cam_dict
            # print(cams.shape)
            # print(id)
            cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=yuzhi)
            # cams = np.pad([cams], ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=yuzhi)
            # keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
            # keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
            cls_labels = np.argmax(cams, axis=0)
            # cls_labels = imutils.crf_inference_label(img, cls_labels, t=5, n_labels=2)
            # for i in range(len(keys)):
            #     keys[i] = class_value[keys[i]]

            # cls_labels = keys[cls_labels]
            imageio.imsave(
                os.path.join(pred_dir, id[0:-4] + '.png'),
                (cls_labels * 255).astype(np.uint8))
        # gt_dir = '/data3/masx/data/ProMRI/DL_Label'
        gt_dir = '/data3/masx/data/CZ_ACDC/DL_Label'
        dice = score.score(pred_dir, gt_dir)
        if dice >= best_dice:
            best_dice = dice
        else:
            break
        yuzhi = yuzhi + 0.05
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
