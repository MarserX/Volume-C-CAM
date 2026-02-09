import numpy as np
import os

import imageio
from tools import score


def run(args):
    pred_dir = os.path.join(args.round_out_dir, 'prediction/', args.sem_seg_out_dir)
    gt_dir = '/data3/masx/data/prostate_MR/DL_Label'
    cam_dir = os.path.join(args.round_out_dir, 'prediction/rw')
    cam_list = os.listdir(cam_dir)
    yuzhi = 0.05
    while yuzhi < 1:
        print('yuzhi: ', yuzhi)
        for id in cam_list:
            cam_dict = np.load(os.path.join(cam_dir, id), allow_pickle=True)
            cams = cam_dict[0]
            cams = np.pad([cams], ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=yuzhi)
            cls_labels = np.argmax(cams, axis=0)
            imageio.imsave(
                os.path.join(pred_dir, id[0:-4] + '.png'),
                (cls_labels * 255).astype(np.uint8))
        score.score(pred_dir, gt_dir)
        yuzhi = yuzhi + 0.05
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    # score.score(pred_dir, gt_dir)


if __name__ == '__main__':
    from options.experiment_options import MyOptions

    args = MyOptions().parse()
    args.round_out_dir = '/data3/masx/code/Volume-C-CAM/myExperiment/results/EM_rounds_prostate/' + 'round_' + str(
        0) + '/'
    run(args)