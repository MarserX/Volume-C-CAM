import sys
sys.path.append("/data3/masx/code/Volume-C-CAM")
import numpy as np
import os
from misc import imutils
import imageio


if __name__ == '__main__':
    class_value = np.array([0, 255], dtype=np.uint8)
    img_dir = '/data3/masx/data/CZ_ACDC/DL_Image'
    round_out_dir = '/data3/masx/code/Volume-C-CAM/Volume-C-CAM-for-ACDC/myExperiment/results/EM_rounds_acdc/' + 'round_1'
    # pred_dir = os.path.join(round_out_dir, 'prediction/cam_png')
    # os.makedirs(pred_dir)
    cam_dir = os.path.join(round_out_dir, 'prediction/lpcam')
    cam_list = os.listdir(cam_dir)
    for id in cam_list:
        cam_dict = np.load(os.path.join(cam_dir, id), allow_pickle=True).item()
        img_ = np.asarray(imageio.imread(os.path.join(img_dir, id[0:-4] + '.png')))
        img = np.zeros((img_.shape[0], img_.shape[1], 3))
        img[:, :, 0] = img_
        img[:, :, 1] = img_
        img[:, :, 2] = img_
        img = np.uint8(img)
        cams = cam_dict['high_res']
        # cams = cam_dict['filt_cam']
        cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0.75)
        # keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
        keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
        cls_labels = np.argmax(cams, axis=0)
        # for i in range(len(keys)):
        #     keys[i] = class_value[keys[i]]

        cls_labels = keys[cls_labels]
        cams_crf = imutils.crf_inference_label(img, cls_labels, t=7, n_labels=keys.shape[0])
        for i in range(len(keys)):
            keys[i] = class_value[keys[i]]
        cams_crf = keys[cams_crf]
        imageio.imsave(os.path.join(round_out_dir, 'prediction', 'lpcam_crf',
                                    id[0:-4] + '.png'), (cams_crf).astype(np.uint8))
        print(id)
