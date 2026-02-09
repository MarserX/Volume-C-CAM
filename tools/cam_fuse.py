import sys
sys.path.append("/data3/masx/code/Volume-C-CAM")
import os
import numpy as np


def create_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

# load cams for .npy
root_dir = 'myExperiment/results/EM_rounds_prostate/round_0/prediction'
cam_dir = os.path.join(root_dir, 'cam')
lpcam_dir = os.path.join(root_dir, 'lpcam')
fuse_cam_dir = create_directory(os.path.join(root_dir, 'fusecam'))
cam_list = os.listdir(cam_dir)

for cam_id in cam_list:
    print(cam_id)
    cam_dict = np.load(os.path.join(cam_dir, cam_id), allow_pickle=True).item()
    lpcam_dict = np.load(os.path.join(lpcam_dir, cam_id), allow_pickle=True).item()
    cam = cam_dict['high_res']
    lpcam = lpcam_dict['high_res'].numpy()

    ## fuse cam and lpcam
    # multiplication
    fuse_cam_multi = np.sqrt(cam * lpcam)

    # # average
    # fuse_cam_avg = (cam + lpcam) / 2

    np.save(os.path.join(fuse_cam_dir, cam_id.replace('jpg','npy')),{"high_res": fuse_cam_multi})


