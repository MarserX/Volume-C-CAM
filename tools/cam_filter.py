import os
import numpy as np
import torch
import torch.nn.functional as F

def create_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

if __name__ == "__main__":
    root_dir = '/data3/masx/code/Volume-C-CAM/Volume-C-CAM-for-ACDC'
    input_cam_dir = os.path.join(root_dir, 'myExperiment/results/EM_rounds_acdc/round_1/prediction/rw')
    save_cam_dir = create_directory(os.path.join(root_dir, 'myExperiment/results/EM_rounds_acdc/round_1/prediction/rw_filter'))
    confounder_dir = os.path.join(root_dir, 'myExperiment/results/EM_rounds_acdc/round_0')
    confounder_dict = np.load(os.path.join(confounder_dir, 'confounder_full' + '.npy'), allow_pickle=True)
    filter = confounder_dict[1, :, :]
    min_res = filter.min()
    filter[filter > min_res] = 1
    filter[filter <= min_res] = 0
    if not os.path.exists(save_cam_dir):
        os.makedirs(save_cam_dir)

    cam_list = os.listdir(input_cam_dir)
    for cam_name in cam_list:
        cam_name = cam_name[0:-4]
        print(cam_name)
        # cam_dict = np.load(os.path.join(input_cam_dir, cam_name + '.npy'), allow_pickle=True).item()
        cam_dict = np.load(os.path.join(input_cam_dir, cam_name + '.npy'), allow_pickle=True)
        # cams = cam_dict['high_res']
        cams = cam_dict
        # keys = list(cam_dict['keys'].numpy())
        # keys = list(cam_dict['keys'])
        filter = filter.squeeze()
        cam_filter = np.zeros_like(cams)
        # for i in range(len(keys)):
        for i in range(1):
            filter = F.interpolate(torch.tensor(filter[np.newaxis, np.newaxis, :, :]),
                                   size=[cams[i].shape[0], cams[i].shape[1]],
                                   mode='bilinear')
            cam_filter[i] = cams[i] * filter.squeeze().numpy()
        # np.save(os.path.join(save_cam_dir, cam_name + '.npy'),
        #         {"keys": cam_dict['keys'], "filt_cam": cam_filter})
        np.save(os.path.join(save_cam_dir, cam_name + '.npy'),
                {"filt_cam": cam_filter})
