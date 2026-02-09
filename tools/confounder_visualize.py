import os
import numpy as np
import imageio
import cv2
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


if __name__ == "__main__":
    root_dir = '/data3/masx/code/Volume-C-CAM/Volume-C-CAM-for-ACDC'
    input_cam_dir = os.path.join(root_dir, 'myExperiment/results/EM_rounds_acdc/round_0/')
    # save_cam_dir = os.path.join(root_dir, 'myExperiment/results/EM_rounds_prostate/round_1/prediction/cam_overlay')
    # if not os.path.exists(save_cam_dir):
    #     os.makedirs(save_cam_dir)

    confounder_dict = np.load(os.path.join(input_cam_dir, 'confounder_full' + '.npy'), allow_pickle=True)
    # confounder = confounder_dict[1,:,:]
    # plt.imshow(confounder)
    # plt.show()
    
    cv2.imwrite(os.path.join(input_cam_dir, 'confounder_full.png'), confounder_dict[1,:,:]*255)
    print(np.shape(confounder_dict))
    