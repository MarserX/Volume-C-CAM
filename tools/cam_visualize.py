import os
import numpy as np
import imageio
import cv2


if __name__ == "__main__":
    root_dir = '/data3/masx/code/Volume-C-CAM/Volume-C-CAM-for-ACDC'
    # root_dir = '/home/tzq-cz/code/causal_graph_semi_seg_auto'
    # input_cam_dir = os.path.join(root_dir, 'myExperiment/results/EM_rounds_prostate/round_0/prediction/cam_ll')
    input_cam_dir = os.path.join(root_dir, 'myExperiment/results/EM_rounds_acdc/round_1/prediction/cam')
    save_cam_dir = os.path.join(root_dir, 'myExperiment/results/EM_rounds_acdc/round_1/prediction/cam_vis')
    # input_cam_dir = os.path.join(root_dir, 'myExperiment/results/EM_rounds_LiTS17_2/round_0/prediction/cam')
    # save_cam_dir = os.path.join(root_dir, 'myExperiment/results/EM_rounds_LiTS17_2/round_0/prediction/cam_vis')
    if not os.path.exists(save_cam_dir):
        os.makedirs(save_cam_dir)

    cam_list = os.listdir(input_cam_dir)
    for cam_name in cam_list:
        cam_name = cam_name[0:-4]
        print(cam_name)
        img = np.asarray(
            imageio.imread(os.path.join('/data3/masx/data/CZ_ACDC/DL_Image', cam_name + '.png')))
        # img = np.asarray(
        #     imageio.imread(os.path.join('/data5/chenz/dataset/LiTS17/Images', cam_name + '.png')))
        if len(img.shape) < 3:
            img_rgb = np.zeros((img.shape[0], img.shape[1], 3))
            img_rgb[:,:,0] = img
            img_rgb[:,:,1] = img
            img_rgb[:,:,2] = img
            img = img_rgb
        cam_dict = np.load(os.path.join(input_cam_dir, cam_name + '.npy'), allow_pickle=True).item()
        # cam_dict = np.load(os.path.join(input_cam_dir, cam_name + '.npy'), allow_pickle=True)
        cams = cam_dict['high_res']
        # cams = cam_dict
        # keys = list(cam_dict['keys'].numpy())
        heatmap = cv2.applyColorMap(np.uint8(cams[0] * 255), cv2.COLORMAP_JET)
        result = heatmap * 0.5 + img * 0.5
        cv2.imwrite(os.path.join(save_cam_dir, cam_name + '_overlay.jpg'), np.uint8(result))
        # for i in range(len(keys)):
        #     heatmap = cv2.applyColorMap(np.uint8(cams[i] * 255), cv2.COLORMAP_JET)
        #     result = heatmap * 0.5 + img * 0.5
        #     cv2.imwrite(os.path.join(save_cam_dir, cam_name + '_' + str(keys[i] + 1) + '_overlay.jpg'), result)

