import sys
sys.path.append("/data3/masx/code/Volume-C-CAM")
import os
import numpy as np
import torch


def create_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

# load cams for .npy
root_dir = 'myExperiment/results/EM_rounds_prostate/round_0/prediction'
# cam_dir = os.path.join(root_dir, 'cam')
lpcam_dir = os.path.join(root_dir, 'lpcam_best')
lpcam_save_dir =create_directory(os.path.join(root_dir, 'lpcam_post'))
# fuse_cam_dir = create_directory(os.path.join(root_dir, 'fusecam'))
cam_list = os.listdir(lpcam_dir)

list1 = [i for i in range(1, 107)]
list2 = [i for i in range(108, 116)]
list3 = [i for i in range(117, 131)]
list4 = [i for i in range(135, 136)]
list5 = [i for i in range(137, 141)]
volume_list = list1+list2+list3+list4+list5
# volume_list = [3]

for volume_id in volume_list:
    dict_pool = []
    print("volume_" + str(volume_id).zfill(4))
    infer_list = "data/train_valid_prostate_v2_" + str(volume_id).zfill(4) + ".txt"
    img_gt_name_list = open(infer_list).read().splitlines()
    img_name_list = [gt_name.split(' ')[0] for gt_name in img_gt_name_list]
    lpcam_0_dict = np.load(os.path.join(lpcam_dir, img_name_list[0] + '.npy'), allow_pickle=True).item() 
    lpcam_0 = lpcam_0_dict['high_res'].numpy()
    lpcam_sum = np.zeros_like(lpcam_0)

    # print("computing mean cam ...")
    for img_name in img_name_list:
        dict_slice = {}
        # print(img_name)
        lpcam_dict = np.load(os.path.join(lpcam_dir, img_name + '.npy'), allow_pickle=True).item()
        lpcam = lpcam_dict['high_res'].numpy()
        lpcam_sum = lpcam_sum + lpcam
        dict_slice["mat"] = lpcam
        dict_slice["tag"] = 1
        dict_slice["name"] = img_name
        dict_pool.append(dict_slice)
    lpcam_mean = lpcam_sum / len(img_name_list)

    # print("computing diff of each slice from mean cam ...")
    for i in range(len(dict_pool)):
        # print(img_name)
        lpcam = dict_pool[i]["mat"]

        diff = abs(lpcam - lpcam_mean)
        mean_diff = diff.mean()
        # print(f'{img_name}: {mean_diff}')
        if mean_diff > 0.03:
            dict_pool[i]["tag"] = 0
            print(f'{img_name}: {mean_diff}')

    for i in range(round(len(dict_pool)/2), -1, -1):
        print(i)
        if dict_pool[i]["tag"]==0:
            print(dict_pool[i]["name"])
            dict_pool[i]["mat"] = dict_pool[i+1]["mat"]
            dict_pool[i]["tag"] = 1
            lpcam_dict = np.load(os.path.join(lpcam_dir, dict_pool[i]["name"] + '.npy'), allow_pickle=True).item()
            lpcam_dict['high_res'] = torch.tensor(dict_pool[i]["mat"])
            np.save(os.path.join(lpcam_save_dir, dict_pool[i]["name"].replace('jpg', 'npy')), lpcam_dict)

    for i in range(round(len(dict_pool)/2)+1, len(dict_pool)):
        print(i)
        if dict_pool[i]["tag"]==0:
            print(dict_pool[i]["name"])
            dict_pool[i]["mat"] = dict_pool[i-1]["mat"]
            dict_pool[i]["tag"] = 1
            lpcam_dict = np.load(os.path.join(lpcam_dir, dict_pool[i]["name"] + '.npy'), allow_pickle=True).item()
            lpcam_dict['high_res'] = torch.tensor(dict_pool[i]["mat"])
            np.save(os.path.join(lpcam_save_dir, dict_pool[i]["name"].replace('jpg', 'npy')), lpcam_dict)

    
        
        

