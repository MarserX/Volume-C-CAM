import sys
sys.path.append("/data3/masx/code/Volume-C-CAM")
import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os
import imageio


import data.dataloader_prostate
from misc import torchutils, indexing
import cv2

cudnn.enabled = True

def _work(process_id, model, dataset, args):

    n_gpus = torch.cuda.device_count()
    databin = dataset[process_id]
    data_loader = DataLoader(databin,
                             shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()

        for iter, pack in enumerate(data_loader):
            # img_name = voc12.dataloader.decode_int_filename(pack['name'][0])
            img_name = pack['name'][0]
            orig_img_size = np.asarray(pack['size'])

            edge, dp = model(pack['img'][0].cuda(non_blocking=True))

            cam_dict_fuzhu = np.load(
                os.path.join(args.round_out_dir, 'prediction/cam', img_name + '.npy'),
                allow_pickle=True).item()
            cam_dict = np.load(os.path.join(args.round_out_dir, 'prediction', args.cam_out_dir, img_name + '.npy'),
                               allow_pickle=True).item()

            cams_fuzhu = cam_dict_fuzhu['cam_ll']
            cams = cam_dict['cam_ll']
            # cams = cam_dict['cam']
            # cams = torch.tensor(cam_dict['filt_cam'])
            cams = F.interpolate(torch.unsqueeze(cams, 0), (cams_fuzhu.shape[1], cams_fuzhu.shape[2]), mode='bilinear',
                                 align_corners=False)
            # keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')

            cam_downsized_values = cams.cuda()

            rw = indexing.propagate_to_edge(cam_downsized_values, edge, beta=args.beta, exp_times=args.exp_times, radius=5)

            rw_up = F.interpolate(rw, scale_factor=4, mode='bilinear', align_corners=False)[..., 0, :orig_img_size[0], :orig_img_size[1]]
            rw_up = rw_up / torch.max(rw_up)

            np.save(os.path.join(args.round_out_dir, 'prediction', args.save_rw_dir, img_name + '.npy'), rw_up.cpu().numpy())
            # img = np.asarray(
            #     imageio.imread(os.path.join('/home/tzq-cz/data/prostate_MR/DL_Image', img_name + '.png')))
            # heatmap = cv2.applyColorMap(np.uint8(rw_up[0].cpu().numpy() * 255), cv2.COLORMAP_JET)
            # result = heatmap * 0.5 + img * 0.5
            # save_cam_dir = os.path.join(
            #     '/home/tzq-cz/code/causal_graph_semi_seg_auto/myExperiment/results/EM_rounds_prostate/round_1/prediction/rw_overlay')
            # cv2.imwrite(os.path.join(save_cam_dir, img_name.replace('Image', 'Label') + '_overlay.jpg'), result)

            # rw_up_bg = F.pad(rw_up, (0, 0, 0, 0, 1, 0), value=args.sem_seg_bg_thres)
            # rw_pred = torch.argmax(rw_up_bg, dim=0).cpu().numpy()
            #
            # rw_pred = keys[rw_pred]
            #
            # imageio.imsave(os.path.join(args.round_out_dir, 'prediction', args.sem_seg_out_dir, img_name.replace('Image', 'Label') + '.png'),
            #                (rw_pred*255).astype(np.uint8))

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')


def run(args):
    model = getattr(importlib.import_module(args.irn_network), 'EdgeDisplacement')()
    model.load_state_dict(torch.load(os.path.join(args.round_out_dir, 'checkpoints', args.irn_weights_name)), strict=False)
    # model.load_state_dict(torch.load(os.path.join(
    #     '/home/tzq-cz/code/causal_graph_semi_seg_auto/myExperiment/results/EM_rounds_prostate/round_0/checkpoints',
    #     args.irn_weights_name)), strict=False)
    model.eval()

    n_gpus = torch.cuda.device_count()

    dataset = data.dataloader_prostate.ProstateClassificationDatasetMSF(args.infer_list,
                                                             prostate_root=args.data_root,
                                                             scales=(1.0,))
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print("[", end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print("]")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    from options.experiment_options import MyOptions
    args = MyOptions().parse()
    print(args)
    # args.num_workers = 1
    # args.voc12_root = '/home/tzq-cz/data/prostate_MR'
    # args.infer_list = '../../data/train_valid_prostate.txt'
    it_n = 1
    print(f'round = {it_n}')
    args.currRound = it_n
    args.round_out_dir = args.output_dir + 'round_' + str(it_n) + '/'
    args.round_in_dir = args.output_dir + 'round_' + str(it_n-1) + '/'
    # args.conf_fg_thres = 0.6
    # args.conf_bg_thres = 0.2
    # args.sem_seg_bg_thres = 0.75
    args.exp_times = 1
    args.cam_out_dir = 'lpcam'
    args.save_rw_dir = 'volume_rw'
    args.ir_label_out_dir = 'volume_ir_label'
    args.irn_weights_name = 'volume_res50_irn.pth'
    run(args)