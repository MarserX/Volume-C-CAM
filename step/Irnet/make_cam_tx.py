import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os

import data.dataloader_prostate
from misc import torchutils, imutils

cudnn.enabled = True

def _work(process_id, model, dataset, args):

    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()

        for iter, pack in enumerate(data_loader):

            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']

            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)

            outputs = [model(img[0].cuda(non_blocking=True), seg_pred[0].unsqueeze(1).type(torch.float32).cuda(non_blocking=True))
                       for (img, seg_pred) in zip(pack['img'], pack['seg_pred'])]
            # print('outputs.len:', len(outputs))
            # print('outputs[0].shape:', outputs[0].shape)
            # print('strided_size:', strided_size)
            # print('pack.len:', len(pack['img']))
            # print('img.shape:', pack['img'][0].shape)
            # print('seg_pred.shape:', pack['seg_pred'][0].shape)
            # print('size:', size)
            strided_cam = torch.sum(torch.stack(
                [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o
                 in outputs]), 0)

            highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                         mode='bilinear', align_corners=False) for o in outputs]
            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]

            valid_cat = torch.nonzero(label)[:, 0]

            strided_cam = strided_cam[valid_cat]
            strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5

            highres_cam = highres_cam[valid_cat]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5

            # save cams
            np.save(os.path.join(args.round_out_dir, 'prediction', args.cam_out_dir, img_name + '.npy'),
                    {"keys": valid_cat, "cam_ll": strided_cam.cpu(), "high_res": highres_cam.cpu().numpy()})

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')


def run(args):
    DIC_FILE = os.path.join(args.round_in_dir, 'confounder_full.npy')
    model = getattr(importlib.import_module(args.cam_network), 'CAM')(DIC_FILE)
    # model.load_state_dict(torch.load(args.cam_weights_name), strict=True)
    model.load_state_dict(torch.load(os.path.join(args.round_out_dir, 'checkpoints', args.cam_weights_name)),
                         strict=True)
    model.eval()

    n_gpus = torch.cuda.device_count()

    dataset = data.dataloader_prostate.ProstateClassificationDatasetTxMSF(args.infer_list, prostate_root=args.data_root,
                                                               label_dir=args.seg_pred_dir, scales=args.cam_scales)
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print(']')

    torch.cuda.empty_cache()