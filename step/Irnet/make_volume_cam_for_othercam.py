import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn
import numpy as np
import importlib
import os

import sys
sys.path.append("/data3/masx/code/Volume-C-CAM")

import data.dataloader_prostate
from misc import torchutils, imutils
from kmeans_pytorch import kmeans
import random
import time
from tqdm import tqdm
from options.experiment_options import MyOptions
from misc import pyutils
import matplotlib
import matplotlib.pyplot as plt


# matplotlib.use('Agg')

cudnn.enabled = True

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class_id_to_name = ['prostate']


def make_volume_cam(args, model):
    # generate feature from resnet50 and save (optional) at 'cluster/cam_feature/'
    # ckpt_path = ckpt_path = os.path.join(
    #     args.round_out_dir, 'checkpoints', args.cam_weights_name)
    # model = getattr(importlib.import_module(args.cam_network), 'Net_CAM')()
    # model.load_state_dict(torch.load((ckpt_path)), strict=True)
    # model.eval()
    # model.cuda()

    infer_dataset = data.dataloader_prostate.ProstateClassificationDataset(args.infer_list, prostate_root=args.data_root,
                                                                           phase='train')
    infer_data_loader = DataLoader(
        infer_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    # print()
    print(len(infer_data_loader))
    tensor_logits = torch.zeros(len(infer_data_loader), 1)
    tensor_label = torch.zeros(len(infer_data_loader), 1)
    tensor_feature = {}
    name2id = dict()

    if not args.load_feature:
        with torch.no_grad():
            for i, pack in enumerate(infer_data_loader):

                img_name = pack['name']
                label = pack['label']
                img = pack['img'].cuda(non_blocking=True)

                x, cams, feature = model(img)
                name2id[img_name[0]] = i
                tensor_logits[i] = x[0].cpu()
                tensor_feature[i] = feature[0].cpu()
                tensor_label[i] = label[0]

        if args.save_feature:
            os.makedirs(os.path.join(args.round_out_dir, 'cam_feature'), exist_ok=True)
            torch.save(tensor_logits, os.path.join(args.round_out_dir,
                    'cam_feature/tensor_logits_'+str(args.volume_id).zfill(4)+'.pt'))
            torch.save(tensor_feature, os.path.join(args.round_out_dir,
                    'cam_feature/tensor_feature_'+str(args.volume_id).zfill(4)+'.pt'))
            torch.save(tensor_label, os.path.join(args.round_out_dir,
                    'cam_feature/tensor_label_'+str(args.volume_id).zfill(4)+'.pt'))
            np.save(os.path.join(args.round_out_dir, 'cam_feature/name2id_' +
                    str(args.volume_id).zfill(4)+'.npy'), name2id)
    

    if args.load_feature:
        tensor_feature = torch.load(os.path.join(
            args.round_out_dir, 'cam_feature/tensor_feature_'+str(args.volume_id).zfill(4)+'.pt'))
        tensor_label = torch.load(os.path.join(
            args.round_out_dir, 'cam_feature/tensor_label_'+str(args.volume_id).zfill(4)+'.pt'))
        name2id = np.load(os.path.join(args.round_out_dir, 'cam_feature/name2id_' +
                          str(args.volume_id).zfill(4)+'.npy'), allow_pickle=True).item()
        tensor_logits = torch.load(os.path.join(
            args.round_out_dir, 'cam_feature/tensor_logits_'+str(args.volume_id).zfill(4)+'.pt'))

    if args.confounder:
        confounder_dict = np.load(os.path.join(
            args.round_in_dir, 'confounder_full' + '.npy'), allow_pickle=True)
        confounder = torch.tensor(confounder_dict[1, :, :]).unsqueeze(0)
        confounder = F.interpolate(confounder.unsqueeze(0), (32, 32))[0][0]

    id2name = {}
    for key in name2id.keys():
        id2name[name2id[key]] = key

    # obtain model parameters for calc similarity
    w = model.classifier.weight.data.squeeze()

    ####### obtain feature center (fg and bg) #####
    selected_fg_centers = {}
    selected_bg_context = {}
    class_id = 0
    # print()
    print('class id: ', class_id, ', class name:', class_id_to_name[0])
    cluster_result_dir = os.path.join(args.round_out_dir, 'cluster_result')
    selected_cluster_result_dir = os.path.join(
        args.round_out_dir, 'selected_cluster')
    os.makedirs(cluster_result_dir, exist_ok=True)
    os.makedirs(selected_cluster_result_dir, exist_ok=True)

    if args.load_cluster:
        cluster_centers = torch.load(os.path.join(
            cluster_result_dir, 'cluster_centers_'+str(0)+'_'+str(args.volume_id).zfill(4)+'.pt'))
        cluster_centers2 = torch.load(os.path.join(
            cluster_result_dir, 'cluster_centers2_'+str(0)+'_'+str(args.volume_id).zfill(4)+'.pt'))
    elif not args.load_selected_cluster:
        img_selected = torch.nonzero(tensor_label[:, class_id])[:, 0].numpy()
        fg_feature_selected = []
        bg_context_selected = []

        mask_dir = os.path.join(
            '/data1/chenz/code/pytorch-grad-cam-master/cam_npy/ACDC/layercam')
        for idx in img_selected:
            name = id2name[idx]
            # cam = np.load(os.path.join(mask_dir, name+'.npy'),
            #               allow_pickle=True).item()
            # mask = cam['high_res']
            # valid_cat = cam['keys']
            valid_cat = [0]
            mask = np.load(os.path.join(mask_dir, name+'.npy'))
            feature_map = tensor_feature[idx].permute(1, 2, 0)
            size = feature_map.shape[:2]
            mask = F.interpolate(torch.tensor(mask).unsqueeze(0).unsqueeze(0), size)[0]
            for i in range(len(valid_cat)):
                if valid_cat[i] == class_id:
                    mask = mask[i]
                    fg_position_selected = (mask) > 0.9
                    bg_position_selected = (mask) < 0.7
                    fg_feature_selected.append(
                        feature_map[fg_position_selected])
                    bg_context_selected.append(
                        feature_map[bg_position_selected])
        fg_feature_selected = torch.cat(fg_feature_selected, 0)
        bg_context_selected = torch.cat(bg_context_selected, 0)

        if args.cluster:
            cluster_num = 5
            cluster_ids_x, cluster_centers = kmeans(
                X=fg_feature_selected, num_clusters=cluster_num, distance='cosine', device=torch.device('cuda:0'), tol=5)
            cluster_ids_x2, cluster_centers2 = kmeans(
                X=bg_context_selected, num_clusters=cluster_num, distance='cosine', device=torch.device('cuda:0'), tol=5)

            # calc similarity for fg centers
            w = w.unsqueeze(0).cpu()
            sim = torch.cosine_similarity(cluster_centers, w, dim=1)
            s_sim, loc = torch.sort(sim, descending=True)
            # select center
            pre_k = 1
            selected_cluster = torch.nn.functional.one_hot(
                loc[0:pre_k], num_classes=cluster_num).sum(dim=0) > 0
            cluster_center = cluster_centers[selected_cluster]
            selected_fg_centers[class_id] = cluster_center.cpu()

            # calc similarity for bg centers
            sim2 = torch.cosine_similarity(cluster_centers2, w, dim=1)
            s_sim2, loc2 = torch.sort(sim2, descending=False)
            # select context
            pre_k = 1
            selected_cluster2 = torch.nn.functional.one_hot(
                loc2[0:pre_k], num_classes=cluster_num).sum(dim=0) > 0
            cluster_center2 = cluster_centers2[selected_cluster2]
            selected_bg_context[class_id] = cluster_center2.cpu()

        if args.mean:
            cluster_centers = fg_feature_selected.mean(keepdim=True, dim=0)
            cluster_centers2 = bg_context_selected.mean(keepdim=True, dim=0)
            selected_fg_centers[class_id] = cluster_centers.cpu()
            selected_bg_context[class_id] = cluster_centers2.cpu()

        torch.save(cluster_centers.cpu(), os.path.join(cluster_result_dir,
                                                       'cluster_centers_'+str(class_id)+'_'+str(args.volume_id).zfill(4)+'.pt'))
        torch.save(cluster_centers2.cpu(), os.path.join(cluster_result_dir,
                                                        'cluster_centers2_'+str(class_id)+'_'+str(args.volume_id).zfill(4)+'.pt'))

        torch.save(selected_fg_centers, os.path.join(
            selected_cluster_result_dir, 'class_ceneters'+'_'+str(args.volume_id).zfill(4)+'.pt'))
        torch.save(selected_bg_context, os.path.join(
            selected_cluster_result_dir, 'class_context'+'_'+str(args.volume_id).zfill(4)+'.pt'))
        
    else:
        selected_fg_centers = torch.load(os.path.join(selected_cluster_result_dir,'class_ceneters'+'_'+str(args.volume_id).zfill(4)+'.pt'))
        selected_bg_context = torch.load(os.path.join(selected_cluster_result_dir,'class_context'+'_'+str(args.volume_id).zfill(4)+'.pt'))


    # make lpcam
    dataset = data.dataloader_prostate.ProstateClassificationDatasetMSF(args.infer_list,
                                                                        prostate_root=args.data_root, scales=args.cam_scales)
    data_loader = DataLoader(
        dataset, shuffle=False, num_workers=18 // 1, pin_memory=False)
    start_time = time.time()

    with torch.no_grad():
        for i, pack in enumerate(tqdm(data_loader)):
            imgs = pack['img']
            label = pack['label'][0]
            img_name = pack['name'][0]
            size = pack['size']
            valid_cat = torch.nonzero(label)[:, 0].numpy()

            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)

            features = []
            for img in imgs:
                _, _, feature = model(img[0].cuda())
                features.append((feature[0]+feature[1].flip(-1)))

            strided_cams = []
            highres_cams = []
            for class_id in valid_cat:
                strided_cam = []
                highres_cam = []
                for feature in features:
                    hh, ww = feature.shape[1], feature.shape[2]
                    confounder_r = F.interpolate(confounder.unsqueeze(0).unsqueeze(0), (hh,ww), mode='bilinear', align_corners=False)
                    # calc fg att_map
                    cluster_feature = selected_fg_centers[class_id]
                    att_maps = []
                    for j in range(cluster_feature.shape[0]):
                        # cluster_feature_here = cluster_feature[j].repeat(hh,ww,1).cuda()
                        cluster_feature_here = cluster_feature[j].detach().cpu()
                        cluster_feature_here = cluster_feature_here / torch.norm(cluster_feature_here, dim=0, keepdim=True)
                        feature_here = feature.permute(
                            (1, 2, 0)).view(1, hh*ww, -1).detach().cpu()
                        # attention_map = F.cosine_similarity(feature_here,cluster_feature_here,2).unsqueeze(0).unsqueeze(0)
                        attention_map = torch.matmul(feature_here, cluster_feature_here.view(1, -1, 1)).view(1, 1, hh, ww)
                        # attention_map = confounder_r * attention_map
                        # attention_map = torch.mean(feature_here,-1).unsqueeze(0).unsqueeze(0)
                        att_maps.append(attention_map.cpu())
                    att_map = torch.mean(
                        torch.cat(att_maps, 0), 0, keepdim=True)
                    # att_map_norm = att_map/torch.max(att_map)

                    # calc bg contex_attmap
                    context_feature = selected_bg_context[class_id]
                    if context_feature.shape[0] > 0:
                        context_attmaps = []
                        for j in range(context_feature.shape[0]):
                            context_feature_here = context_feature[j]
                            context_feature_here = context_feature_here / torch.norm(context_feature_here, dim=0, keepdim=True)
                            # context_feature_here = context_feature_here.repeat(
                            #     hh, ww, 1).cuda()
                            # context_attmap = F.cosine_similarity(
                            #     feature_here, context_feature_here, 2).unsqueeze(0).unsqueeze(0)
                            context_attation_map = torch.matmul(feature_here,context_feature_here).unsqueeze(0).unsqueeze(0)
                            # confounder_r = F.interpolate(confounder.unsqueeze(0).unsqueeze(0), (hh,ww), mode='bilinear', align_corners=False)
                            # context_attation_map = ((1-confounder_r).view(1, 1, 1, hh*ww) * context_attation_map).unsqueeze(0).unsqueeze(0)
                            # max_value, max_idx = torch.max(feature_here, -1)
                            # context_attmap = max_value.unsqueeze(0).unsqueeze(0)
                            context_attmaps.append(
                                context_attation_map.unsqueeze(0))
                        context_attmap = torch.mean(
                            torch.cat(context_attmaps, 0), 0)
                        # context_attmap_norm = 1- context_attmap / torch.max(context_attmap)
                        # context_attmap_norm = 1 - confounder_r
                        # att_map = F.relu(att_map_norm - context_attmap_norm.view(1, 1, hh, ww))
                        att_map = F.relu(att_map - context_attmap.view(1, 1, hh, ww))
                        # att_map = F.relu(att_map)
                        # att_map = F.relu(context_attmap)

                    attention_map1 = F.interpolate(
                        att_map, strided_size, mode='bilinear', align_corners=False)[:, 0, :, :]
                    attention_map2 = F.interpolate(
                        att_map, strided_up_size, mode='bilinear', align_corners=False)[:, 0, :size[0], :size[1]]
                    strided_cam.append(attention_map1.cpu())
                    highres_cam.append(attention_map2.cpu())

                strided_cam = torch.mean(torch.cat(strided_cam, 0), 0)
                highres_cam = torch.mean(torch.cat(highres_cam, 0), 0)
                strided_cam = strided_cam/torch.max(strided_cam)
                highres_cam = highres_cam/torch.max(highres_cam)
                strided_cams.append(strided_cam.unsqueeze(0))
                highres_cams.append(highres_cam.unsqueeze(0))

            strided_cams = torch.cat(strided_cams, 0)
            highres_cams = torch.cat(highres_cams, 0)
            lpcam_out_dir = os.path.join(args.round_out_dir,'prediction',args.lpcam_out_dir)
            os.makedirs(lpcam_out_dir, exist_ok=True)
            np.save(os.path.join(lpcam_out_dir, img_name.replace('jpg', 'npy')), {
                    "keys": valid_cat, "cam": strided_cams, "high_res": highres_cams})
            del attention_map


if __name__ == "__main__":
    # list1 = [i for i in range(1, 107)]
    # list2 = [i for i in range(108, 116)]
    # list3 = [i for i in range(117, 131)]
    # list4 = [i for i in range(135, 136)]
    # list5 = [i for i in range(137, 141)]
    # volume_list = list1+list2+list3+list4+list5
    volume_list = [i for i in range(1, 175)]
    args = MyOptions().parse()
    pyutils.Logger(args.log_name + '_make_volume_cam_cz.log')
    print(vars(args))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    it_n = 0
    print(f'round = {it_n}')
    args.currRound = it_n
    args.round_out_dir = args.output_dir + 'layercam' + '/'
    args.round_in_dir = args.output_dir + 'round_' + str(it_n) + '/'

    args.save_feature = True  # whether save image features
    args.load_feature = False  # whether load saved image features
    args.confounder = True  # whether use confounder
    # whether use cluster method to generate feature centers (fg or bg)
    args.cluster = True
    args.load_cluster = False  # whether load saved clusters
    args.load_selected_cluster = False  # whether load selected clusters
    # whether use mean method to generate feature centers (fg or bg)
    args.mean = False

    ##### load model
    ckpt_path = ckpt_path = os.path.join(
        '/data1/chenz/code/pytorch-grad-cam-master/checkpoints/', 'ACDC', args.cam_weights_name)
    model = getattr(importlib.import_module(args.cam_network), 'Net_CAM')()
    model.load_state_dict(torch.load((ckpt_path)), strict=True)
    model.eval()
    model.cuda()
    
    for volume_id in volume_list:
        print()
        print("volume_" + str(volume_id).zfill(4))
        args.volume_id = volume_id
        args.infer_list = "data/acdc/train_valid_acdc_" + str(volume_id).zfill(2) + ".txt"
        make_volume_cam(args, model)
