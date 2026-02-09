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

import data.dataloader_lits
from misc import torchutils, imutils
from kmeans_pytorch import kmeans
import random
import time
from tqdm import tqdm
from options.experiment_options import MyOptions
from misc import pyutils
import matplotlib.pyplot as plt

cudnn.enabled = True

class_id_to_name = ['liver']

def save_feature(save_dir,ckpt_path,data_root): ####### save feature from resnet50 at 'cluster/cam_feature/'
    model = getattr(importlib.import_module("net.resnet50_cam"), 'Net_CAM')()
    model.load_state_dict(torch.load((ckpt_path)), strict=True)
    model.eval()
    model.cuda()

    infer_dataset = data.dataloader_lits.LitsClassificationDataset("data/valid_1_2_train.txt", data_root==data_root,
                                                                phase='train', resize_long=(320, 640), hor_flip=True,
                                                                crop_size=512, crop_method="random")
    infer_data_loader = DataLoader(infer_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    print(len(infer_data_loader))
    tensor_logits = torch.zeros(len(infer_data_loader),1)
    tensor_label = torch.zeros(len(infer_data_loader),1)
    tensor_feature = {}
    name2id = dict()
    with torch.no_grad():
        for i, pack in enumerate(infer_data_loader):

            img_name = pack['name']
            label = pack['label']
            img = pack['img'].cuda(non_blocking=True)
            
            x,cams,feature = model(img)
            name2id[img_name[0]] = i
            tensor_logits[i] = x[0].cpu()
            tensor_feature[i] = feature[0].cpu()
            tensor_label[i] = label[0]

    os.makedirs(save_dir,exist_ok=True)
    torch.save(tensor_logits, os.path.join(save_dir,'tensor_logits.pt'))
    torch.save(tensor_feature,os.path.join(save_dir,'tensor_feature.pt'))
    torch.save(tensor_label,os.path.join(save_dir,'tensor_label.pt'))
    np.save(os.path.join(save_dir,'name2id.npy'), name2id)

def load_feature_select_and_cluster(workspace, feature_dir, mask_dir, ckpt_path, load_cluster=False, num_cluster=12,select_thres=0.1,class_thres=0.9,context_thres=0.9,context_thres_low=0.05,tol=5):
    tensor_feature = torch.load(os.path.join(feature_dir,'tensor_feature.pt'))
    tensor_label = torch.load(os.path.join(feature_dir,'tensor_label.pt'))
    name2id = np.load(os.path.join(feature_dir,'name2id.npy'), allow_pickle=True).item()
    id2name = {}
    for key in name2id.keys():
        id2name[name2id[key]] = key
    
    ### load model for calc similarity
    model = getattr(importlib.import_module("net.resnet50_cam"), 'Net_CAM')()
    model.load_state_dict(torch.load((ckpt_path)), strict=True)
    w = model.classifier.weight.data.squeeze()
    
    ####### feature cluster #####
    centers = {}
    context = {}
    for class_id in range(1):
        print()
        print('class id: ', class_id,', class name:',class_id_to_name[class_id])
        cluster_result_dir = os.path.join(workspace,'cluster_result')
        os.makedirs(cluster_result_dir, exist_ok=True)
        
        if load_cluster:
            cluster_centers = torch.load(os.path.join(cluster_result_dir,'cluster_centers_'+str(class_id)+'.pt'))
            cluster_centers2 = torch.load(os.path.join(cluster_result_dir,'cluster_centers2_'+str(class_id)+'.pt'))
            cluster_ids_x = torch.load(os.path.join(cluster_result_dir,'cluster_ids_x_'+str(class_id)+'.pt'))
            cluster_ids_x2 = torch.load(os.path.join(cluster_result_dir,'cluster_ids_x2_'+str(class_id)+'.pt'))
        else:
            img_selected = torch.nonzero(tensor_label[:,class_id])[:,0].numpy()
            random.shuffle(img_selected)
            img_selected = np.sort(img_selected[0:200])
            feature_selected = []
            feature_not_selected = []
            for idx in img_selected:
                name = id2name[idx]
                cam = np.load(os.path.join(mask_dir, name+'.npy'), allow_pickle=True).item()
                mask = cam['high_res']
                valid_cat = cam['keys']
                feature_map = tensor_feature[idx].permute(1,2,0)
                size = feature_map.shape[:2]
                mask = F.interpolate(torch.tensor(mask).unsqueeze(0),size)[0]
                for i in range(len(valid_cat)):
                    if valid_cat[i]==class_id:
                        mask = mask[i]
                        position_selected = mask>select_thres
                        position_not_selected = mask<select_thres
                        feature_selected.append(feature_map[position_selected])
                        feature_not_selected.append(feature_map[position_not_selected])
            feature_selected = torch.cat(feature_selected,0)
            feature_not_selected = torch.cat(feature_not_selected,0)
            

            cluster_ids_x, cluster_centers = kmeans(X=feature_selected, num_clusters=num_cluster, distance='cosine', device=torch.device('cuda:0'), tol=tol)
            cluster_ids_x2, cluster_centers2 = kmeans(X=feature_not_selected, num_clusters=num_cluster, distance='cosine', device=torch.device('cuda:0'), tol=tol)
        
            torch.save(cluster_centers.cpu(), os.path.join(cluster_result_dir,'cluster_centers_'+str(class_id)+'.pt'))
            torch.save(cluster_centers2.cpu(), os.path.join(cluster_result_dir,'cluster_centers2_'+str(class_id)+'.pt'))
            torch.save(cluster_ids_x.cpu(), os.path.join(cluster_result_dir,'cluster_ids_x_'+str(class_id)+'.pt'))
            torch.save(cluster_ids_x2.cpu(), os.path.join(cluster_result_dir,'cluster_ids_x2_'+str(class_id)+'.pt'))
        

        ###### calc similarity
        w = w.unsqueeze(0)
        sim = torch.cosine_similarity(cluster_centers, w, dim=1)
        s_sim, loc = torch.sort(sim, descending=True)
        # sim = torch.mm(cluster_centers,w.T)
        # prob = F.softmax(sim,dim=1)
        
        ###### select center
        pre_k = 3
        selected_cluster = torch.nn.functional.one_hot(loc[0:pre_k], num_classes=num_cluster).sum(dim=0) > 0
        # selected_cluster = prob[:,class_id]>class_thres
        cluster_center = cluster_centers[selected_cluster]
        centers[class_id] = cluster_center.cpu()
        
        ##### print similarity matrix
        # print_tensor(prob.numpy())
        # for i in range(num_cluster):
        #     print(selected_cluster[i].item(), round(prob[i,class_id].item(),3), torch.sum(cluster_ids_x==i).item())
        
        ###### calc similarity
        sim = torch.cosine_similarity(cluster_centers2, w, dim=1)
        s_sim, loc = torch.sort(sim, descending=False)
        # sim = torch.mm(cluster_centers2,w.T)
        # prob = F.softmax(sim,dim=1)
        
        ###### select context
        pre_k = 10
        selected_cluster = torch.nn.functional.one_hot(loc[0:pre_k], num_classes=num_cluster).sum(dim=0) > 0
        # selected_cluster = (prob[:,class_id]>context_thres_low)*(prob[:,class_id]<context_thres)
        cluster_center2 = cluster_centers2[selected_cluster]
        context[class_id] = cluster_center2.cpu()
        
        ##### print similarity matrix
        # print_tensor(prob.numpy())
        # for i in range(num_cluster):
        #     print(selected_cluster[i].item(), round(prob[i,class_id].item(),3), torch.sum(cluster_ids_x2==i).item())

    # torch.save(centers.cpu(), osp.join(workspace+'class_ceneters'+'.pt'))
    torch.save(centers, os.path.join(workspace,'class_ceneters'+'.pt'))
    torch.save(context, os.path.join(workspace,'class_context'+'.pt'))

def make_lpcam(args,workspace,lpcam_out_dir,ckpt_path,data_root,list_name='voc12/train.txt'):
    cluster_centers = torch.load(os.path.join(workspace,'class_ceneters'+'.pt'))
    cluster_context = torch.load(os.path.join(workspace,'class_context'+'.pt'))
    
    model = getattr(importlib.import_module(args.cam_network), 'Net_CAM')()
    model.load_state_dict(torch.load(os.path.join(args.round_out_dir, 'checkpoints', args.cam_weights_name)), strict=True)
    model.eval()
    model.cuda()

    dataset = data.dataloader_lits.LitsClassificationDatasetMSF(args.infer_list,
                                                             data_root=args.data_root, scales=args.cam_scales)
    data_loader = DataLoader(dataset, shuffle=False, num_workers=args.num_workers // 1, pin_memory=False)
    start_time = time.time()
    with torch.no_grad():
        for i, pack in enumerate(tqdm(data_loader)):
            imgs = pack['img']
            label = pack['label'][0]
            img_name = pack['name'][0]
            size = pack['size']
            valid_cat = torch.nonzero(label)[:, 0].numpy()
            
            # if osp.exists(os.path.join(cam_out_dir, img_name+'.npy')):
            #     continue
            
            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)
            
            features = []
            for img in imgs:
                _,_,feature = model(img[0].cuda())
                features.append((feature[0]+feature[1].flip(-1)))
                
            
            strided_cams = []
            highres_cams = []
            for class_id in valid_cat:
                strided_cam = []
                highres_cam = []
                for feature in features:
                    h,w = feature.shape[1],feature.shape[2]    
                    cluster_feature = cluster_centers[class_id]
                    att_maps = []
                    for j in range(cluster_feature.shape[0]): 
                        cluster_feature_here = cluster_feature[j].repeat(h,w,1).cuda()
                        feature_here = feature.permute((1,2,0)).reshape(h,w,2048)
                        attention_map = F.cosine_similarity(feature_here,cluster_feature_here,2).unsqueeze(0).unsqueeze(0)
                        att_maps.append(attention_map.cpu())
                    att_map = torch.mean(torch.cat(att_maps,0),0,keepdim=True).cuda()
                    
                    context_feature = cluster_context[class_id]
                    if context_feature.shape[0]>0:
                        context_attmaps = []
                        for j in range(context_feature.shape[0]):
                            context_feature_here = context_feature[j]
                            context_feature_here = context_feature_here.repeat(h,w,1).cuda()
                            context_attmap = F.cosine_similarity(feature_here,context_feature_here,2).unsqueeze(0).unsqueeze(0)
                            context_attmaps.append(context_attmap.unsqueeze(0))
                        context_attmap = torch.mean(torch.cat(context_attmaps,0),0)
                        # att_map = F.relu(att_map - context_attmap)
                        att_map = F.relu(1- context_attmap)
                    
                    attention_map1 = F.interpolate(att_map, strided_size,mode='bilinear', align_corners=False)[:,0,:,:]
                    attention_map2 = F.interpolate(att_map, strided_up_size,mode='bilinear', align_corners=False)[:,0,:size[0],:size[1]]
                    strided_cam.append(attention_map1.cpu())
                    highres_cam.append(attention_map2.cpu())
                strided_cam = torch.mean(torch.cat(strided_cam,0),0)
                highres_cam = torch.mean(torch.cat(highres_cam,0),0)
                strided_cam = strided_cam/torch.max(strided_cam)                
                highres_cam = highres_cam/torch.max(highres_cam)                
                strided_cams.append(strided_cam.unsqueeze(0))
                highres_cams.append(highres_cam.unsqueeze(0))
            strided_cams = torch.cat(strided_cams,0)
            highres_cams = torch.cat(highres_cams,0)
            np.save(os.path.join(lpcam_out_dir, img_name.replace('jpg','npy')),{"keys": valid_cat,"cam": strided_cams,"high_res": highres_cams})

def _work(process_id, model, dataset, args):

    databin = dataset[process_id]
    # n_gpus = torch.cuda.device_count()
    n_gpus = 1
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()

        for iter, pack in enumerate(data_loader):

            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']

            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)

            outputs = [model(img[0].cuda(non_blocking=True))
                       for img in pack['img']]
            # print('img_name:', img_name)
            # print('label:', label)
            # print('size:', size)
            # print('outputs.shape', outputs[0].shape)
            # print('outputs.len:', len(outputs))

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


# def make_lpcam(args,workspace,lpcam_out_dir,ckpt_path,voc12_root,list_name='voc12/train.txt'):
#     cluster_centers = torch.load(osp.join(workspace,'class_ceneters'+'.pt'))
#     cluster_context = torch.load(osp.join(workspace,'class_context'+'.pt'))

#     model = getattr(importlib.import_module(args.cam_network), 'CAM')()
#     model.load_state_dict(torch.load(os.path.join(args.round_out_dir, 'checkpoints', args.cam_weights_name)), strict=True)
#     model.eval()

#     n_gpus = torch.cuda.device_count()

#     dataset = data.dataloader_prostate.ProstateClassificationDatasetMSF(args.infer_list,
#                                                              prostate_root=args.data_root, scales=args.cam_scales)
#     dataset = torchutils.split_dataset(dataset, n_gpus)

#     print('[ ', end='')
#     multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
#     print(']')

#     torch.cuda.empty_cache()

def run(args):
    feature_dir_ = os.path.join(args.round_out_dir,'cam_feature')
    mask_dir_ = os.path.join(args.round_out_dir, 'prediction', args.cam_out_dir)
    ckpt_path_ = os.path.join(args.round_out_dir, 'checkpoints', args.cam_weights_name)
    lpcam_out_dir_ = os.path.join(args.round_out_dir,'prediction',args.lpcam_out_dir)
    save_feature(save_dir=os.path.join(args.round_out_dir,'cam_feature'),ckpt_path=os.path.join(args.round_out_dir, 'checkpoints', args.cam_weights_name),data_root=args.data_root)
    load_feature_select_and_cluster(args.round_out_dir,feature_dir=feature_dir_, mask_dir=mask_dir_, ckpt_path=ckpt_path_, select_thres=0.5)
    make_lpcam(args,workspace=args.round_out_dir,lpcam_out_dir=lpcam_out_dir_,ckpt_path=ckpt_path_,data_root=args.data_root,list_name=args.infer_list)

if __name__ == "__main__":
    args = MyOptions().parse()
    pyutils.Logger(args.log_name + '_make_lpcam.log')
    print(vars(args))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    it_n = 0
    print(f'round = {it_n}')
    args.currRound = it_n
    args.round_out_dir = args.output_dir + 'round_' + str(it_n) + '/'
    args.round_in_dir = args.output_dir + 'round_' + str(it_n) + '/'
    run(args)