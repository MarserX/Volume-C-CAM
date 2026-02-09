import numpy as np
import torch
from torch.utils.data import Dataset
import os.path
import imageio
from misc import imutils
from PIL import Image


def load_img_name_list(dataset_index):
    img_gt_name_list = open(dataset_index).read().splitlines()
    img_name_list = [gt_name.split(' ')[0] for gt_name in img_gt_name_list]
    return img_name_list


def load_label_list(dataset_index):
    img_gt_name_list = open(dataset_index).read().splitlines()
    gt_list = []
    for img in img_gt_name_list:
        cls_label = np.zeros(1, np.float32)
        cat_name = img.split(' ')[1]
        if cat_name == '3':
            cls_label[0] = 1.0
        gt_list.append(cls_label)
    return gt_list


class TorchvisionNormalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img


class GetAffinityLabelFromIndices():

    def __init__(self, indices_from, indices_to):

        self.indices_from = indices_from
        self.indices_to = indices_to

    def __call__(self, segm_map):

        segm_map_flat = np.reshape(segm_map, -1)

        segm_label_from = np.expand_dims(segm_map_flat[self.indices_from], axis=0)
        segm_label_to = segm_map_flat[self.indices_to]

        # valid_label = np.logical_and(np.less(segm_label_from, 21), np.less(segm_label_to, 21))
        valid_label = np.logical_and(np.less(segm_label_from, 2), np.less(segm_label_to, 2))

        equal_label = np.equal(segm_label_from, segm_label_to)

        pos_affinity_label = np.logical_and(equal_label, valid_label)

        bg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(segm_label_from, 0)).astype(np.float32)
        fg_pos_affinity_label = np.logical_and(pos_affinity_label, np.greater(segm_label_from, 0)).astype(np.float32)

        neg_affinity_label = np.logical_and(np.logical_not(equal_label), valid_label).astype(np.float32)

        return torch.from_numpy(bg_pos_affinity_label), torch.from_numpy(fg_pos_affinity_label), \
               torch.from_numpy(neg_affinity_label)


class ProstateImageDataset(Dataset):

    def __init__(self, img_name_list_path, prostate_root, phase='train',
                 resize_long=None, rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_size=None, crop_method=None, to_torch=True):

        self.img_name_list = load_img_name_list(img_name_list_path)
        self.prostate_root = prostate_root

        self.resize_long = resize_long
        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method
        self.to_torch = to_torch
        self.phase = phase

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str = name

        if self.phase == 'train':
            img_in = np.asarray(imageio.imread(os.path.join(self.prostate_root, 'DL_Image', name + '.png')))
            if len(img_in.shape) == 2:
                img_out = np.zeros((img_in.shape[0], img_in.shape[1], 3))
                img_out[:, :, 0] = img_in
                img_out[:, :, 1] = img_in
                img_out[:, :, 2] = img_in
            else:
                img_out = img_in
            # img = np.zeros((img_.shape[0], img_.shape[1], 3))
            # img[:, :, 0] = img_
            # img[:, :, 1] = img_
            # img[:, :, 2] = img_
        if self.phase == 'val':
            img_in = np.asarray(imageio.imread(os.path.join(self.prostate_root, 'DL_Image', name + '.png')))
            if len(img_in.shape) == 2:
                img_out = np.zeros((img_in.shape[0], img_in.shape[1], 3))
                img_out[:, :, 0] = img_in
                img_out[:, :, 1] = img_in
                img_out[:, :, 2] = img_in
            else:
                img_out = img_in

        if self.resize_long:
            img_out = imutils.random_resize_long(img_out, self.resize_long[0], self.resize_long[1])

        if self.rescale:
            img = imutils.random_scale(img_out, scale_range=self.rescale, order=3)

        if self.img_normal:
            img_out = self.img_normal(img_out)

        if self.hor_flip:
            img_out = imutils.random_lr_flip(img_out)

        if self.crop_size:
            if self.crop_method == "random":
                img_out = imutils.random_crop(img_out, self.crop_size, 0)
            else:
                img_out = imutils.top_left_crop(img_out, self.crop_size, 0)

        if self.to_torch:
            img_out = imutils.HWC_to_CHW(img_out)

        return {'name': name_str, 'img': img_out}


class ProstateClassificationDataset(ProstateImageDataset):

    def __init__(self, img_name_list_path, prostate_root, phase,
                 resize_long=None, rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_size=None, crop_method=None):
        super().__init__(img_name_list_path, prostate_root, phase,
                 resize_long, rescale, img_normal, hor_flip,
                 crop_size, crop_method)
        self.label_list = load_label_list(img_name_list_path)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)

        out['label'] = torch.from_numpy(self.label_list[idx])

        return out


class ProstateClassificationDatasetTx(Dataset):

    def __init__(self, img_name_list_path, prostate_root, label_dir, phase='train',
                 resize_long=None, rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_size=None, crop_method=None, to_torch=True):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.prostate_root = prostate_root

        self.resize_long = resize_long
        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method
        self.to_torch = to_torch
        self.label_dir = label_dir
        self.label_list = load_label_list(img_name_list_path)
        self.phase = phase

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str = name
        if self.phase == 'train':
            img_in = np.asarray(imageio.imread(os.path.join(self.prostate_root, 'DL_Image', name + '.png')))
            if len(img_in.shape) == 2:
                img_out = np.zeros((img_in.shape[0], img_in.shape[1], 3))
                img_out[:, :, 0] = img_in
                img_out[:, :, 1] = img_in
                img_out[:, :, 2] = img_in
            else:
                img_out = img_in
        if self.phase == 'val':
            img_in = np.asarray(imageio.imread(os.path.join(self.prostate_root, 'DL_Image', name + '.png')))
            if len(img_in.shape) == 2:
                img_out = np.zeros((img_in.shape[0], img_in.shape[1], 3))
                img_out[:, :, 0] = img_in
                img_out[:, :, 1] = img_in
                img_out[:, :, 2] = img_in
            else:
                img_out = img_in
        # seg_pred = np.asarray(imageio.imread(os.path.join(self.label_dir, name_str + '.jpg')))
        # seg_pred = Image.open(os.path.join(self.label_dir, name_str.replace('Image', 'Label') + '.png'))
        seg_pred = Image.open(os.path.join(self.label_dir, name_str + '.png'))
        seg_pred = np.asarray(seg_pred.resize((img_out.shape[0], img_out.shape[1]), Image.NEAREST)).transpose(1, 0)
        label = torch.from_numpy(self.label_list[idx])

        if self.resize_long:
            img_out, seg_pred = imutils.random_resize_long((img_out, seg_pred), self.resize_long[0], self.resize_long[1])

        if self.rescale:
            # img = imutils.random_scale(img, scale_range=self.rescale, order=3)
            img_out, seg_pred = imutils.random_scale((img_out, seg_pred), scale_range=self.rescale, order=(3, 0))

        if self.img_normal:
            img_out = self.img_normal(img_out)

        if self.hor_flip:
            # img = imutils.random_lr_flip(img)
            img_out, seg_pred = imutils.random_lr_flip((img_out, seg_pred))

        if self.crop_size:
            if self.crop_method == "random":
                # img = imutils.random_crop(img, self.crop_size, 0)
                img_out, seg_pred = imutils.random_crop((img_out, seg_pred), self.crop_size, (0, 255))
            else:
                img_out = imutils.top_left_crop(img_out, self.crop_size, 0)
                seg_pred = imutils.top_left_crop(seg_pred, self.crop_size, 255)

        if self.to_torch:
            img_out = imutils.HWC_to_CHW(img_out)
            # seg_pred = imutils.HWC_to_CHW(seg_pred)

        return {'name': name_str, 'img': img_out, 'seg_pred': seg_pred, 'label': label}


class ProstateClassificationDatasetMSF(ProstateClassificationDataset):

    def __init__(self, img_name_list_path, prostate_root, phase='train',
                 img_normal=TorchvisionNormalize(),
                 scales=(1.0,)):
        self.scales = scales

        super().__init__(img_name_list_path, prostate_root, phase, img_normal=img_normal)
        self.scales = scales
        self.phase = phase

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str = name

        if self.phase == 'train':
            img_in = np.asarray(imageio.imread(os.path.join(self.prostate_root, 'DL_Image', name + '.png')))
            if len(img_in.shape) == 2:
                img_out = np.zeros((img_in.shape[0], img_in.shape[1], 3))
                img_out[:, :, 0] = img_in
                img_out[:, :, 1] = img_in
                img_out[:, :, 2] = img_in
            else:
                img_out = img_in
        if self.phase == 'val':
            img_in = np.asarray(imageio.imread(os.path.join(self.prostate_root, 'DL_Image', name + '.png')))
            if len(img_in.shape) == 2:
                img_out = np.zeros((img_in.shape[0], img_in.shape[1], 3))
                img_out[:, :, 0] = img_in
                img_out[:, :, 1] = img_in
                img_out[:, :, 2] = img_in
            else:
                img_out = img_in

        ms_img_list = []
        for s in self.scales:
            if s == 1:
                s_img = img_out
            else:
                s_img = imutils.pil_rescale(img_out, s, order=3)
            s_img = self.img_normal(s_img)
            s_img = imutils.HWC_to_CHW(s_img)
            ms_img_list.append(np.stack([s_img, np.flip(s_img, -1)], axis=0))
        if len(self.scales) == 1:
            ms_img_list = ms_img_list[0]

        out = {"name": name_str, "img": ms_img_list, "size": (img_out.shape[0], img_out.shape[1]),
               "label": torch.from_numpy(self.label_list[idx])}
        return out


# class ProstateClassificationDatasetTxMSF(ProstateClassificationDatasetTx):
#
#     def __init__(self, img_name_list_path, prostate_root, label_dir, phase='train',
#                  img_normal=TorchvisionNormalize(),
#                  scales=(1.0,)):
#         self.scales = scales
#
#         super().__init__(img_name_list_path, prostate_root, label_dir, phase=phase, img_normal=img_normal)
#         self.scales = scales
#
#     def __getitem__(self, idx):
#         name = self.img_name_list[idx]
#         name_str = name
#
#         if self.phase == 'train':
#             img_in = np.asarray(imageio.imread(os.path.join(self.prostate_root, 'DL_Image', name + '.png')))
#             if len(img_in.shape) == 2:
#                 img_out = np.zeros((img_in.shape[0], img_in.shape[1], 3))
#                 img_out[:, :, 0] = img_in
#                 img_out[:, :, 1] = img_in
#                 img_out[:, :, 2] = img_in
#             else:
#                 img_out = img_in
#         if self.phase == 'val':
#             img_in = np.asarray(imageio.imread(os.path.join(self.prostate_root, 'test_convertPNG', name + '.png')))
#             if len(img_in.shape) == 2:
#                 img_out = np.zeros((img_in.shape[0], img_in.shape[1], 3))
#                 img_out[:, :, 0] = img_in
#                 img_out[:, :, 1] = img_in
#                 img_out[:, :, 2] = img_in
#             else:
#                 img_out = img_in
#         # seg_pred = np.asarray(imageio.imread(os.path.join(self.label_dir, name_str.replace('Image', 'Label') + '.png')))
#         seg_pred = np.asarray(imageio.imread(os.path.join(self.label_dir, name_str + '.png')))
#         # seg_pred = Image.open(os.path.join(self.label_dir, name_str + '.jpg'))
#         # seg_pred = np.asarray(seg_pred.resize((img.shape[0], img.shape[1]), Image.NEAREST)).transpose(1, 0)
#
#         ms_img_list = []
#         ms_seg_pred_list = []
#         for s in self.scales:
#             if s == 1:
#                 s_img = img_out
#                 s_seg_pred = seg_pred
#             else:
#                 s_img = imutils.pil_rescale(img_out, s, order=3)
#                 # s_seg_pred = imutils.pil_rescale(seg_pred, s, order=3)
#             s_img = self.img_normal(s_img)
#             s_img = imutils.HWC_to_CHW(s_img)
#             ms_img_list.append(np.stack([s_img, np.flip(s_img, -1)], axis=0))
#             ms_seg_pred_list.append(np.stack([s_seg_pred, np.flip(s_seg_pred, -1)], axis=0))
#         if len(self.scales) == 1:
#             ms_img_list = ms_img_list[0]
#             ms_seg_pred_list = ms_seg_pred_list[0]
#
#         out = {"name": name_str, "img": ms_img_list, "size": (img_out.shape[0], img_out.shape[1]),
#                "label": torch.from_numpy(self.label_list[idx]), "seg_pred": ms_seg_pred_list}
#         # print("seg_pred.shape:", s_seg_pred.shape)
#         # print("img.shape:", s_img.shape)
#         # print("ms_seg_pred_list.len:", len(ms_seg_pred_list))
#         # print("ms_img_list.len:", len(ms_img_list))
#         return out

class ProstateClassificationDatasetTxMSF(ProstateClassificationDatasetTx):

    def __init__(self, img_name_list_path, prostate_root, label_dir, phase='train',
                 img_normal=TorchvisionNormalize(),
                 scales=(1.0,)):
        self.scales = scales

        super().__init__(img_name_list_path, prostate_root, label_dir, phase=phase, img_normal=img_normal)
        self.scales = scales

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str = name

        if self.phase == 'train':
            img_in = np.asarray(imageio.imread(os.path.join(self.prostate_root, 'DL_Image', name + '.png')))

            if len(img_in.shape) == 2:
                img_out = np.zeros((img_in.shape[0], img_in.shape[1], 3))
                img_out[:, :, 0] = img_in
                img_out[:, :, 1] = img_in
                img_out[:, :, 2] = img_in
            else:
                img_out = img_in
            img = np.uint8(img_out)
        if self.phase == 'val':
            img_ = np.asarray(imageio.imread(os.path.join(self.prostate_root, 'DL_Image', name + '.png')))
            # print(img.shape)
            if len(img_in.shape) == 2:
                img_out = np.zeros((img_in.shape[0], img_in.shape[1], 3))
                img_out[:, :, 0] = img_in
                img_out[:, :, 1] = img_in
                img_out[:, :, 2] = img_in
            else:
                img_out = img_in
            img = np.uint8(img_out)
        # seg_pred = np.asarray(imageio.imread(os.path.join(self.label_dir, name_str.replace('Image', 'Label') + '.png')))
        seg_pred = Image.open(os.path.join(self.label_dir, name_str + '.png'))
        seg_pred = np.asarray(seg_pred.resize((512, 512), Image.NEAREST)).transpose(1, 0)

        ms_img_list = []
        ms_seg_pred_list = []
        for s in self.scales:
            if s == 1:
                s_img = img
                s_seg_pred = seg_pred
            else:
                s_img = imutils.pil_rescale(img, s, order=3)
                # s_seg_pred = imutils.pil_rescale(seg_pred, s, order=3)
            s_img = self.img_normal(s_img)
            s_img = imutils.HWC_to_CHW(s_img)
            ms_img_list.append(np.stack([s_img, np.flip(s_img, -1)], axis=0))
            ms_seg_pred_list.append(np.stack([s_seg_pred, np.flip(s_seg_pred, -1)], axis=0))
        if len(self.scales) == 1:
            ms_img_list = ms_img_list[0]
            ms_seg_pred_list = ms_seg_pred_list[0]

        out = {"name": name_str, "img": ms_img_list, "size": (img.shape[0], img.shape[1]),
               "label": torch.from_numpy(self.label_list[idx]), "seg_pred": ms_seg_pred_list}
        # print("seg_pred.shape:", s_seg_pred.shape)
        # print("img.shape:", s_img.shape)
        # print("ms_seg_pred_list.len:", len(ms_seg_pred_list))
        # print("ms_img_list.len:", len(ms_img_list))
        return out

class ProstateSegmentationDataset(Dataset):

    def __init__(self, img_name_list_path, label_dir, phase, crop_size, prostate_root,
                 rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_method = 'random'):

        self.img_name_list = load_img_name_list(img_name_list_path)
        self.prostate_root = prostate_root

        self.label_dir = label_dir

        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method
        self.phase = phase

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str = name

        if self.phase == 'train':
            img_in = np.asarray(imageio.imread(os.path.join(self.prostate_root, 'DL_Image', name + '.png')))
            if len(img_in.shape) == 2:
                img_out = np.zeros((img_in.shape[0], img_in.shape[1], 3))
                img_out[:, :, 0] = img_in
                img_out[:, :, 1] = img_in
                img_out[:, :, 2] = img_in
            else:
                img_out = img_in
        if self.phase == 'val':
            img_in = np.asarray(imageio.imread(os.path.join(self.prostate_root, 'DL_Image', name + '.png')))
            if len(img_in.shape) == 2:
                img_out = np.zeros((img_in.shape[0], img_in.shape[1], 3))
                img_out[:, :, 0] = img_in
                img_out[:, :, 1] = img_in
                img_out[:, :, 2] = img_in
            else:
                img_out = img_in
        label = imageio.imread(os.path.join(self.label_dir, name_str + '.png'))
        label[label > 0] = 1

        img_out = np.asarray(img_out)

        if self.rescale:
            img_out, label = imutils.random_scale((img_out, label), scale_range=self.rescale, order=(3, 0))

        if self.img_normal:
            img_out = self.img_normal(img_out)

        if self.hor_flip:
            img_out, label = imutils.random_lr_flip((img_out, label))

        if self.crop_method == "random":
            img_out, label = imutils.random_crop((img_out, label), self.crop_size, (0, 255))
        else:
            img_out = imutils.top_left_crop(img_out, self.crop_size, 0)
            label = imutils.top_left_crop(label, self.crop_size, 255)

        img_out = imutils.HWC_to_CHW(img_out)

        return {'name': name, 'img': img_out, 'label': label}


class ProstateAffinityDataset(ProstateSegmentationDataset):
    def __init__(self, img_name_list_path, label_dir, phase, crop_size, prostate_root,
                 indices_from, indices_to,
                 rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False, crop_method=None):
        super().__init__(img_name_list_path, label_dir, phase, crop_size, prostate_root, rescale, img_normal, hor_flip, crop_method=crop_method)

        self.extract_aff_lab_func = GetAffinityLabelFromIndices(indices_from, indices_to)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)

        reduced_label = imutils.pil_rescale(out['label'], 0.25, 0)

        out['aff_bg_pos_label'], out['aff_fg_pos_label'], out['aff_neg_label'] = self.extract_aff_lab_func(reduced_label)

        return out