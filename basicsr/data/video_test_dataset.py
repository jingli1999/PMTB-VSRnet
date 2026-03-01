import glob
import torch
from os import path as osp

from torch.masked import masked_tensor
from torch.utils import data as data

from basicsr.data.data_util import duf_downsample, generate_frame_indices, read_img_seq
from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import DATASET_REGISTRY
import numpy as np
import cv2


@DATASET_REGISTRY.register()
class VideoTestDataset(data.Dataset):
    """Video test dataset.

    Supported datasets: Vid4, REDS4, REDSofficial.
    More generally, it supports testing dataset with following structures:

    ::

        dataroot
        ├── subfolder1
            ├── frame000
            ├── frame001
            ├── ...
        ├── subfolder2
            ├── frame000
            ├── frame001
            ├── ...
        ├── ...

    For testing datasets, there is no need to prepare LMDB files.
    """

    def __init__(self, opt):
        super(VideoTestDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.data_info = {'lq_path': [], 'gt_path': [], 'folder': [], 'idx': [], 'border': []}
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        assert self.io_backend_opt['type'] != 'lmdb', 'No need to use lmdb during validation/test.'

        logger = get_root_logger()
        logger.info(f'Generate data info for VideoTestDataset - {opt["name"]}')
        self.imgs_lq, self.imgs_gt = {}, {}
        if 'meta_info_file' in opt:
            with open(opt['meta_info_file'], 'r') as fin:
                subfolders = [line.split(' ')[0] for line in fin]
                subfolders_lq = [osp.join(self.lq_root, key) for key in subfolders]
                subfolders_gt = [osp.join(self.gt_root, key) for key in subfolders]
        else:
            subfolders_lq = sorted(glob.glob(osp.join(self.lq_root, '*')))
            subfolders_gt = sorted(glob.glob(osp.join(self.gt_root, '*')))

        for subfolder_lq, subfolder_gt in zip(subfolders_lq, subfolders_gt):
            subfolder_name = osp.basename(subfolder_lq)
            img_paths_lq = sorted(list(scandir(subfolder_lq, full_path=True)))
            img_paths_gt = sorted(list(scandir(subfolder_gt, full_path=True)))

            max_idx = len(img_paths_lq)
            assert max_idx == len(img_paths_gt), (f'Different number of images in lq ({max_idx})'
                                                  f' and gt folders ({len(img_paths_gt)})')

            self.data_info['lq_path'].extend(img_paths_lq)
            self.data_info['gt_path'].extend(img_paths_gt)
            self.data_info['folder'].extend([subfolder_name] * max_idx)
            for i in range(max_idx):
                self.data_info['idx'].append(f'{i}/{max_idx}')
            border_l = [0] * max_idx
            for i in range(self.opt['num_frame'] // 2):
                border_l[i] = 1
                border_l[max_idx - i - 1] = 1
            self.data_info['border'].extend(border_l)

            if self.cache_data:
                logger.info(f'Cache {subfolder_name} for VideoTestDataset...')
                self.imgs_lq[subfolder_name] = read_img_seq(img_paths_lq)
                self.imgs_gt[subfolder_name] = read_img_seq(img_paths_gt)
            else:
                self.imgs_lq[subfolder_name] = img_paths_lq
                self.imgs_gt[subfolder_name] = img_paths_gt

    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]
        lq_path = self.data_info['lq_path'][index]

        select_idx = generate_frame_indices(idx, max_idx, self.opt['num_frame'], padding=self.opt['padding'])

        if self.cache_data:
            imgs_lq = self.imgs_lq[folder].index_select(0, torch.LongTensor(select_idx))
            img_gt = self.imgs_gt[folder][idx]
        else:
            img_paths_lq = [self.imgs_lq[folder][i] for i in select_idx]
            imgs_lq = read_img_seq(img_paths_lq)
            img_gt = read_img_seq([self.imgs_gt[folder][idx]])
            img_gt.squeeze_(0)

        # 新增：构造掩膜（像素值==0 视为掩膜外，其余为掩膜内）
        # gt_path = self.data_info['gt_path'][index]
        # gt_np = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        # assert gt_np is not None, f"读取GT失败：{gt_path}"
        # mask = (gt_np != 0).astype(np.float32)  # 掩膜内=1，掩膜外(=0)=0
        # mask = torch.from_numpy(mask).unsqueeze(0)  # (1,H,W)

        # assert imgs_lq.shape[1] == 1, f"LQ通道数错误，期望1，实际{imgs_lq.shape[1]}"
        # assert img_gt.shape[0] == 1, f"GT通道数错误，期望1，实际{img_gt.shape[0]}"
        # assert mask.shape == img_gt.shape, f"掩膜与GT尺寸不一致：mask {mask.shape} vs GT {img_gt.shape}"

        return {
            'lq': imgs_lq,
            'gt': img_gt,
            'folder': folder,
            'idx': self.data_info['idx'][index],
            'border': border,
            'lq_path': lq_path,

        }

    def __len__(self):
        return len(self.data_info['gt_path'])


@DATASET_REGISTRY.register()
class VideoTestVimeo90KDataset(data.Dataset):
    """Video test dataset for Vimeo90k-Test dataset."""
    def __init__(self, opt):
        super(VideoTestVimeo90KDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        if self.cache_data:
            raise NotImplementedError('cache_data in Vimeo90K-Test dataset is not implemented.')
        self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.data_info = {'lq_path': [], 'gt_path': [], 'folder': [], 'idx': [], 'border': []}
        neighbor_list = [i + (9 - opt['num_frame']) // 2 for i in range(opt['num_frame'])]

        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        assert self.io_backend_opt['type'] != 'lmdb', 'No need to use lmdb during validation/test.'

        logger = get_root_logger()
        logger.info(f'Generate data info for VideoTestDataset - {opt["name"]}')
        with open(opt['meta_info_file'], 'r') as fin:
            subfolders = [line.split(' ')[0] for line in fin]
        for idx, subfolder in enumerate(subfolders):
            gt_path = osp.join(self.gt_root, subfolder, 'im4.png')
            self.data_info['gt_path'].append(gt_path)
            lq_paths = [osp.join(self.lq_root, subfolder, f'im{i}.png') for i in neighbor_list]
            self.data_info['lq_path'].append(lq_paths)
            self.data_info['folder'].append(subfolder)
            self.data_info['idx'].append(f'{idx}/{len(subfolders)}')
            self.data_info['border'].append(0)

    def __getitem__(self, index):
        lq_path = self.data_info['lq_path'][index]
        gt_path = self.data_info['gt_path'][index]
        imgs_lq = read_img_seq(lq_path)
        img_gt = read_img_seq([gt_path])
        img_gt.squeeze_(0)

        return {
            'lq': imgs_lq,
            'gt': img_gt,
            'folder': self.data_info['folder'][index],
            'idx': self.data_info['idx'][index],
            'border': self.data_info['border'][index],
            'lq_path': lq_path[self.opt['num_frame'] // 2]
        }

    def __len__(self):
        return len(self.data_info['gt_path'])


@DATASET_REGISTRY.register()
class VideoTestDUFDataset(VideoTestDataset):
    """ Video test dataset for DUF dataset. """
    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]
        lq_path = self.data_info['lq_path'][index]

        select_idx = generate_frame_indices(idx, max_idx, self.opt['num_frame'], padding=self.opt['padding'])

        if self.cache_data:
            if self.opt['use_duf_downsampling']:
                imgs_lq = self.imgs_gt[folder].index_select(0, torch.LongTensor(select_idx))
                imgs_lq = duf_downsample(imgs_lq, kernel_size=13, scale=self.opt['scale'])
            else:
                imgs_lq = self.imgs_lq[folder].index_select(0, torch.LongTensor(select_idx))
            img_gt = self.imgs_gt[folder][idx]
        else:
            if self.opt['use_duf_downsampling']:
                img_paths_lq = [self.imgs_gt[folder][i] for i in select_idx]
                imgs_lq = read_img_seq(img_paths_lq, require_mod_crop=True, scale=self.opt['scale'])
                imgs_lq = duf_downsample(imgs_lq, kernel_size=13, scale=self.opt['scale'])
            else:
                img_paths_lq = [self.imgs_lq[folder][i] for i in select_idx]
                imgs_lq = read_img_seq(img_paths_lq)
            img_gt = read_img_seq([self.imgs_gt[folder][idx]], require_mod_crop=True, scale=self.opt['scale'])
            img_gt.squeeze_(0)

        return {
            'lq': imgs_lq,
            'gt': img_gt,
            'folder': folder,
            'idx': self.data_info['idx'][index],
            'border': border,
            'lq_path': lq_path
        }


@DATASET_REGISTRY.register()
class VideoRecurrentTestDataset(VideoTestDataset):
    """按文件夹为单位返回一个序列样本：(T,1,H,W)。"""

    def __init__(self, opt):
        super(VideoRecurrentTestDataset, self).__init__(opt)
        self.folders = sorted(list(set(self.data_info['folder'])))
        self.folder_to_indices = {}
        for i, f in enumerate(self.data_info['folder']):
            self.folder_to_indices.setdefault(f, []).append(i)
        self.folder_to_lqpaths = {}
        self.folder_to_gtpaths = {}
        for f, idcs in self.folder_to_indices.items():
            lq_list = [self.data_info['lq_path'][k] for k in idcs]
            gt_list = [self.data_info['gt_path'][k] for k in idcs]
            for lp, gp in zip(lq_list, gt_list):
                assert osp.basename(lp) == osp.basename(gp), \
                    f"[{f}] LQ/GT 文件名不一致: {lp} vs {gp}"
            self.folder_to_lqpaths[f] = lq_list
            self.folder_to_gtpaths[f] = gt_list

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, index):
        folder = self.folders[index]
        if self.cache_data:
            imgs_lq = self.imgs_lq[folder]
            imgs_gt = self.imgs_gt[folder]
            gt_paths = self.folder_to_gtpaths[folder]
        else:
            lq_paths = self.folder_to_lqpaths[folder]
            gt_paths = self.folder_to_gtpaths[folder]
            imgs_lq = read_img_seq(lq_paths)
            imgs_gt = read_img_seq(gt_paths)

        imgs_lq = imgs_lq.contiguous().float()
        imgs_gt = imgs_gt.contiguous().float()

        # 掩膜：像素值==0 为掩膜外，其余为掩膜内
        # mask_list = []
        # for p in gt_paths:
        #     gt16 = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        #     if gt16.ndim == 3:
        #         gt16 = gt16[..., 0]
        #     m = (gt16 != 0).astype(np.float32)  # !=0 → 掩膜内
        #     mask_list.append(torch.from_numpy(m).unsqueeze(0))
        # masks = torch.stack(mask_list, dim=0).float()
        #
        # assert imgs_lq.shape[1] == 1, f"LQ通道数错误，期望1，实际{imgs_lq.shape[1]}"
        # assert imgs_gt.shape[1] == 1, f"GT通道数错误，期望1，实际{imgs_gt.shape[1]}"
        # assert masks.shape == imgs_gt.shape, f"掩膜与GT尺寸不一致：mask {masks.shape} vs GT {imgs_gt.shape}"

        return {
            'lq': imgs_lq,
            'gt': imgs_gt,
            'folder': folder,
            'idx': f'0/{imgs_gt.size(0)}',
            'border': 0,
            'lq_path': self.folder_to_lqpaths[folder][len(self.folder_to_lqpaths[folder]) // 2]
        }
