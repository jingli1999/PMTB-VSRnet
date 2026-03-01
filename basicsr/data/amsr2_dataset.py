import numpy as np
import random
import torch
from pathlib import Path
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class AMSR2Dataset(data.Dataset):
    """
    AMSR2 单通道灰度图数据集，支持光流可选，结构类似REDS。
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        self.flow_root = Path(opt.get('dataroot_flow')) if opt.get('dataroot_flow') else None
        assert opt['num_frame'] % 2 == 1, 'num_frame应为奇数'

        self.num_frame = opt['num_frame']
        self.num_half_frames = self.num_frame // 2

        self.keys = []
        with open(opt['meta_info_file'], 'r') as f:
            for line in f:
                folder, frame_num, _ = line.split(' ')
                self.keys.extend([f'{folder}/{i:08d}' for i in range(int(frame_num))])

        if opt['val_partition'] not in ['official', 'AMSR2']:
            raise ValueError(f"val_partition 必须是 'official' 或 'AMSR2'")

        val_partition = ['00000', '03995', '03996', '03997'] if opt['val_partition'] == 'AMSR2' else [f'{v:08d}' for v in range(240, 270)]
        self.keys = [v for v in self.keys if v.split('/')[0] not in val_partition]

        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        interval_str = ','.join(str(x) for x in opt['interval_list'])
        logger = get_root_logger()
        logger.info(f'AMSR2 Dataset加载完成，时间间隔: [{interval_str}]，随机反转: {opt["random_reverse"]}')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale, gt_size = self.opt['scale'], self.opt['gt_size']
        clip_name, frame_idx = self.keys[index].split('/')

        center_idx = int(frame_idx)
        interval = random.choice(self.opt['interval_list'])

        start_idx = center_idx - self.num_half_frames * interval
        end_idx = center_idx + self.num_half_frames * interval

        frame_list = list(range(start_idx, end_idx + 1, interval))

        # 防止越界
        total_frames = len(list((self.lq_root / clip_name).glob('*.png')))
        while start_idx < 0 or end_idx >= total_frames:
            center_idx = random.randint(0, total_frames - 1)
            start_idx = center_idx - self.num_half_frames * interval
            end_idx = center_idx + self.num_half_frames * interval
            frame_list = list(range(start_idx, end_idx + 1, interval))

        if self.opt['random_reverse'] and random.random() < 0.5:
            frame_list.reverse()

        assert len(frame_list) == self.num_frame

        img_lqs, img_gts = [], []
        for idx in frame_list:
            lq_path = self.lq_root / clip_name / f'{idx:08d}.png'
            gt_path = self.gt_root / clip_name / f'{idx:08d}.png'
            # 修改前：
            img_lq = imfrombytes(self.file_client.get(lq_path, 'lq'), flag='unchanged').astype(np.float32)
            img_gt = imfrombytes(self.file_client.get(gt_path, 'gt'), flag='unchanged').astype(np.float32)

            if img_lq.ndim == 2:
                img_lq = img_lq[..., None]
            if img_gt.ndim == 2:
                img_gt = img_gt[..., None]
            img_lq = img_lq.astype(np.float32) / 65535.
            img_lqs.append(img_lq)
            img_gts.append(img_gt)

        img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size, scale, gt_path)
        img_lqs.extend(img_gts)
        img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])
        img_results = img2tensor(img_results, bgr2rgb=False, float32=True)

        img_lqs = torch.stack(img_results[:self.num_frame], dim=0)
        img_gts = torch.stack(img_results[self.num_frame:], dim=0)

        return {'lq': img_lqs, 'gt': img_gts, 'key': f'{clip_name}/{frame_idx}'}

    def __len__(self):
        return len(self.keys)


@DATASET_REGISTRY.register()
class AMSR2RecurrentDataset(data.Dataset):
    """
    AMSR2 数据集，支持递归网络，clip整体输出，动态帧数。
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        self.num_frame = opt['num_frame']

        self.keys = []
        with open(opt['meta_info_file'], 'r') as f:
            for line in f:
                folder, frame_num, _ = line.split(' ')
                self.keys.extend([f'{folder}/{i:08d}' for i in range(int(frame_num))])

        if opt['val_partition'] not in ['official', 'AMSR2']:
            raise ValueError(f"val_partition 必须是 'official' 或 'AMSR2'")

        val_partition = ['00000', '03995', '03996', '03997'] if opt['val_partition'] == 'AMSR2' else [f'{v:08d}' for v in range(240, 270)]
        if opt['test_mode']:
            self.keys = [v for v in self.keys if v.split('/')[0] in val_partition]
        else:
            self.keys = [v for v in self.keys if v.split('/')[0] not in val_partition]

        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        interval_str = ','.join(str(x) for x in opt.get('interval_list', [1]))
        logger = get_root_logger()
        logger.info(f'AMSR2 Recurrent Dataset加载完成，时间间隔: [{interval_str}]，随机反转: {opt.get("random_reverse", False)}')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale, gt_size = self.opt['scale'], self.opt['gt_size']
        clip_name, frame_idx = self.keys[index].split('/')

        start_idx = int(frame_idx)
        total_frames = len(list((self.lq_root / clip_name).glob('*.png')))
        max_start = total_frames - self.num_frame
        if max_start <= 0:
            raise ValueError(f'Clip {clip_name} 不足 {self.num_frame} 帧')

        if start_idx > max_start:
            start_idx = random.randint(0, max_start)
        neighbor_list = list(range(start_idx, start_idx + self.num_frame))

        if self.opt.get('random_reverse', False) and random.random() < 0.5:
            neighbor_list.reverse()

        img_lqs, img_gts = [], []
        for idx in neighbor_list:
            lq_path = self.lq_root / clip_name / f'{idx:08d}.png'
            gt_path = self.gt_root / clip_name / f'{idx:08d}.png'
            # 修改前：
            # img_lq = imfrombytes(self.file_client.get(lq_path, 'lq'), flag='unchanged').astype(np.float32)
            # img_gt = imfrombytes(self.file_client.get(gt_path, 'gt'), flag='unchanged').astype(np.float32)
            # if img_lq.ndim == 2:
            #     img_lq = img_lq[..., None]
            # if img_gt.ndim == 2:
            #     img_gt = img_gt[..., None]
            # 修改后：
            # ---------- 读取并处理 LQ ----------
            img_lq = imfrombytes(self.file_client.get(lq_path, 'lq'), flag='unchanged').astype(np.float32)
            if img_lq.ndim == 2:
                img_lq = img_lq[..., None]
            img_lq = img_lq / 65535.  # 归一化到 [0, 1]

            # ---------- 读取并处理 GT ----------
            img_gt = imfrombytes(self.file_client.get(gt_path, 'gt'), flag='unchanged').astype(np.float32)
            if img_gt.ndim == 2:
                img_gt = img_gt[..., None]
            img_gt = img_gt / 65535.
            img_lqs.append(img_lq)
            img_gts.append(img_gt)

        img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size, scale, gt_path)
        img_lqs.extend(img_gts)
        img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])
        img_results = img2tensor(img_results, bgr2rgb=False, float32=True)

        img_lqs = torch.stack(img_results[:self.num_frame], dim=0)
        img_gts = torch.stack(img_results[self.num_frame:], dim=0)

        return {'lq': img_lqs, 'gt': img_gts, 'key': f'{clip_name}/{frame_idx}'}

    def __len__(self):
        return len(self.keys)
