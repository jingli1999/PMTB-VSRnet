import torch
from collections import Counter
from os import path as osp
from torch import distributed as dist
from tqdm import tqdm
import os

from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.registry import MODEL_REGISTRY
from .video_base_model import VideoBaseModel
import numpy as np
import cv2
import math
import torch.nn.functional as F
from torch.cuda.amp import autocast
from basicsr.utils.registry import METRIC_REGISTRY
from basicsr.archs.arch_util import flow_warp  # 确保你工程里有这个工具；若路径不同，按工程实际修改
from torch.nn.utils import clip_grad_norm_


# 1) 定义：把 to_u16 改成带 eps
def to_u16(x: torch.Tensor, eps: float) -> np.ndarray:
    x = x.detach().cpu().clamp(0, 1)
    if x.dim() == 4: x = x[0, 0]
    elif x.dim() == 3: x = x[0]
    arr = x.numpy().astype(np.float32)
    arr = np.where(arr < eps, 0.0, arr)  # <--- 用同一个 eps 把极小值清零
    return np.rint(arr * 65535.0).astype(np.uint16)

@MODEL_REGISTRY.register()
class VideoRecurrentModel(VideoBaseModel):

    def __init__(self, opt):
        super(VideoRecurrentModel, self).__init__(opt)
        self.opt = opt
        self.is_train = opt.get('is_train', False)
        self.opt_train = opt.get('train', {}) if self.is_train else {}

        t = self.opt_train
        # ---------------- Loss weights ----------------
        self.lambda_pix = t.get('lambda_pix', 1.0)
        self.lambda_hf = t.get('lambda_hf', 0.0)

        # ---------------- Sobel kernels (buffer) ----------------
        with torch.no_grad():
            self.sobel_x = torch.tensor([[-1., 0., 1.],
                                         [-2., 0., 2.],
                                         [-1., 0., 1.]]).view(1, 1, 3, 3)
            self.sobel_y = torch.tensor([[-1., -2., -1.],
                                         [0., 0., 0.],
                                         [1., 2., 1.]]).view(1, 1, 3, 3)

        # ---------------- Gaussian kernel for HF suppression ----------------
        def _make_gaussian_kernel(ks=5, sigma=1.0):
            ax = torch.arange(ks) - ks // 2
            xx, yy = torch.meshgrid(ax, ax, indexing='ij')
            kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
            kernel = kernel / kernel.sum()
            return kernel

        ks = int(self.opt.get('hf_gauss_ks', 5))
        sigma = float(self.opt.get('hf_gauss_sigma', 1.0))
        gauss = _make_gaussian_kernel(ks, sigma)
        # self.register_buffer('gauss_kernel', gauss.view(1, 1, ks, ks))
        self.gauss_kernel = gauss.view(1, 1, ks, ks)

        # ---------------- Gaussian kernel for LF consistency ----------------
        ks_lf = int(self.opt.get('lf_gauss_ks', 11))
        sigma_lf = float(self.opt.get('lf_gauss_sigma', 2.0))
        gauss_lf = _make_gaussian_kernel(ks_lf, sigma_lf)
        self.gauss_kernel_lf = gauss_lf.view(1, 1, ks_lf, ks_lf)

        get_root_logger().info(
            f'Loss weights | pix:{self.lambda_pix}, '
            f'hf:{self.lambda_hf}'
        )

    # ---------------------------------------------------------
    # Sobel gradient magnitude
    # ---------------------------------------------------------
    def _sobel_grad(self, x: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        orig_dim = x.dim()
        if orig_dim == 5:
            B, T, C, H, W = x.shape
            x4 = x.reshape(B * T, C, H, W)
        elif orig_dim == 4:
            x4 = x
        else:
            raise ValueError(f'_sobel_grad expects 4D/5D, got {x.shape}')

        # kx = self.sobel_x.to(dtype=x4.dtype)
        # ky = self.sobel_y.to(dtype=x4.dtype)
        kx = self.sobel_x.to(device=x4.device, dtype=x4.dtype)
        ky = self.sobel_y.to(device=x4.device, dtype=x4.dtype)

        Cx = x4.shape[1]
        kx = kx.expand(Cx, 1, 3, 3)
        ky = ky.expand(Cx, 1, 3, 3)

        gx = F.conv2d(x4, kx, padding=1, groups=Cx)
        gy = F.conv2d(x4, ky, padding=1, groups=Cx)
        grad = torch.sqrt(gx * gx + gy * gy + 1e-12)

        if orig_dim == 5:
            grad = grad.reshape(B, T, C, H, W)
        return grad

    # ---------------------------------------------------------
    # Optimizer (freeze SpyNet only if exists)
    # ---------------------------------------------------------
    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_opt = train_opt['optim_g'].copy()
        optim_type = optim_opt.pop('type')
        lr = optim_opt.pop('lr')

        normal_params = []
        frozen = []

        for name, p in self.net_g.named_parameters():
            if 'spynet' in name:
                p.requires_grad_(False)
                frozen.append(name)
            else:
                normal_params.append(p)

        self.optimizer_g = self.get_optimizer(
            optim_type, normal_params, lr=lr, **optim_opt
        )
        self.optimizers = [self.optimizer_g]

        lg = get_root_logger()
        lg.info(f'Frozen params: {len(frozen)}')
        if frozen:
            lg.info('Examples: ' + '; '.join(frozen[:10]))

    # ---------------------------------------------------------
    # Training step
    # ---------------------------------------------------------
    def optimize_parameters(self, current_iter):
        import torch.nn.functional as F

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)  # (N,T,1,H,W)

        loss_total = 0.0

        # ---------- Pixel loss ----------
        l_pix = torch.tensor(0.0, device=self.device)
        if self.cri_pix is not None:
            l_pix = self.cri_pix(self.output, self.gt)
            loss_total += self.lambda_pix * l_pix

        # ---------- Gradient difference loss (stable version) ----------
        # L_gdr = torch.tensor(0.0, device=self.device)
        # if self.lambda_grad > 0:
        #     grad_sr = self._sobel_grad(self.output)
        #     grad_gt = self._sobel_grad(self.gt).detach()
        #     L_gdr = torch.mean(torch.abs(grad_sr - grad_gt))
        #     loss_total += self.lambda_grad * L_gdr

        # ---------- GT-aware Band-limited HF suppression ----------
        L_hfsup = torch.tensor(0.0, device=self.device)
        if self.lambda_hf > 0:
            B, T, C, H, W = self.output.shape
            sr = self.output.view(B * T, C, H, W)
            gt = self.gt.view(B * T, C, H, W)

            ks = self.gauss_kernel.shape[-1]
            kernel = self.gauss_kernel.to(sr.device, dtype=sr.dtype).expand(C, 1, ks, ks)

            low_sr = F.conv2d(sr, kernel, padding=ks // 2, groups=C)
            low_gt = F.conv2d(gt, kernel, padding=ks // 2, groups=C)

            hf_sr = sr - low_sr
            hf_gt = gt - low_gt

            # 只惩罚“超过 GT 的高频能量”
            L_hfsup = torch.mean(F.relu(torch.abs(hf_sr) - torch.abs(hf_gt)))
            loss_total += self.lambda_hf * L_hfsup

        # ---------- Low-frequency consistency loss ----------
        # L_lf = torch.tensor(0.0, device=self.device)
        # if self.lambda_lf > 0:
        #     B, T, C, H, W = self.output.shape
        #     sr = self.output.view(B * T, C, H, W)
        #     gt = self.gt.view(B * T, C, H, W)
        #
        #     ks = self.gauss_kernel_lf.shape[-1]
        #     kernel = self.gauss_kernel_lf.to(sr.device, dtype=sr.dtype).expand(C, 1, ks, ks)
        #
        #     low_sr = F.conv2d(sr, kernel, padding=ks // 2, groups=C)
        #     low_gt = F.conv2d(gt, kernel, padding=ks // 2, groups=C)
        #
        #     L_lf = torch.mean(torch.abs(low_sr - low_gt))
        #
        #     loss_total += self.lambda_lf * L_lf

        # ---------- Backward ----------
        loss_total.backward()
        clip_grad_norm_(self.net_g.parameters(), 1.0)
        self.optimizer_g.step()

        # ---------- Log ----------
        self.log_dict = {
            'l_total': float(loss_total.item()),
            'l_pix': float(l_pix.item()),
            'l_hfsup': float(L_hfsup.item()),
        }

    # 在文件：basicsr/models/video_recurrent_model.py 内替换
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset = dataloader.dataset
        dataset_name = dataset.opt['name']

        # with_metrics = self.opt['val']['metrics'] is not None
        # if with_metrics:
        #     if not hasattr(self, 'metric_results'):
        #         self.metric_results = {}
        #         num_frame_each_folder = Counter(dataset.data_info['folder'])
        #         for folder, num_frame in num_frame_each_folder.items():
        #             self.metric_results[folder] = torch.zeros(
        #                 num_frame, len(self.opt['val']['metrics']),
        #                 dtype=torch.float32, device='cuda')
        #     self._initialize_best_metric_results(dataset_name)
        #
        # rank, world_size = get_dist_info()
        # if with_metrics:
        #     for _, tensor in self.metric_results.items():
        #         tensor.zero_()
        #
        # num_folders = len(dataset)
        # num_pad = (world_size - (num_folders % world_size)) % world_size
        # if rank == 0:
        #     pbar = tqdm(total=len(dataset), unit='folder')
        #
        # self.center_frame_only = getattr(
        #     self, 'center_frame_only',
        #     self.opt.get('val', {}).get('center_frame_only', True)
        # )
        # with_metrics = self.opt['val']['metrics'] is not None
        with_metrics = bool(self.opt.get('val', {}).get('metrics'))

        # 先确定是否只评中心帧（放到初始化 metric_results 之前）
        self.center_frame_only = bool(self.opt.get('val', {}).get('center_frame_only', False))

        if with_metrics:
            # 每次验证都重新初始化，避免沿用上一次的形状
            self.metric_results = {}
            num_frame_each_folder = Counter(dataset.data_info['folder'])

            # 关键修正：只评中心帧时，每个序列就统计 1 帧
            if self.center_frame_only:
                for k in num_frame_each_folder:
                    num_frame_each_folder[k] = 1

            for folder, n in num_frame_each_folder.items():
                self.metric_results[folder] = torch.zeros(
                    n, len(self.opt['val']['metrics']),
                    dtype=torch.float32, device='cuda')
            self._initialize_best_metric_results(dataset_name)

        rank, world_size = get_dist_info()
        num_folders = len(dataset)
        num_pad = (world_size - (num_folders % world_size)) % world_size
        pbar = None

        if rank == 0:
            pbar = tqdm(total=len(dataset), unit='folder')

        # def frame_to_u16_hw(x, t_idx):
        #     if isinstance(x, torch.Tensor):
        #         x = x.detach().cpu().clamp(0, 1)
        #         if x.dim() == 6:
        #             t = min(max(t_idx, 0), x.size(2) - 1)
        #             x = x[0, 0, t]
        #         elif x.dim() == 5:
        #             t = min(max(t_idx, 0), x.size(1) - 1)
        #             x = x[0, t]
        #         elif x.dim() == 4:
        #             x = x[0]
        #         elif x.dim() == 3:
        #             pass
        #         elif x.dim() == 2:
        #             pass
        #         else:
        #             raise ValueError(f'Unexpected tensor shape: {tuple(x.shape)}')
        #         if x.dim() == 3 and x.size(0) == 1:
        #             x = x[0]
        #         # arr = x.numpy()
        #         # eps = 0.5 / 65535.0
        #         # x = np.where(x < eps, 0.0, x)
        #         # return np.rint(x.numpy() * 65535.0).astype(np.uint16)
        #         # x 是 [0,1] 的 torch.Tensor
        #         arr = x.numpy()
        #         eps = 0.5 / 65535.0
        #         arr[arr < eps] = 0.0
        #         return np.rint(arr * 65535.0).astype(np.uint16)
        #     arr = np.asarray(x)
        #     while arr.ndim > 2 and arr.shape[0] == 1:
        #         arr = arr[0]
        #     return arr.astype(np.uint16)
        # def frame_to_u16_hw(x, t_idx):
        #     if isinstance(x, torch.Tensor):
        #         x = x.detach().cpu().clamp(0, 1)
        #         if x.dim() == 6:
        #             t = min(max(t_idx, 0), x.size(2) - 1);
        #             x = x[0, 0, t]
        #         elif x.dim() == 5:
        #             t = min(max(t_idx, 0), x.size(1) - 1);
        #             x = x[0, t]
        #         elif x.dim() == 4:
        #             x = x[0]
        #         if x.dim() == 3 and x.size(0) == 1:
        #             x = x[0]
        #         # ---- 关键：先转为 numpy，再做极小值置零 ----
        #         # x = x.numpy()
        #         x = x.numpy().astype(np.float32)
        #         eps = 0.5 / 65535.0
        #         x = np.where(x < eps, 0.0, x)
        #         return np.rint(x * 65535.0).astype(np.uint16)
        #     arr = np.asarray(x)
        #     while arr.ndim > 2 and arr.shape[0] == 1:
        #         arr = arr[0]
        #     return arr.astype(np.uint16)

        def frame_to_u16_hw(x, t_idx: int):
            """
            统一把各种形状的张量/数组转成 HxW 的 uint16。
            支持:
              - torch.Tensor: (N,T,C,H,W)/(T,C,H,W)/(N,C,H,W)/(C,H,W)/(H,W)
              - np.ndarray  : (H,W,C)/(C,H,W)/(H,W)
            规则:
              - 若有时间维，按 t_idx 取该帧
              - 若是多通道，取通道均值（如需改成取第0通道，改最后那一段）
            """

            def _to_numpy_01(t: torch.Tensor) -> np.ndarray:
                return t.detach().float().cpu().clamp(0, 1).numpy()

            # 1) 先拿到“单 batch、单帧”
            if isinstance(x, torch.Tensor):
                t = x
                # 去 batch 维 (N,...) -> (...)
                if t.dim() >= 1 and t.size(0) == 1:
                    t = t[0]
                # 现在常见两种：5D(极少见) 或 4D(T,C,H,W) —— 一定要在 4D 也按 t_idx 取帧
                if t.dim() == 5:  # (T, C, H, W) 之前还有个多余 batch
                    t = t[t_idx]
                elif t.dim() == 4:  # (T, C, H, W)
                    t = t[t_idx]
                # 其余：3D(C,H,W) / 2D(H,W) 都不用再取帧
                arr = _to_numpy_01(t)
            else:
                arr = np.asarray(x)

            # 2) 统一到 HxW 或 HxWxC
            # 常见 3D 情况：CHW 或 HWC
            if arr.ndim == 3:
                # 若是 CHW（C 在第0维，且 C∈{1,2,3,4}），转成 HWC
                if arr.shape[0] in (1, 2, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))  # CHW->HWC

                # 灰度化：>1通道则做均值；1通道则直接去掉通道维
                if arr.shape[2] == 1:
                    arr = arr[..., 0]
                else:
                    arr = arr.mean(axis=2)

            # 3) [0,1] → uint16
            # 修改后：在返回之前新增下面两行
            eps = float(self.opt.get('val', {}).get('zero_eps', 2.0 / 65535.0))
            arr = np.where(arr < eps, 0.0, arr)
            return np.rint(arr * 65535.0).astype(np.uint16)

        for i in range(rank, num_folders + num_pad, world_size):
            idx_ds = min(i, num_folders - 1)
            val_data = dataset[idx_ds]
            folder = val_data['folder']

            val_data['lq'].unsqueeze_(0)
            if 'gt' in val_data and val_data['gt'] is not None:
                val_data['gt'].unsqueeze_(0)

            self.feed_data(val_data)

            val_data['lq'].squeeze_(0)
            if 'gt' in val_data and val_data['gt'] is not None:
                val_data['gt'].squeeze_(0)

            self.test()
            visuals = self.get_current_visuals()

            if 'result' in visuals:
                res = visuals['result']
                if res.dim() == 4:
                    res = res.unsqueeze(1)
                visuals['result'] = res

            if 'gt' in visuals:
                gt = visuals['gt']
                if gt.dim() == 5:
                    if self.center_frame_only:
                        mid = gt.size(1) // 2
                        gt = gt[:, mid:mid + 1, ...]
                elif gt.dim() == 4:
                    gt = gt.unsqueeze(1)
                else:
                    raise ValueError(f'Unexpected GT shape: {tuple(gt.shape)}')
                visuals['gt'] = gt

            del self.lq
            del self.output
            if 'gt' in visuals:
                del self.gt
            torch.cuda.empty_cache()

            # if i < num_folders:
            #     T = visuals['result'].size(1)
            #     for t in range(T):
            #         sr_u16 = frame_to_u16_hw(visuals['result'], t)
            #         gt_u16 = frame_to_u16_hw(visuals['gt'], t) if 'gt' in visuals else None
            #
            #         if gt_u16 is not None:
            #             assert sr_u16.shape == gt_u16.shape, \
            #                 f"shape mismatch: SR {sr_u16.shape} vs GT {gt_u16.shape}"
            #
            #         if save_img:
            #             img_dir = osp.join(self.opt['path']['visualization'], dataset_name, folder)
            #             os.makedirs(img_dir, exist_ok=True)
            #             # cv2.imwrite(osp.join(img_dir, f"im4.png"), sr_u16)
            #             cv2.imwrite(osp.join(img_dir, f"im_{t:02d}.png"), sr_u16)
            #
            #         if with_metrics and gt_u16 is not None:
            #             metric_data = {'img': sr_u16, 'img2': gt_u16}
            #             for metric_idx, opt_ in enumerate(self.opt['val']['metrics'].values()):
            #                 result = calculate_metric(metric_data, dict(opt_))
            #                 self.metric_results[folder][t, metric_idx] += result
            if i < num_folders:
                res = visuals['result']  # (N,T,1,Hr,Wr) 或 (N,1,Hr,Wr)
                if res.dim() == 4:  # 统一补时间维
                    res = res.unsqueeze(1)  # -> (N,1,1,Hr,Wr)
                T = res.size(1)

                img_dir = osp.join(self.opt['path']['visualization'], dataset_name, folder)
                os.makedirs(img_dir, exist_ok=True)

                # 取 LQ 序列（优先 visuals，其次 val_data）
                lq_seq = visuals.get('lq', None)
                if lq_seq is None:
                    lq_seq = val_data['lq']  # (T,1,Hl,Wl) 或 (1,T,1,Hl,Wl)
                if isinstance(lq_seq, torch.Tensor) and lq_seq.dim() == 4:
                    lq_seq = lq_seq.unsqueeze(0)  # -> (1,T,1,Hl,Wl)

                # ---- 自动推断放大倍数 & 阈值 ----
                scale = res.size(-1) // lq_seq.size(-1)
                # eps = float(self.opt.get('val', {}).get('zero_eps', 0.5 / 65535.0))

                for t in range(T):
                    # --- SR 第 t 帧 (u16) ---
                    eps = float(self.opt.get('val', {}).get('zero_eps', 2.0 / 65535.0))
                    sr_u16 = to_u16(res[:, t, ...],eps)

                    # --- LR 第 t 帧 最近邻 ×scale (u16) + 零掩膜 ---
                    lr_t = lq_seq[:, t, :1, :, :]  # (1,1,Hl,Wl)
                    lr_up_nn = F.interpolate(lr_t, scale_factor=scale, mode='nearest')
                    lr_up_u16 = to_u16(lr_up_nn,eps)
                    zero_mask = (lr_up_nn <= eps).squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8) * 255

                    # --- 漏零图：SR!=0 且 LR_up==0 ---
                    leak_map = ((sr_u16 != 0) & (lr_up_u16 == 0)).astype(np.uint8) * 255

                    # --- 保存 ---
                    cv2.imwrite(osp.join(img_dir, f"sr_{t:03d}.png"), sr_u16)
                    # 新增：可选是否输出其它检查类图片
                    if bool(self.opt.get('val', {}).get('dump_debug_vis', False)):
                        cv2.imwrite(osp.join(img_dir, f"lq_up_{t:03d}.png"), lr_up_u16)
                        cv2.imwrite(osp.join(img_dir, f"lr_zero_mask_{t:03d}.png"), zero_mask)
                        cv2.imwrite(osp.join(img_dir, f"sr_leak_map_{t:03d}.png"), leak_map)

                    # --- 度量：用 16-bit（可选逐帧或只中心帧）
                    # 修改前：
                    # if with_metrics and 'gt' in visuals:
                    #     sr8 = tensor2img([res[0, t]])  # uint8
                    #     gt8 = tensor2img([visuals['gt'][0, t]])
                    #     metric_data = {'img': sr8, 'img2': gt8}
                    #     for metric_idx, opt_ in enumerate(self.opt['val']['metrics'].values()):
                    #         self.metric_results[folder][t, metric_idx] += calculate_metric(metric_data, dict(opt_))
                    #     修改完：
                    # --- 度量：16-bit（H, W）
                    if with_metrics and 'gt' in visuals:
                        gt_u16 = frame_to_u16_hw(visuals['gt'], t)  # HxW, uint16
                        metric_data = {'img': sr_u16, 'img2': gt_u16}
                        for metric_idx, opt_ in enumerate(self.opt['val']['metrics'].values()):
                            self.metric_results[folder][t, metric_idx] += calculate_metric(metric_data, dict(opt_))

                if rank == 0:
                    for _ in range(world_size):
                        pbar.update(1)
                        pbar.set_description(f'Folder: {folder}')

        if rank == 0:
            pbar.close()

        if with_metrics:
            if self.opt['dist']:
                for _, tensor in self.metric_results.items():
                    torch.distributed.reduce(tensor, 0)
                torch.distributed.barrier()
            if rank == 0:
                self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

        # print("SR_u16 min/max:", int(sr_u16.min()), int(sr_u16.max()))
        # print("GT_u16 min/max:", int(gt_u16.min()), int(gt_u16.max()))
        # print("背景像素统计: SR 非零且 GT 为 0 的像素数 =",
        #               int(((sr_u16 != 0) & (gt_u16 == 0)).sum()))

                # 修改前：
    # def test(self):
    #     n = self.lq.size(1)
    #     self.net_g.eval()
    #
    #     flip_seq = self.opt['val'].get('flip_seq', False)
    #     # 修改前：(输出所有帧的超分结果)
    #     # self.center_frame_only = self.opt['val'].get('center_frame_only', False)
    #     # 修改后：（只输出中心帧的超分结果）
    #     self.center_frame_only = self.opt['val'].get('center_frame_only', True)
    #
    #     if flip_seq:
    #         self.lq = torch.cat([self.lq, self.lq.flip(1)], dim=1)
    #
    #     with torch.no_grad():
    #         # 修改前：
    #         # self.output = self.net_g(self.lq)
    #         self.output = self.net_g(self.lq, mask=getattr(self, 'mask_lq', None))
    #
    #     with torch.no_grad():
    #         n = self.lq.size(1)
    #         # 取中心帧的 LR 零像素图 -> 最近邻上采样到 HR
    #         lr_center = self.lq[:, n // 2, :1, :, :]  # (N,1,Hl,Wl)
    #         zero_map_hr = (F.interpolate(lr_center, scale_factor=4, mode='nearest') == 0)  # bool, (N,1,Hh,Wh)
    #         # 把这些位置在超分输出中强制为 0
    #         self.output = self.output.masked_fill(zero_map_hr, 0.0)
    #
    #     if flip_seq:
    #         output_1 = self.output[:, :n, :, :, :]
    #         output_2 = self.output[:, n:, :, :, :].flip(1)
    #         self.output = 0.5 * (output_1 + output_2)
    #
    #     if self.center_frame_only:
    #         self.output = self.output[:, n // 2, :, :, :]
    #
    #     self.net_g.train()

    # def test(self):
    #     """无掩膜版本：只做前向；可选只取中心帧。"""
    #     self.net_g.eval()
    #     with torch.no_grad():  # 如需 AMP，把这一行改成：with torch.no_grad(), autocast():
    #         out = self.net_g(self.lq)  # (N,T,1,Hr,Wr) 或 (N,1,Hr,Wr)
    #
    #     # 只评中心帧（按配置）
    #     if bool(self.opt.get('val', {}).get('center_frame_only', True)) and out.dim() == 5:
    #         out = out[:, out.size(1) // 2]  # -> (N,1,Hr,Wr)
    #
    #     # 回 CPU；不做任何 masked_fill/置零
    #     if hasattr(out, "is_cuda") and out.is_cuda:
    #         out = out.float().cpu()
    #         torch.cuda.empty_cache()
    #
    #     # --- 额外保险：在 test() 里统一做一次保零 ---
    #     if out.dim() == 5:  # (N,T,1,Hr,Wr)
    #         n = self.lq.size(1)
    #         lr_center = self.lq[:, n // 2, :1, :, :]
    #         zero_map_hr = (F.interpolate(lr_center, scale_factor=4, mode='nearest') <= 0)
    #         out = out.masked_fill(zero_map_hr.unsqueeze(1), 0.0)  # 对所有 T 帧
    #     elif out.dim() == 4:  # (N,1,Hr,Wr)，只评中心帧时
    #         n = self.lq.size(1)
    #         lr_center = self.lq[:, n // 2, :1, :, :]
    #         zero_map_hr = (F.interpolate(lr_center, scale_factor=4, mode='nearest') <= 0)
    #         out = out.masked_fill(zero_map_hr, 0.0)
    #
    #     self.output = out
    #     self.net_g.train()
    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            out = self.net_g(self.lq)  # (N,T,1,Hr,Wr) or (N,1,Hr,Wr)

        scale = out.size(-1) // self.lq.size(-1)
        eps = float(self.opt.get('val', {}).get('zero_eps', 2.0 / 65535.0))

        if out.dim() == 5:  # (N,T,1,H,W)
            lr_zero_seq = (self.lq[:, :, :1, :, :] <= eps).float()
            hr_zero_seq = F.interpolate(lr_zero_seq, scale_factor=scale, mode='nearest') > 0.5
            out = out.masked_fill(hr_zero_seq, 0.0)
        elif out.dim() == 4:  # 只中心帧
            n = self.lq.size(1)
            lr_zero = (self.lq[:, n // 2, :1, :, :] <= eps).float()
            hr_zero = F.interpolate(lr_zero, scale_factor=scale, mode='nearest') > 0.5
            out = out.masked_fill(hr_zero, 0.0)

        self.output = out.float().cpu()
        self.net_g.train()






