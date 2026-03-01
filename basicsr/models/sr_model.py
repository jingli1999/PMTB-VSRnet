import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import torch.nn.functional as F
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
import numpy as np
import cv2


@MODEL_REGISTRY.register()
class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    # 修改前：
    # def feed_data(self, data):
    #     """把数据（含 mask）喂给模型；并在送入网络前对 LQ 做掩膜门控。
    #        - LR 门控用 LR 尺度的 mask（由 HR mask 下采样得到，nearest）
    #        - HR mask 保留给指标与可视化
    #     """
    #
    #     # LQ / GT 上到设备
    #     self.lq = data['lq'].to(self.device)  # (N,T,C,Hlr,Wlr) 或 (T,C,Hlr,Wlr)
    #     self.gt = data.get('gt', None)
    #     if self.gt is not None:
    #         self.gt = self.gt.to(self.device)
    #
    #     # 取进原始（HR）mask：
    #     # ⭐ 统一口径：像素值==0 为掩膜外；非0 为掩膜内
    #     mask_hr = data.get('mask', None)
    #     if mask_hr is None:
    #         self.mask_hr = None
    #         self.mask_lq = None
    #         return
    #
    #     # 将任意数值 mask 归一为 {0,1}：非0->1，0->0（0 为掩膜外）
    #     self.mask_hr = (mask_hr != 0).to(self.device).float()
    #     #修改后，增加下面一行
    #     self.inside_mask = self.mask_hr  # ★ 训练时的 HR 掩膜给像素损失使用
    #
    #     # ---- 统一维度：确保都是 (N,T,1,H,W) ----
    #     if self.lq.dim() == 4:  # (T,C,H,W) -> (1,T,C,H,W)
    #         self.lq = self.lq.unsqueeze(0)
    #     N, T, C, Hlr, Wlr = self.lq.shape
    #
    #     m = self.mask_hr
    #     if m.dim() == 4:  # (T,1,Hhr,Whr) -> (1,T,1,Hhr,Whr)
    #         m = m.unsqueeze(0)
    #     if m.size(0) != N:
    #         m = m.expand(N, -1, -1, -1, -1)
    #     if m.size(2) != 1:
    #         m = m[:, :, :1, ...]  # 强制单通道
    #
    #     # ---- 得到 LR mask（和 LQ 尺度完全一致）----
    #     NT, Hhr, Whr = N * T, m.size(-2), m.size(-1)
    #     m2d = m.view(NT, 1, Hhr, Whr)
    #     m2d_lq = F.interpolate(m2d, size=(Hlr, Wlr), mode='nearest')  # 最近邻
    #     self.mask_lq = m2d_lq.view(N, T, 1, Hlr, Wlr)
    #
    #     #修改前：
    #     # ---- 门控 LQ：阻断掩膜外对卷积的污染（掩膜外=0）----
    #     outside_fill_input = float(self.opt.get('val', {}).get('outside_fill_input', 0.0))
    #     if outside_fill_input == 0.0:
    #         self.lq = self.lq * self.mask_lq
    #     else:
    #         self.lq = self.lq * self.mask_lq + (1.0 - self.mask_lq) * outside_fill_input
    def feed_data(self, data, **kwargs):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        # 修改前：
        self.output = self.net_g(self.lq)
        # self.output = self.net_g(self.lq, mask=getattr(self, 'mask_lq', None))

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        # 修改前：
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # 修改后：
        # ===== 像素损失：掩膜加权（默认仅中心帧） =====
        # ====== 掩膜加权像素损失（默认仅中心帧；并对掩膜做 r 像素腐蚀） ======
        # if self.cri_pix:
        #     out = self.output
        #     gt = self.gt
        #     w = getattr(self, 'inside_mask', None)  # (N,T,1,H,W) 或 (N,1,H,W)

            # # 仅中心帧计损失（建议，和推理策略一致）
            # if out.dim() == 5 and gt.dim() == 5:
            #     mid = out.size(1) // 2
            #     out = out[:, mid:mid + 1, ...]
            #     gt = gt[:, mid:mid + 1, ...]
            #     if w is not None and w.dim() == 5:
            #         w = w[:, mid:mid + 1, ...]  # (N,1,1,H,W) 或 (N,1,H,W)

            # 对 HR 掩膜做 r 像素腐蚀，去掉边界一圈，缓解 halo
            # def erode_mask_bin(w_bin: torch.Tensor, r: int) -> torch.Tensor:
            #     if (w_bin is None) or (r <= 0):
            #         return w_bin
            #     if w_bin.dim() == 4:  # (N,1,H,W)
            #         w4 = w_bin
            #     elif w_bin.dim() == 5:  # (N,1,1,H,W) 兼容
            #         w4 = w_bin[:, 0, ...]
            #     else:
            #         return w_bin
            #     # 形态学腐蚀（min-pool = 1 - max-pool(1 - x)）
            #     k = 2 * r + 1
            #     w4e = 1.0 - torch.nn.functional.max_pool2d(1.0 - w4, kernel_size=k, stride=1, padding=r)
            #     if w_bin.dim() == 5:
            #         w4e = w4e.unsqueeze(1)
            #     return w4e
            #
            # # 腐蚀半径（HR 像素）。r=1~2 通常足够；也可以写到 yml: train.mask_erode_train
            # r = int(self.opt.get('train', {}).get('mask_erode_train', 1))
            # if w is not None:
            #     w = erode_mask_bin(w, r)
            #
            # l_pix = self.cri_pix(out, gt, weight=w)  # BasicSR 的 L1/MSE/Charbonnier 都支持 weight=
            # l_total += l_pix
            # loss_dict['l_pix'] = l_pix

        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        # —— 可选：梯度损失 —— #
        if self.opt['train'].get('lambda_grad', 0.0) > 0:
            # 准备 Sobel 核
            if not hasattr(self, '_sobel_kx'):
                fx = torch.tensor([[-1., 0., 1.],
                                   [-2., 0., 2.],
                                   [-1., 0., 1.]], device=self.output.device, dtype=self.output.dtype).view(1, 1, 3, 3)
                fy = torch.tensor([[-1., -2., -1.],
                                   [0., 0., 0.],
                                   [1., 2., 1.]], device=self.output.device, dtype=self.output.dtype).view(1, 1, 3, 3)
                self._sobel_kx, self._sobel_ky = fx, fy

            def _grad_map(z):
                if z.dim() == 4:
                    c = z.size(1)
                    kx = self._sobel_kx.expand(c, 1, 3, 3)
                    ky = self._sobel_ky.expand(c, 1, 3, 3)
                    gx = F.conv2d(z, kx, padding=1, groups=c)
                    gy = F.conv2d(z, ky, padding=1, groups=c)
                    return torch.sqrt(gx * gx + gy * gy + 1e-12)
                return z

            g_sr = _grad_map(self.output)
            g_gt = _grad_map(self.gt)
            l_grad = torch.mean(torch.abs(g_sr - g_gt))
            l_total = l_total + float(self.opt['train'].get('lambda_grad')) * l_grad
            loss_dict['l_grad'] = l_grad

        l_total.backward()
        from torch.nn.utils import clip_grad_norm_
        clip_grad_norm_(self.net_g.parameters(), max_norm=float(self.opt['train'].get('clip_grad_norm', 1.0)))
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                # 修改前：
                self.output = self.net_g_ema(self.lq)
                # self.output = self.net_g_ema(self.lq, mask=getattr(self, 'mask_lq', None))
        else:
            self.net_g.eval()
            with torch.no_grad():
                # 修改前：
                self.output = self.net_g(self.lq)
                # self.output = self.net_g(self.lq, mask=getattr(self, 'mask_lq', None))
            self.net_g.train()

    def test_selfensemble(self):
        # TODO: to be tested
        # 8 augmentations
        # modified from https://github.com/thstkdgus35/EDSR-PyTorch

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        # prepare augmented data
        lq_list = [self.lq]
        for tf in 'v', 'h', 't':
            lq_list.extend([_transform(t, tf) for t in lq_list])

        # inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
        else:
            # self.net_g.eval()
            # with torch.no_grad():
            #     out_list = [self.net_g_ema(aug) for aug in lq_list]
            # self.net_g.train()
            self.net_g.eval()
            with torch.no_grad():
                out_list = [self.net_g(aug) for aug in lq_list]
            self.net_g.train()

        # merge results
        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 't')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')
        output = torch.cat(out_list, dim=0)

        self.output = output.mean(dim=0, keepdim=True)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        def _to_u16_hw_from_tensor(x, eps=2.0 / 65535.0):
            # x: (1,1,H,W)   or (1,C,H,W)  or (1,H,W)  or (H,W)
            import numpy as np, torch
            if isinstance(x, torch.Tensor):
                t = x.detach().float().cpu().clamp(0, 1)
                if t.dim() == 4 and t.size(0) == 1 and t.size(1) == 1:
                    t = t[0, 0]
                elif t.dim() == 4 and t.size(0) == 1 and t.size(1) > 1:
                    t = t[0].mean(0, keepdim=False)  # 多通道取均值
                elif t.dim() == 3 and t.size(0) == 1:
                    t = t[0]
                arr = t.numpy().astype(np.float32)
            else:
                arr = np.asarray(x, dtype=np.float32)
            arr = np.where(arr < eps, 0.0, arr)
            return np.rint(arr * 65535.0).astype(np.uint16)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            # —— 统一转 16-bit 全图 —— #
            sr_u16 = _to_u16_hw_from_tensor(visuals['result'])
            metric_data['img'] = sr_u16
            if 'gt' in visuals:
                gt_u16 = _to_u16_hw_from_tensor(visuals['gt'])
                metric_data['img2'] = gt_u16
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                # 直接以 16-bit 保存
                cv2.imwrite(save_img_path, sr_u16)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
