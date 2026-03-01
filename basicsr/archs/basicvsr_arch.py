import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, flow_warp, make_layer
from .edvr_arch import PCDAlignment, TSAFusion
# from .spynet_arch import SpyNet  # 仍禁用光流

@ARCH_REGISTRY.register()
class BasicVSR(nn.Module):
    """A recurrent network for video SR. Now only x4 is supported.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
    """

    def __init__(self, num_feat=64, num_block=15, spynet_path=None):
        super().__init__()
        self.num_feat = num_feat

        # propagation
        self.backward_trunk = ConvResidualBlocks(num_feat + 1, num_feat, num_block)
        self.forward_trunk = ConvResidualBlocks(num_feat + 1, num_feat, num_block)

        # reconstruction
        self.fusion = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0, bias=True)
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 1, 3, 1, 1)  # 单通道输出

        self.pixel_shuffle = nn.PixelShuffle(2)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_flow(self, x):
        # 禁用真实光流：返回零张量
        b, n, c, h, w = x.size()
        flows_forward = x.new_zeros(b, n - 1, 2, h, w)
        flows_backward = x.new_zeros(b, n - 1, 2, h, w)
        return flows_forward, flows_backward

    def forward(self, x):
        """x: (B, N, 1, H, W)"""
        flows_forward, flows_backward = self.get_flow(x)
        b, n, _, h, w = x.size()

        # backward branch
        out_l = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(n - 1, -1, -1):
            x_i = x[:, i, :, :, :]
            # 不进行光流 warp
            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop)

        # forward branch
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            # 不进行光流 warp
            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = torch.cat([out_l[i], feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l[i] = out

        return torch.stack(out_l, dim=1)


class ConvResidualBlocks(nn.Module):
    """Conv and residual block used in BasicVSR."""
    def __init__(self, num_in_ch, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch)
        )

    def forward(self, fea):
        return self.main(fea)


@ARCH_REGISTRY.register()
class PMTB_VSR(nn.Module):
    """
     PMTB_VSR（逐帧补充机制版，padding=1）：
    - 去除“关键帧策略”，改为对序列里**每一帧**都做一次信息补充：
      对第 i 帧，取窗口 [i-1, i, i+1]（边界处夹取） → EDVR → feat_refill[i]
    - 在前/后向传播两个分支，均把 feat_prop 与 feat_refill[i] 拼接并用 3×3 Conv 融合
    - 仍禁用外部光流（SpyNet）；EDVR 内部使用 PCD/TSA 做局部窗口内的时空对齐与融合
    - 单通道输入/输出
    """
    def __init__(self,
                 num_feat=64,
                 num_block=15,
                 temporal_padding=1,   # 固定为 1（窗长 2*1+1=3）
                 spynet_path=None,
                 edvr_path=None):
        super().__init__()

        # ---- 固定 padding=1 ----
        self.num_feat = num_feat
        self.temporal_padding = 1

        # 逐帧补充用 EDVR（输入帧数=3）
        self.edvr = EDVRFeatureExtractor(num_input_frame=3, num_feat=num_feat, load_path=edvr_path)

        # propagation
        self.backward_fusion = nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1, bias=True)
        self.backward_trunk  = ConvResidualBlocks(num_feat + 1, num_feat, num_block)

        self.forward_fusion  = nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1, bias=True)
        self.forward_trunk   = ConvResidualBlocks(2 * num_feat + 1, num_feat, num_block)

        # reconstruction
        self.upconv1   = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2   = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hr   = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 1, 3, 1, 1)  # 单通道输出
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.fusion = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    # --------- 工具函数 ---------
    def pad_spatial(self, x):
        """确保 H、W 可被 4 整除（EDVR 的 PCD 习惯）。"""
        n, t, c, h, w = x.size()
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        x = x.view(-1, c, h, w)
        # 修改前：
        # x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        #修改后：
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='replicate')
        return x.view(n, t, c, h + pad_h, w + pad_w)

    def get_flow(self, x):
        """禁用外部光流（返回全 0）。"""
        b, n, c, h, w = x.size()
        flows_forward  = x.new_zeros(b, n - 1, 2, h, w)
        flows_backward = x.new_zeros(b, n - 1, 2, h, w)
        return flows_forward, flows_backward

    def _window3(self, x, i):
        """取 [i-1, i, i+1]，边界夹取。x:(B,N,1,H,W) -> (B,3,1,H,W)"""
        b, n, c, h, w = x.size()
        i0 = max(i - 1, 0)
        i1 = i
        i2 = min(i + 1, n - 1)
        return torch.stack([x[:, i0], x[:, i1], x[:, i2]], dim=1)

    def get_refill_features(self, x):
        B, N, C, H, W = x.size()
        device = x.device
        idx = torch.arange(N, device=device)
        im1 = torch.clamp(idx - 1, 0, N - 1)
        ip1 = torch.clamp(idx + 1, 0, N - 1)

        win_im1 = x[:, im1]  # (B,N,1,H,W)
        win_i = x  # (B,N,1,H,W)
        win_ip1 = x[:, ip1]  # (B,N,1,H,W)

        # (B,N,3,1,H,W) → (B*N,3,1,H,W)
        win = torch.stack([win_im1, win_i, win_ip1], dim=2).contiguous()
        win = win.view(B * N, 3, 1, H, W)

        feat_bn = self.edvr(win)  # (B*N, C, H, W)
        feat_bn = feat_bn.view(B, N, -1, H, W)
        return {i: feat_bn[:, i] for i in range(N)}

    # --------- 主过程 ---------
    def forward(self, x):
        """
        x: (B, N, 1, H, W)  # 灰度
        """
        b, n, _, h_input, w_input = x.size()

        # 空间填充（供 EDVR/PixelShuffle）
        x = self.pad_spatial(x)
        _, _, _, h, w = x.size()

        # 禁用光流
        flows_forward, flows_backward = self.get_flow(x)

        # 逐帧 EDVR 补充特征
        feats_refill = self.get_refill_features(x)    # {i: (B,C,H,W)}

        # ===== backward branch =====
        out_l = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(n - 1, -1, -1):
            x_i = x[:, i, :, :, :]

            # 注入对应帧的补充特征
            feat_prop = torch.cat([feat_prop, feats_refill[i]], dim=1)  # (B,2C,H,W)
            feat_prop = self.backward_fusion(feat_prop)                 # (B,C,H,W)

            # 与当前像素拼接，进入 backward 主干
            feat_prop = torch.cat([x_i, feat_prop], dim=1)              # (B,1+C,H,W)
            feat_prop = self.backward_trunk(feat_prop)                  # (B,C,H,W)

            out_l.insert(0, feat_prop)

        # ===== forward branch =====
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]

            # 注入对应帧的补充特征
            feat_prop = torch.cat([feat_prop, feats_refill[i]], dim=1)  # (B,2C,H,W)
            feat_prop = self.forward_fusion(feat_prop)                  # (B,C,H,W)

            # 结合 backward 特征与当前像素
            feat_prop = torch.cat([x_i, out_l[i], feat_prop], dim=1)    # (B,1+2C,H,W)
            feat_prop = self.forward_trunk(feat_prop)                   # (B,C,H,W)

            # 重建
            out = torch.cat([out_l[i], feat_prop], dim=1)               # (B,2C,H,W)
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            # 修改前：
            # base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            # 修改后：
            base = F.interpolate(x_i, scale_factor=4, mode='bicubic', align_corners=False)
            out += base
            out_l[i] = out

        # 裁到原 H,W 的 4 倍（去除 pad 影响）
        return torch.stack(out_l, dim=1)[..., :4 * h_input, :4 * w_input]


class EDVRFeatureExtractor(nn.Module):
    """EDVR feature extractor for refill (num_input_frame should be 3 for padding=1)."""
    def __init__(self, num_input_frame, num_feat, load_path):
        super(EDVRFeatureExtractor, self).__init__()
        assert num_input_frame == 3, "逐帧补充机制固定使用窗口大小 3（padding=1）"
        self.center_frame_idx = num_input_frame // 2

        # extract pyramid features
        self.conv_first = nn.Conv2d(1, num_feat, 3, 1, 1)
        self.feature_extraction = make_layer(ResidualBlockNoBN, 5, num_feat=num_feat)
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # pcd and tsa module
        # 修改前：
        # self.pcd_align = PCDAlignment(num_feat=num_feat, deformable_groups=8)
        # 修改后：
        self.pcd_align = PCDAlignment(num_feat=num_feat, deformable_groups=4)

        self.fusion = TSAFusion(num_feat=num_feat, num_frame=num_input_frame, center_frame_idx=self.center_frame_idx)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        if load_path:
            self.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage)['params'])

    def forward(self, x):
        """
        x: (B, 3, 1, H, W)  # [i-1, i, i+1] 的小窗口
        return: (B, C, H, W)
        """
        b, n, c, h, w = x.size()

        # extract features for each frame
        # L1
        feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
        feat_l1 = self.feature_extraction(feat_l1)
        # L2
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        # L3
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))

        feat_l1 = feat_l1.view(b, n, -1, h, w)
        feat_l2 = feat_l2.view(b, n, -1, h // 2, w // 2)
        feat_l3 = feat_l3.view(b, n, -1, h // 4, w // 4)

        # PCD alignment
        ref_feat_l = [
            feat_l1[:, self.center_frame_idx, :, :, :].clone(),
            feat_l2[:, self.center_frame_idx, :, :, :].clone(),
            feat_l3[:, self.center_frame_idx, :, :, :].clone()
        ]
        aligned_feat = []
        for i in range(n):
            nbr_feat_l = [
                feat_l1[:, i, :, :, :].clone(),
                feat_l2[:, i, :, :, :].clone(),
                feat_l3[:, i, :, :, :].clone()
            ]
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))

        aligned_feat = torch.stack(aligned_feat, dim=1)  # (B, 3, C, H, W)

        # aligned_feat: (B, 3, C, H, W), x: (B,3,1,H,W)
        with torch.no_grad():
            # 低频 NCC：4× 平均池化
            x_low = F.avg_pool2d(x.view(b * n, c, h, w), kernel_size=4, stride=4)
            x_low = x_low.view(b, n, c, h // 4, w // 4)
            ref_low = x_low[:, self.center_frame_idx]  # (B,1,h/4,w/4)

            def ncc(a, b, eps=1e-6):
                a = a - a.mean(dim=(2, 3), keepdim=True)
                b = b - b.mean(dim=(2, 3), keepdim=True)
                a = a / (a.std(dim=(2, 3), keepdim=True) + eps)
                b = b / (b.std(dim=(2, 3), keepdim=True) + eps)
                return (a * b).mean(dim=(2, 3), keepdim=True)  # (B,1,1,1)

            scores = []
            for i in range(n):  # n=3
                scores.append(ncc(ref_low, x_low[:, i]))
            S = torch.stack(scores, dim=1)  # (B,3,1,1,1)
            w = torch.softmax(S / 0.1, dim=1)  # 温度 0.1，可调

        # 相似度加权（广播到 C,H,W）
        aligned_feat = aligned_feat * w.expand_as(aligned_feat)

        # TSA fusion (到中心帧尺度)
        return self.fusion(aligned_feat)
