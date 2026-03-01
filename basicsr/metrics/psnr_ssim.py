import cv2
import numpy as np
import torch
import torch.nn.functional as F

from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.color_util import rgb2ycbcr_pt
from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_psnr(img, img2, mask=None, crop_border=0, input_order='HWC',
                   test_y_channel=False, normalize=False, ignore_ge=None, **kwargs):
    """
    改动要点：
      1) 将像素==0 的位置视为掩膜外（两幅图任意一幅为0都排除）。
      2) 不进行归一化（normalize=False）。动态选择 max_val（1/255/65535）。
    其他流程（裁边、可选Y通道）不变。
    """
    assert img.shape == img2.shape, f'Image shapes are different: {img.shape}, {img2.shape}.'
    img  = reorder_image(img,  input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    # 灰度兼容：保证有通道维
    if img.ndim == 2:  img  = img[..., None]
    if img2.ndim == 2: img2 = img2[..., None]

    # 裁边
    if crop_border:
        img  = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
        if mask is not None:
            mask = mask[crop_border:-crop_border, crop_border:-crop_border, ...]

    # 可选：Y通道
    if test_y_channel:
        img  = to_y_channel(img)
        img2 = to_y_channel(img2)
        if img.ndim  == 2: img  = img[..., None]
        if img2.ndim == 2: img2 = img2[..., None]

    img  = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    # —— 新增：以“0”为掩膜外；两幅图任意一幅为0都视为无效 —— #
    # zeros1 = (img  == 0)
    # zeros2 = (img2 == 0)
    # if zeros1.ndim == 3:
    #     zeros1 = zeros1.any(axis=2)  # (H,W)
    #     zeros2 = zeros2.any(axis=2)
    # zero_invalid = (zeros1 | zeros2)  # (H,W)
    # —— 可选：同时支持 ignore_ge（保留原有能力） —— #
    # auto_valid = ~zero_invalid
    # if ignore_ge is not None:
    #     bad_ge = (img >= ignore_ge) | (img2 >= ignore_ge)
    #     if bad_ge.ndim == 3:
    #         bad_ge = bad_ge.any(axis=2)
    #     auto_valid = auto_valid & (~bad_ge)

    # # 组合外部 mask（若提供）
    # if mask is not None:
    #     mask = mask.astype(bool)
    #     if mask.ndim == 3:
    #         mask = mask.any(axis=2)
    #     valid = (mask & auto_valid)
    # else:
    #     valid = auto_valid  # (H,W) 或 None

    # 不归一化：依据原值域选择 max_val
    # 若数据范围在[0,1]，则 max_val=1；若有>255 则按 16 位，反之按 8 位
    max_val = 1.0 if (img.max() <= 1.0 and img2.max() <= 1.0) else (
        65535.0 if (img.max() > 255 or img2.max() > 255) else 255.0)
    mse = np.mean((img - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10((max_val ** 2) / mse)



@METRIC_REGISTRY.register()
def calculate_ssim(img, img2, crop_border, input_order='HWC', test_y_channel=False,
                   mask=None, ignore_value=None, ignore_threshold=None, normalize=False, **kwargs):
    """Calculate SSIM with optional mask and 8/16-bit support.
       改动要点：将像素==0 的位置视为掩膜外（两幅图任意一幅为0都排除）；不归一化。
    """

    # def _ssim_one(x, y, c1, c2, valid_map=None):
    #     kernel = cv2.getGaussianKernel(11, 1.5)
    #     window = np.outer(kernel, kernel.transpose())
    #     mu1 = cv2.filter2D(x, -1, window)[5:-5, 5:-5]
    #     mu2 = cv2.filter2D(y, -1, window)[5:-5, 5:-5]
    #     mu1_sq = mu1 ** 2
    #     mu2_sq = mu2 ** 2
    #     mu1_mu2 = mu1 * mu2
    #     sigma1_sq = cv2.filter2D(x * x, -1, window)[5:-5, 5:-5] - mu1_sq
    #     sigma2_sq = cv2.filter2D(y * y, -1, window)[5:-5, 5:-5] - mu2_sq
    #     sigma12 = cv2.filter2D(x * y, -1, window)[5:-5, 5:-5] - mu1_mu2
    #     ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    #     if valid_map is None:
    #         return ssim_map.mean()
    #     vm = valid_map[5:-5, 5:-5]
    #     return ssim_map[vm].mean() if vm.any() else np.nan

    assert img.shape == img2.shape, f'Image shapes are different: {img.shape}, {img2.shape}.'
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}.')
    img  = reorder_image(img,  input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    # 灰度兼容：保证有通道维
    if img.ndim == 2:  img  = img[...,  None]
    if img2.ndim == 2: img2 = img2[..., None]

    # 裁边
    if crop_border != 0:
        img  = img [crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
        if mask is not None:
            mask = mask[crop_border:-crop_border, crop_border:-crop_border, ...]

    # # 在原值域上构造“有效区”：
    # # 1) 0 值为掩膜外（两幅图任意一幅为0就无效）
    # zeros1 = (img  == 0)
    # zeros2 = (img2 == 0)
    # if zeros1.ndim == 3:
    #     zeros1 = zeros1.any(axis=2)
    #     zeros2 = zeros2.any(axis=2)
    # auto_valid = ~(zeros1 | zeros2)  # (H,W)
    #
    # # 2) 可选 ignore_value（保持原能力）
    # if ignore_value is not None:
    #     inv1 = (img  == ignore_value)
    #     inv2 = (img2 == ignore_value)
    #     if inv1.ndim == 3:
    #         inv1 = inv1.any(axis=2)
    #         inv2 = inv2.any(axis=2)
    #     auto_valid = auto_valid & ~(inv1 | inv2)
    #
    # # 3) 可选阈值忽略
    # if ignore_threshold is not None:
    #     thr = float(ignore_threshold)
    #     thr1 = (img  >= thr)
    #     thr2 = (img2 >= thr)
    #     if thr1.ndim == 3:
    #         thr1 = thr1.any(axis=2)
    #         thr2 = thr2.any(axis=2)
    #     auto_valid = auto_valid & ~(thr1 | thr2)

    # 可选：Y通道
    if test_y_channel:
        img  = to_y_channel(img)
        img2 = to_y_channel(img2)
        if img.ndim  == 2: img  = img[...,  None]
        if img2.ndim == 2: img2 = img2[..., None]

    img  = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    max_val = 1.0 if (img.max() <= 1.0 and img2.max() <= 1.0) else (
        65535.0 if (img.max() > 255 or img2.max() > 255) else 255.0)
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    ssims = []
    for i in range(img.shape[2]):
        ssims.append(_ssim(img[..., i], img2[..., i], c1, c2))
    return np.array(ssims).mean()


def _ssim(img, img2, c1, c2, valid_map=None):
    """SSIM for one-channel images, supports 8/16-bit, with optional mask mean."""
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    if valid_map is None:
        return ssim_map.mean()
    else:
        vm = valid_map[5:-5, 5:-5]
        if vm.any():
            return ssim_map[vm].mean()
        else:
            return np.nan


def _ssim_pth(img, img2):
    """PyTorch版 SSIM 基函数（保持不变：假设输入已在[0,1]）。"""
    max_val = 1.0
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    window = torch.from_numpy(window).view(1, 1, 11, 11).expand(img.size(1), 1, 11, 11).to(img.dtype).to(img.device)

    mu1 = F.conv2d(img, window, stride=1, padding=0, groups=img.shape[1])  # valid mode
    mu2 = F.conv2d(img2, window, stride=1, padding=0, groups=img2.shape[1])  # valid mode
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img * img, window, stride=1, padding=0, groups=img.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu1_mu2

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
    return ssim_map.mean([1, 2, 3])

@METRIC_REGISTRY.register()
def calculate_grad_psnr(img, img2, crop_border=0, input_order='HWC', test_y_channel=False, **kwargs):
    """
    GRAD-PSNR: 先把两张图分别做 Sobel 梯度幅值（在 [0,max] 值域上），
    再按 PSNR 公式计算。内部会把源图按其 max_val 归一到 [0,1] 后取梯度，
    使得不同位深的结果可比。
    """
    assert img.shape == img2.shape, f'Image shapes are different: {img.shape}, {img2.shape}.'
    img  = reorder_image(img,  input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    if img.ndim == 2:  img  = img[..., None]
    if img2.ndim == 2: img2 = img2[..., None]
    if crop_border:
        img  = img [crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    if test_y_channel:
        img  = to_y_channel(img);   img  = img[..., None] if img.ndim  == 2 else img
        img2 = to_y_channel(img2);  img2 = img2[..., None] if img2.ndim == 2 else img2

    img  = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    # 归一到 [0,1] 再做 Sobel，保证位深无关
    max_val = 1.0 if (img.max() <= 1.0 and img2.max() <= 1.0) else (65535.0 if (img.max() > 255 or img2.max() > 255) else 255.0)
    a = img  / max_val
    b = img2 / max_val

    def sobel_mag(x):  # x: HWC, [0,1]
        k = cv2.getGaussianKernel(1, 0)  # no blur, 占位
        # OpenCV Sobel 在边界使用默认的 BORDER_DEFAULT，保持一致性
        gx = cv2.Sobel(x, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(x, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx*gx + gy*gy + 1e-12)
        return mag

    # 对每个通道做梯度幅值，再求均值
    mags = []
    for i in range(a.shape[2]):
        mags.append( (sobel_mag(a[..., i]) , sobel_mag(b[..., i])) )
    A = np.mean([m[0] for m in mags], axis=0)
    B = np.mean([m[1] for m in mags], axis=0)

    mse = np.mean((A - B) ** 2)
    if mse == 0: return float('inf')
    # 在 [0,1] 上计算 PSNR
    return 10.0 * np.log10(1.0 / mse)


@METRIC_REGISTRY.register()
def calculate_grad_ssim(img, img2, crop_border=0, input_order='HWC', test_y_channel=False, **kwargs):
    """
    GRAD-SSIM: 先把两张图归一到 [0,1]，做 Sobel 幅值，再在 [0,1] 上计算 SSIM。
    """
    assert img.shape == img2.shape, f'Image shapes are different: {img.shape}, {img2.shape}.'
    img  = reorder_image(img,  input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    if img.ndim == 2:  img  = img[..., None]
    if img2.ndim == 2: img2 = img2[..., None]
    if crop_border:
        img  = img [crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img  = to_y_channel(img);   img  = img[..., None] if img.ndim  == 2 else img
        img2 = to_y_channel(img2);  img2 = img2[..., None] if img2.ndim == 2 else img2

    img  = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    max_val = 1.0 if (img.max() <= 1.0 and img2.max() <= 1.0) else (65535.0 if (img.max() > 255 or img2.max() > 255) else 255.0)
    a = img  / max_val
    b = img2 / max_val

    def sobel_mag(x):
        gx = cv2.Sobel(x, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(x, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(gx*gx + gy*gy + 1e-12)

    # 单通道化（多通道取均值）
    A = np.mean([sobel_mag(a[..., i]) for i in range(a.shape[2])], axis=0)
    B = np.mean([sobel_mag(b[..., i]) for i in range(b.shape[2])], axis=0)

    # 在 [0,1] 上配套 SSIM 的 c1/c2
    c1 = (0.01 * 1.0) ** 2
    c2 = (0.03 * 1.0) ** 2
    return _ssim(A, B, c1, c2)
