import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
# from basicsr.archs.iconvsr_arch import IconVSR
from basicsr.archs import basicvsr_arch

def save_tensor_image(tensor, path):
    """保存 tensor 图像，默认归一化 [0,1]->uint16"""
    img = tensor.detach().cpu().clamp(0, 1).squeeze().numpy()
    img = np.rint(img * 65535).astype(np.uint16)
    cv2.imwrite(path, img)

@torch.no_grad()
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ### 参数设置（模拟 AMSR2 配置）
    T = 5               # 帧数
    H, W = 64, 64       # 原始低分图尺寸
    scale = 4           # 放大倍数
    model_in_size = (H, W)
    model_out_size = (H * scale, W * scale)

    ### 构造测试输入
    lq = torch.zeros(1, T, 1, H, W).to(device)       # 全为0
    mid = T // 2
    lq[0, mid, 0, H//2-1:H//2+2, W//2-1:W//2+2] = 1.0  # 中心帧3x3区域赋值为1

    ### 构造 IconVSR 模型（要替换为你训练过的权重）
    model = basicvsr_arch.IconVSR(
        num_feat=64, num_block=30, keyframe_stride=5, temporal_padding=2,
        spynet_path=None, edvr_path=None
    ).to(device)
    model.eval()

    # 可选加载你自己的模型
    # model.load_state_dict(torch.load('path_to_your_model.pth'), strict=True)

    ### 推理（无掩膜，仅测试结构）
    with torch.cuda.amp.autocast(), torch.no_grad():
        output = model(lq)  # (N,T,C,Hh,Wh)

    ### 保存所有帧的输出图
    os.makedirs('rf_test_output', exist_ok=True)
    for t in range(T):
        out_img = output[0, t, 0]  # 单帧灰度图
        save_tensor_image(out_img, f'rf_test_output/im{t+1}.png')

    print("Saved to ./rf_test_output")

if __name__ == '__main__':
    main()
