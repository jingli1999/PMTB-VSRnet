import os
import cv2
import argparse
from tqdm import tqdm


def mod_crop(img, scale: int):
    """裁剪图像，确保高宽可被 scale 整除."""
    h, w = img.shape[:2]
    h_remainder = h % scale
    w_remainder = w % scale
    return img[0:h - h_remainder, 0:w - w_remainder]


def generate_bicubic_lr(input_dir: str, output_dir: str, scale: int = 4):
    """
    批量生成 Bicubic 缩小图像（适用于 AMSR2 16-bit 单通道 BT 图像）.

    Args:
        input_dir: HR 图像所在文件夹（16-bit 单通道 PNG/TIF）
        output_dir: 生成 LR 图像保存路径
        scale: 下采样比例，默认 4
    """
    assert os.path.isdir(input_dir), f'输入路径不存在：{input_dir}'
    os.makedirs(output_dir, exist_ok=True)

    img_names = sorted(os.listdir(input_dir))
    if not img_names:
        print(f'警告：{input_dir} 为空目录。')
        return

    for name in tqdm(img_names, desc='Generating bicubic LR'):
        in_path = os.path.join(input_dir, name)
        if not os.path.isfile(in_path):
            continue

        # 以“原始深度”读入，保留 16-bit
        img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f'跳过无法读取的文件：{in_path}')
            continue

        # 如果不小心变成多通道，这里强制取第一个通道
        if img.ndim == 3:
            img = img[..., 0]

        # 裁剪到可被 scale 整除
        img = mod_crop(img, scale)
        h, w = img.shape[:2]

        # Bicubic 下采样
        lr = cv2.resize(
            img,
            (w // scale, h // scale),
            interpolation=cv2.INTER_CUBIC
        )

        out_path = os.path.join(output_dir, name)
        # 仍然以 16-bit 保存（假设输入是 uint16）
        cv2.imwrite(out_path, lr)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate bicubic-downsampled LR images from HR AMSR2 BT images.'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='HR 图像所在文件夹（16-bit 单通道 PNG/TIF）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='生成 LR 图像保存文件夹'
    )
    parser.add_argument(
        '--scale',
        type=int,
        default=4,
        help='下采样比例，默认 4'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    generate_bicubic_lr(args.input_dir, args.output_dir, args.scale)