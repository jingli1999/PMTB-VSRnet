# import re
# import matplotlib.pyplot as plt
#
# # 两个日志文件路径
# log_paths = [
#     r"D:\Code\BasicSR-master-16bit\experiments\IconVSR_AMSR2\train_IconVSR_AMSR2_20250701_151719.log",
#     r"D:\Code\BasicSR-master-16bit\experiments\IconVSR_AMSR2\train_IconVSR_AMSR2_20250704_163032.log"
# ]
#
# iters = []
# losses = []
# psnr_iters = []
# psnrs = []
#
# for log_path in log_paths:
#     with open(log_path, 'r', encoding='gbk', errors='ignore') as f:
#         for line in f:
#             # 提取 loss
#             match_loss = re.search(r"iter:\s*([\d,]+).*?l_pix:\s*([\d\.e\+\-]+)", line)
#             if match_loss:
#                 iter_num = int(match_loss.group(1).replace(',', ''))
#                 loss_val = float(match_loss.group(2))
#                 iters.append(iter_num)
#                 losses.append(loss_val)
#
#             # 提取 PSNR
#             match_psnr = re.search(r"psnr:\s*([\d\.]+)", line)
#             if match_psnr:
#                 psnr_val = float(match_psnr.group(1))
#                 last_iter = iters[-1] if iters else 0
#                 psnr_iters.append(last_iter)
#                 psnrs.append(psnr_val)
#
# # 查找最优PSNR
# if psnrs:
#     best_idx = psnrs.index(max(psnrs))
#     best_iter = psnr_iters[best_idx]
#     best_psnr = psnrs[best_idx]
#     print(f"最优PSNR出现在iter={best_iter}，PSNR={best_psnr:.4f}")
# else:
#     print("日志中未找到PSNR信息")
#
# # 绘图
# plt.figure(figsize=(12, 6))
#
# # Loss曲线
# plt.subplot(1, 2, 1)
# plt.plot(iters, losses, label="l_pix Loss", color='blue')
# plt.xlabel("Iteration")
# plt.ylabel("Loss")
# plt.title("Loss Curve")
# plt.grid()
# plt.legend()
#
# # PSNR曲线
# plt.subplot(1, 2, 2)
# plt.plot(psnr_iters, psnrs, marker='o', label="Validation PSNR", color='green')
# plt.axvline(best_iter, color='red', linestyle='--', label=f'Best PSNR @ {best_iter}')
# plt.xlabel("Iteration")
# plt.ylabel("PSNR")
# plt.title("Validation PSNR Curve")
# plt.grid()
# plt.legend()
#
# plt.tight_layout()
# plt.show()

import re
import matplotlib.pyplot as plt

# === 修改为你的 log 文件路径 ===
log_path = "E:\\code\\BasicSR-16bit-improve\\experiments\\train_PMTB_VSR_AMSR2_18.7H\\train_train_PMTB_VSR_AMSR2_18.7H_20260202_163614.log"

# === 初始化 ===
iters, losses = [], []
val_iters, psnrs = [], []

# === 打开文件时使用 GBK 编码（Windows 默认编码）===
# with open(log_path, 'r', encoding='gbk') as f:
#     lines = f.readlines()
with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
    lines = f.read().splitlines()


# 标志：是否刚遇到 Validation AMSR2 行
awaiting_psnr = False

for line in lines:
    # 匹配训练信息 l_pix
    train_match = re.search(r"iter:\s*([0-9,]+).*?l_pix:\s*([\d\.eE+-]+)", line)
    if train_match:
        iter_num = int(train_match.group(1).replace(',', ''))
        loss_val = float(train_match.group(2))
        iters.append(iter_num)
        losses.append(loss_val)

    # 检测到验证标志
    if 'Validation AMSR2' in line:
        awaiting_psnr = True
        continue

    # 下一行读取 PSNR
    if awaiting_psnr:
        psnr_match = re.search(r'psnr:\s*([\d\.]+)', line)
        if psnr_match:
            psnr_val = float(psnr_match.group(1))
            val_iters.append(iters[-1] if iters else len(val_iters) * 1000)
            psnrs.append(psnr_val)
        awaiting_psnr = False

# === 绘图：训练损失 ===
plt.figure(figsize=(8, 5))
plt.plot(iters, losses, label='Train Loss (l_pix)', color='blue')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss over Iterations')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('train_loss_curve.png')
plt.show()

# === 绘图：验证 PSNR ===
plt.figure(figsize=(8, 5))
plt.plot(val_iters, psnrs, label='Validation PSNR', color='green')
plt.xlabel('Iteration')
plt.ylabel('PSNR (dB)')
plt.title('Validation PSNR over Iterations')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('val_psnr_curve.png')
plt.show()

# === 输出最优迭代 ===
if psnrs:
    best_idx = psnrs.index(max(psnrs))
    best_iter = val_iters[best_idx]
    best_psnr = psnrs[best_idx]
    print(f"✅ 最优 PSNR = {best_psnr:.4f} dB，出现在迭代次数 = {best_iter}")
else:
    print("⚠️ 没有找到任何 PSNR 记录，请检查日志格式")