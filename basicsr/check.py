import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch


# 填写你的 .state 路径
# state_path = r'D:\\Code\\BasicSR-master-16bit\\experiments\\IconVSR_AMSR2\\training_states\\300000.state'
#
# print(f"\n>>> 正在检查：{state_path}\n")
#
# try:
#     state = torch.load(state_path, map_location='cpu')
#     print(f"✔️ state文件成功读取，包含的键：{state.keys()}")
#
#     iter_num = state.get('iter', None)
#     epoch_num = state.get('epoch', None)
#     has_ema = 'ema_state_dict' in state
#
#     print(f"✔️ 迭代次数 iter: {iter_num}")
#     print(f"✔️ epoch: {epoch_num}")
#     print(f"✔️ 是否包含EMA: {has_ema}")
#
# except Exception as e:
#     print(f"⚠️ state文件读取失败，错误信息：{e}")

# 检查pth权重文件
# 填写你的 .pth 路径
pth_path = r'D:\Code\BasicSR-master-16bit\experiments\1.IconVSR_AMSR2-good\models\net_g_290000.pth'

print(f"\n>>> 正在检查：{pth_path}\n")

try:
    model_data = torch.load(pth_path, map_location='cpu')
    print(f"✔️ pth文件成功读取，包含的键：{model_data.keys()}")

    param_count = len(model_data.get('params', {}))
    param_ema_count = len(model_data.get('params_ema', {}))

    print(f"✔️ 正常参数数量：{param_count}")
    print(f"✔️ EMA参数数量：{param_ema_count}")

except Exception as e:
    print(f"⚠️ pth文件读取失败，错误信息：{e}")