"""全局配置：所有超参集中在此，便于在 notebook 里一行覆盖。"""
import numpy as np

WORK_W, WORK_H = 672, 448      # 工作分辨率（≈原图1.5:1，且可被32整除以适配编码器）
SEED          = 42
VAL_RATIO     = 0.18           # 从32训练里再切出验证集

# U-Net 训练超参（A100 上 60 轮约几分钟；显存不足可调小 BATCH）
UNET_ENCODER  = "resnet34"
EPOCHS        = 60
BATCH         = 8
LR            = 1e-3

SVM_MAX       = 25000          # RBF-SVM 训练像素上限（O(n²) 复杂度，必须下采样）

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], np.float32)


def img_size():
    return (WORK_W, WORK_H)


def configure(**kw):
    """在 notebook 里用 ex.configure(EPOCHS=80, BATCH=4) 覆盖默认值。"""
    g = globals()
    for k, v in kw.items():
        if k in g:
            g[k] = v
        else:
            raise KeyError(f"未知配置项: {k}")
