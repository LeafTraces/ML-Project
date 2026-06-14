"""
exudate —— 眼底渗出液分割（逻辑回归 / RBF-SVM / U-Net）。

在 Colab 里的典型用法：
    import exudate as ex
    data = ex.load_data(DATA_ROOT)        # 加载并预处理
    clf  = ex.run_classical(data)         # 训练 逻辑回归 + SVM
    unet = ex.run_unet(data)              # 训练 U-Net（GPU）
    df   = ex.report(data, clf, unet)     # 出全部图表 + 指标表
    ex.make_zip()                         # 打包 outputs/

或一行跑完：
    df = ex.run_all(DATA_ROOT)
"""
from . import config
from .config import configure
from .data import load_data, load_split, build_index
from .classical import run_classical, lr_feature_coef, FEATURE_NAMES
from .segmentation import run_unet, build_unet
from .pipeline import run_all, report, assemble_all, make_zip

__all__ = ["config", "configure", "load_data", "load_split", "build_index",
           "run_classical", "lr_feature_coef", "FEATURE_NAMES",
           "run_unet", "build_unet", "run_all", "report", "assemble_all", "make_zip"]
