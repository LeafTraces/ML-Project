# 眼底渗出液分割 — 交付说明与运行步骤

本包包含课程大作业的主要运行材料：**可一键运行的 Colab notebook、模块化源码、已生成的真实图表**。
notebook 现在只作为 Colab 运行接口，具体实验代码放在 `src/` 中；你只需在 Colab(A100/L4) 上运行 notebook，即可得到图表和指标。

---

## 一、文件清单

| 文件 | 用途 |
|---|---|
| `exudate_segmentation_colab.ipynb` | **运行接口**：在 Colab 中安装依赖、获取数据并调用 `src/` 模块 |
| `眼底渗出液分割_报告.docx` | 论文（结构完整，含真实图；结果数字处留有占位待填） |
| `眼底渗出液分割_答辩.pptx` | 答辩幻灯片（11 页；结果页留有占位待填） |
| `src/` | 模块化源码（数据读取、传统 ML、U-Net、评估、绘图与导出） |
| `figures/` | 已生成的真实图（预处理、候选检测、类别不平衡） |

---

## 二、在 Colab 上运行（最关键的步骤）

### 步骤 1：把数据准备好
数据已在你的仓库里（`topic/眼底图像的渗出液分割/EX数据`），**无需上传**——notebook 会自动 `git clone` 你的仓库取数据。
> 如果你 fork 了或改了仓库地址，打开 notebook 第 2 个代码单元，把 `REPO_URL` 改成你的仓库地址即可。

### 步骤 2：打开 notebook
任选其一：
- **方式 A（最简单）**：进入 https://colab.research.google.com → `文件` → `上传 notebook` → 选择 `exudate_segmentation_colab.ipynb`。
- **方式 B**：先把 `exudate_segmentation_colab.ipynb` push 到你的 GitHub 仓库，然后在 Colab `文件` → `打开 notebook` → `GitHub` 标签 → 粘贴仓库地址打开。

### 步骤 3：切换到 GPU
菜单 `代码执行程序` → `更改运行时类型` → 硬件加速器选 **GPU（A100 或 L4）** → 保存。

### 步骤 4：全部运行
菜单 `代码执行程序` → `全部运行`。整个流程在 A100 上约几分钟，包含：
1. 安装依赖、克隆数据；
2. 预处理 + 出图1（预处理示意）；
3. 调用 `src/` 中的传统 ML 模块，训练 SVM 和 Logistic Regression + 出线性模型特征系数图；
4. 训练 U-Net（方法二，ImageNet 预训练编码器）+ 出训练曲线；
5. 出 ROC/PR 曲线、混淆矩阵、定性对比图；
6. 打印**最终指标对比表**，并自动把所有产物打包为 `exudate_outputs.zip` 下载。

### 步骤 5：取回结果
运行结束会自动下载 `exudate_outputs.zip`，解压后得到：
```
outputs/
├── results.json          # SVM / Logistic Regression / U-Net 的全部指标 + 混淆矩阵原始数字
├── results_table.csv     # 方法对比表（可直接对照填表）
└── figures/
    ├── fig1_preprocess.png
    ├── fig2_feat_coefficients.png
    ├── fig3_training_curve.png
    ├── fig4_roc_pr.png
    ├── fig5_confusion.png
    └── fig6_qualitative.png
```

---

## 三、把结果填进论文和 PPT

如果需要把结果填进论文或 PPT，可以从 Colab 输出中取**结果数字与对比图**：

**论文 `眼底渗出液分割_报告.docx`：**
- 第 6 节“表 1”的方法行（SVM / Logistic Regression / U-Net）→ 用 `results_table.csv` 的数字替换占位的 `—`。
- 正文若干括号处（如“U-Net 的 Dice 为 …”）→ 同样按表填。
- 图 4–图 7 的灰底占位框 → 可分别插入 `fig2_feat_coefficients.png`、`fig3_training_curve.png`、`fig4_roc_pr.png`、`fig6_qualitative.png`。
- 混淆矩阵表的 TP/FP/FN/TN → 用 `results.json` 里的 `confusion` / `unet_confusion`。

**PPT `眼底渗出液分割_答辩.pptx`：**
- 第 9 页结果表 → 同上填数字。
- 第 9 页左侧“插入 fig4_roc_pr.png”占位 → 插入该图。

> 提示：论文里的“图 1/图 2/图 3（不平衡分布、预处理、候选检测）”已经是 `figures/` 里的真实图，**无需改动**。

---

## 四、方法概要（便于讲解）

- **方法一（传统 ML）**：预处理(去光照+CLAHE) → 候选检测(Top-hat) → 14 维手工特征 → SVM 与 Logistic Regression（FOV 内稠密像素分类）。
- **方法二（深度学习）**：ImageNet 预训练 ResNet34 编码器的 U-Net；BCE+Tversky 组合损失对抗类别不平衡；数据增强 + 余弦退火 + 混合精度。
- **统一评价**：三种模型共享同一预处理、同一 32/15 划分、同一指标计算（混淆矩阵/准召/ROC/PR/IoU/Dice + 学习时间），保证对比公平。

---

## 五、常见问题

- **找不到数据**：检查 notebook 第 2 单元的 `REPO_URL` 与 `DATA_SUBDIR` 是否与你的仓库一致。
- **显存不足**：把第 3 单元的 `BATCH` 调小（如 4），或把 `WORK_W, WORK_H` 调小。
- **想要更强结果**：可把 `UNET_ENCODER` 换成 `"efficientnet-b3"`、`EPOCHS` 提到 80–100。
- **U-Net 比 SVM/LR 差**：通常是 GPU 没开或 epoch 太少；确认用了 GPU 运行时且预训练权重已下载。
