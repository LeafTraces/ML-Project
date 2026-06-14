# 眼底渗出液分割 — 交付说明与运行步骤

**代码与笔记分离**：所有实现都在 `exudate/` Python 包里；Colab notebook 只是薄壳，几句简单调用即可跑完整套流程。
三种方法对比：**逻辑回归 (LR) · 支持向量机 SVM(RBF) · U-Net（预训练编码器）**。

---

## 一、文件清单

| 文件 | 用途 |
|---|---|
| `exudate/`（代码包） | **核心实现**：config / data / metrics / classical / segmentation / viz / pipeline |
| `exudate_segmentation_colab.ipynb` | **薄壳 notebook**：只写说明 + 调用 `exudate`，在 Colab 运行 |
| `眼底渗出液分割_报告.docx` | 论文（结构完整，含真实图；结果数字处留占位待填） |
| `眼底渗出液分割_答辩.pptx` | 答辩 PPT（11 页；结果页留占位待填） |
| `figures/` | 已生成的真实图（预处理、候选检测、类别不平衡），论文图1–3 已用 |

---

## 二、放置到仓库（关键）

notebook 运行时会 `git clone` 你的仓库，并从中导入 `exudate` 包，所以**代码包必须在仓库里**。
请把 `exudate/` 文件夹放到与 `EX数据` 同级的位置：

```
ML-Proj/
└── topic/眼底图像的渗出液分割/
    ├── EX数据/              ← 老师给的数据（已在仓库）
    └── exudate/             ← 把本包放到这里，然后 push
```

```bash
# 在你本地仓库根目录执行
cp -r exudate "topic/眼底图像的渗出液分割/"
git add "topic/眼底图像的渗出液分割/exudate"
git commit -m "add exudate package"
git push
```

---

## 三、在 Colab 上运行

1. **打开 notebook**：Colab → `文件` → `上传 notebook` → 选 `exudate_segmentation_colab.ipynb`
   （或先把它也 push 到仓库，再用 `文件 → 打开 notebook → GitHub` 打开）。
2. **切 GPU**：`代码执行程序` → `更改运行时类型` → GPU（A100 或 L4）。
3. **全部运行**（或逐格运行，看每步输出与图）。notebook 里的调用就是这几句：
   ```python
   import exudate as ex
   data = ex.load_data(DATA_ROOT)        # 加载 + 预处理
   clf  = ex.run_classical(data)         # 训练 逻辑回归 + RBF-SVM
   unet = ex.run_unet(data)              # 训练 U-Net（GPU）
   df   = ex.report(data, clf, unet, out_dir="outputs")   # 出全部图表 + 指标表
   ex.make_zip("outputs")                # 打包下载
   ```
   想一行跑完也行：`df = ex.run_all(DATA_ROOT)`。
4. 运行结束自动下载 `exudate_outputs.zip`，内含：
   ```
   outputs/
   ├── results.json        # 三种方法全部指标 + 混淆矩阵原始数字
   ├── results_table.csv   # 方法对比表
   └── figures/fig1~fig6.png
   ```

---

## 四、把结果填进论文 / PPT

论文与 PPT 的方法/流程/验证/心得/参考文献都已写好，只剩**结果数字与对比图**待填：
- 论文表 1 / PPT 第 9 页的三行（LR / SVM / U-Net）→ 用 `results_table.csv` 替换占位 `—`。
- 论文图 4–图 7、PPT 结果页占位 → 插入 `figures/` 下的 `fig2_feat_importance / fig3_training_curve / fig4_roc_pr / fig6_qualitative`。
- 论文三个混淆矩阵 → 用 `results.json` 的 `confusion` 字典（按方法名索引）。
- 论文图 1–3（不平衡分布、预处理、候选检测）已是真实图，无需改动。

---

## 五、调参与常见问题

- **调超参**：notebook 第 3 格 `ex.configure(EPOCHS=80, BATCH=4, UNET_ENCODER="efficientnet-b3", SVM_MAX=20000)`。
- **找不到 exudate 包**：确认 `exudate/` 已 push 到 `topic/眼底图像的渗出液分割/` 下。
- **找不到数据**：检查第 2 格的 `REPO_URL` 与 `BASE`。
- **显存不足**：`ex.configure(BATCH=4)` 或调小 `WORK_W/WORK_H`。
- **SVM 很慢**：RBF-SVM 在像素级最慢；`ex.configure(SVM_MAX=15000)`，或把 `classical.make_classifier` 的核改为线性。
- **U-Net 比传统方法差**：多半是没开 GPU 或 epoch 太少；确认 GPU 运行时且预训练权重已下载。

---

## 六、方法概要（便于讲解）

- **逻辑回归 (LR)**：14 维手工特征 + StandardScaler → 线性分类器，FOV 内稠密像素分类，自带概率、可解释。
- **支持向量机 SVM (RBF)**：非线性边界；因 O(n²) 复杂度，训练集下采样至约 2.5 万像素，Platt 缩放得概率。
- **深度学习 U-Net**：ImageNet 预训练 ResNet34 编码器；BCE+Tversky 组合损失对抗类别不平衡；增强 + 余弦退火 + 混合精度。
- **统一评价**：三种方法共享同一预处理、同一 32/15 划分、同一指标计算（混淆矩阵 / 准召 / ROC/PR / IoU/Dice + 学习时间），保证对比公平。
