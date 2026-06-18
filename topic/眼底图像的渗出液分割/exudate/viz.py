"""绘图模块：生成论文/PPT 所需的全部图表（保存 + 在notebook内联显示）。
图内文字用英文以保证 Colab 无中文字体时也不乱码。
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from . import classical

COLORS = ["#4C72B0", "#55A868", "#C44E52"]


def _save(fig_dir, name):
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, name), dpi=120, bbox_inches="tight")


def fig_preprocess(data, fig_dir):
    s = max(data["test"], key=lambda x: x["mask"].mean())
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    ax[0].imshow(s["raw"]); ax[0].set_title("Raw (resized)")
    ax[1].imshow(s["rgb"]); ax[1].set_title("Preprocessed (illum-norm + CLAHE)")
    ax[2].imshow(s["mask"], cmap="gray"); ax[2].set_title("Ground truth")
    ov = s["raw"].copy(); ov[s["mask"] > 0] = [255, 0, 0]; ax[3].imshow(ov); ax[3].set_title("Overlay")
    for a in ax: a.axis("off")
    plt.tight_layout(); _save(fig_dir, "fig1_preprocess.png"); plt.show()


def fig_feature_coef(classical_results, fig_dir):
    imp = list(classical.lr_feature_coef(classical_results).items())
    plt.figure(figsize=(8, 4)); plt.bar([n for n, _ in imp], [v for _, v in imp], color="#4C72B0")
    plt.xticks(rotation=45, ha="right"); plt.ylabel("|coef| (standardized)")
    plt.title("Logistic Regression feature coefficients")
    plt.tight_layout(); _save(fig_dir, "fig2_feat_importance.png"); plt.show()


def fig_training_curve(unet_result, fig_dir):
    h = np.array(unet_result["hist"])
    fig, ax1 = plt.subplots(figsize=(7, 4)); ax2 = ax1.twinx()
    ax1.plot(h[:, 0], color="#C44E52", label="train loss"); ax2.plot(h[:, 1], color="#4C72B0", label="val Dice")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss", color="#C44E52"); ax2.set_ylabel("val Dice", color="#4C72B0")
    plt.title("U-Net training"); plt.tight_layout(); _save(fig_dir, "fig3_training_curve.png"); plt.show()


def fig_roc_pr(ALL, fig_dir):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for (name, _, m, c, tt, thr), col in zip(ALL, COLORS):
        ax[0].plot(*c["roc"], label=f"{name} (AUC={c['roc_auc']:.3f})", color=col)
        ax[1].plot(*c["pr"], label=f"{name} (AP={c['pr_auc']:.3f})", color=col)
    ax[0].plot([0, 1], [0, 1], "k--", lw=.8)
    ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].set_title("ROC"); ax[0].legend()
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_title("Precision-Recall"); ax[1].legend()
    plt.tight_layout(); _save(fig_dir, "fig4_roc_pr.png"); plt.show()


def fig_confusion(ALL, fig_dir):
    def plot_cm(ax, cm, title):
        ax.imshow(cm, cmap="Blues"); ax.set_title(title, fontsize=11)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred -", "Pred +"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["True -", "True +"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center", fontsize=9,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig, ax = plt.subplots(1, len(ALL), figsize=(5 * len(ALL), 4.2))
    if len(ALL) == 1: ax = [ax]
    for a, (name, _, m, c, tt, thr) in zip(ax, ALL): plot_cm(a, m["cm"], f"{name} (thr={thr:.2f})")
    plt.tight_layout(); _save(fig_dir, "fig5_confusion.png"); plt.show()


def fig_ablation(pre, rnd, fig_dir):
    """消融对比图：左=关键测试指标柱状对比，右=验证集Dice收敛曲线（看预训练是否加速收敛）。"""
    metric_keys = [("dice", "Dice"), ("iou", "IoU"), ("recall", "Recall")]
    labels = [lab for _, lab in metric_keys] + ["PR-AUC"]
    pre_vals = [pre["m"][k] for k, _ in metric_keys] + [pre["c"]["pr_auc"]]
    rnd_vals = [rnd["m"][k] for k, _ in metric_keys] + [rnd["c"]["pr_auc"]]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(labels)); w = 0.36
    ax[0].bar(x - w / 2, pre_vals, w, label="ImageNet pretrained", color="#4C72B0")
    ax[0].bar(x + w / 2, rnd_vals, w, label="Random init", color="#C44E52")
    for i, (pv, rv) in enumerate(zip(pre_vals, rnd_vals)):
        ax[0].text(i - w / 2, pv + 0.004, f"{pv:.3f}", ha="center", va="bottom", fontsize=8)
        ax[0].text(i + w / 2, rv + 0.004, f"{rv:.3f}", ha="center", va="bottom", fontsize=8)
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels); ax[0].set_ylabel("Test score")
    ax[0].set_title("(a) Pretrained vs Random init - test metrics"); ax[0].legend()
    hp = np.array(pre["hist"]); hr = np.array(rnd["hist"])
    ax[1].plot(hp[:, 1], color="#4C72B0", label="ImageNet pretrained")
    ax[1].plot(hr[:, 1], color="#C44E52", label="Random init")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("val Dice")
    ax[1].set_title("(b) Validation Dice vs epoch (convergence)"); ax[1].legend()
    plt.tight_layout(); _save(fig_dir, "fig7_ablation.png"); plt.show()


def fig_qualitative(data, ALL, fig_dir, n=4):
    test = data["test"]
    order = sorted(range(len(test)), key=lambda i: -test[i]["mask"].mean())[:n]
    ncol = 2 + len(ALL)
    fig, ax = plt.subplots(len(order), ncol, figsize=(3.2 * ncol, 3.4 * len(order)))
    for r, i in enumerate(order):
        s = test[i]
        ax[r, 0].imshow(s["raw"]); ov = s["raw"].copy(); ov[s["mask"] > 0] = [0, 255, 0]; ax[r, 1].imshow(ov)
        for k, (name, prob, m, c, tt, thr) in enumerate(ALL):
            pp = s["raw"].copy(); pp[prob[i] >= thr] = [255, 0, 0]; ax[r, 2 + k].imshow(pp)
        for c2 in range(ncol): ax[r, c2].axis("off")
    for c2, t in enumerate(["Image", "Ground truth"] + [a[0] for a in ALL]): ax[0, c2].set_title(t, fontsize=11)
    plt.tight_layout(); _save(fig_dir, "fig6_qualitative.png"); plt.show()
