import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np

from .traditional_ml import linear_feature_weights


def plot_preprocess(sample, figure_path):
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    ax[0].imshow(sample["raw"])
    ax[0].set_title("Raw (resized)")
    ax[1].imshow(sample["rgb"])
    ax[1].set_title("Preprocessed (illum-norm + CLAHE)")
    ax[2].imshow(sample["mask"], cmap="gray")
    ax[2].set_title("Ground truth")
    overlay = sample["raw"].copy()
    overlay[sample["mask"] > 0] = [255, 0, 0]
    ax[3].imshow(overlay)
    ax[3].set_title("Overlay")
    for axis in ax:
        axis.axis("off")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=120, bbox_inches="tight")
    plt.show()


def plot_linear_coefficients(models, feature_names, figure_path):
    weights = []
    labels = []
    for model in models:
        weights.append(np.abs(linear_feature_weights(model)))
        labels.append(model.name)
    x = np.arange(len(feature_names))
    width = 0.35
    plt.figure(figsize=(9, 4))
    for i, weight in enumerate(weights):
        plt.bar(x + (i - 0.5) * width, weight, width=width, label=labels[i])
    plt.xticks(x, feature_names, rotation=45, ha="right")
    plt.ylabel("|standardized coefficient|")
    plt.title("SVM/LR feature coefficients")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=120, bbox_inches="tight")
    plt.show()


def plot_training_curve(history, figure_path):
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()
    ax1.plot(history[:, 0], color="#C44E52", label="train loss")
    ax2.plot(history[:, 1], color="#4C72B0", label="val Dice")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss", color="#C44E52")
    ax2.set_ylabel("val Dice", color="#4C72B0")
    plt.title("U-Net training")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=120, bbox_inches="tight")
    plt.show()


def plot_roc_pr(traditional_runs, unet_run, figure_path):
    colors = ["#4C72B0", "#55A868"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for run, color in zip(traditional_runs, colors):
        curves = run["curves"]
        ax[0].plot(*curves["roc"], label=f"{run['name']} (AUC={curves['roc_auc']:.3f})", color=color)
        ax[1].plot(*curves["pr"], label=f"{run['name']} (AP={curves['pr_auc']:.3f})", color=color)
    ax[0].plot(*unet_run["curves"]["roc"], label=f"U-Net (AUC={unet_run['curves']['roc_auc']:.3f})", color="#C44E52")
    ax[1].plot(*unet_run["curves"]["pr"], label=f"U-Net (AP={unet_run['curves']['pr_auc']:.3f})", color="#C44E52")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set_xlabel("FPR")
    ax[0].set_ylabel("TPR")
    ax[0].set_title("ROC")
    ax[0].legend()
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].set_title("Precision-Recall")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=120, bbox_inches="tight")
    plt.show()


def _plot_cm(ax, cm, title):
    ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred -", "Pred +"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True -", "True +"])
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )


def plot_confusion(traditional_runs, unet_run, figure_path):
    cols = len(traditional_runs) + 1
    fig, ax = plt.subplots(1, cols, figsize=(5.5 * cols, 4.5))
    for i, run in enumerate(traditional_runs):
        _plot_cm(ax[i], run["metrics"]["cm"], f"{run['name']} (thr={run['threshold']:.2f})")
    _plot_cm(ax[-1], unet_run["metrics"]["cm"], f"U-Net (thr={unet_run['threshold']:.2f})")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=120, bbox_inches="tight")
    plt.show()


def plot_qualitative(test_s, traditional_runs, unet_run, figure_path):
    order = sorted(range(len(test_s)), key=lambda i: -test_s[i]["mask"].mean())[:4]
    titles = ["Image", "Ground truth"] + [f"{run['name']} prediction" for run in traditional_runs] + ["U-Net prediction"]
    fig, ax = plt.subplots(len(order), len(titles), figsize=(4 * len(titles), 3.6 * len(order)))
    for r, i in enumerate(order):
        sample = test_s[i]
        ax[r, 0].imshow(sample["raw"])
        overlay = sample["raw"].copy()
        overlay[sample["mask"] > 0] = [0, 255, 0]
        ax[r, 1].imshow(overlay)
        col = 2
        for run in traditional_runs:
            pred = sample["raw"].copy()
            pred[run["test_prob"][i] >= run["threshold"]] = [255, 0, 0]
            ax[r, col].imshow(pred)
            col += 1
        pred = sample["raw"].copy()
        pred[unet_run["test_prob"][i] >= unet_run["threshold"]] = [255, 0, 0]
        ax[r, col].imshow(pred)
        for c in range(len(titles)):
            ax[r, c].axis("off")
        ax[r, 0].set_ylabel(sample["id"], rotation=0, labelpad=30)
    for c, title in enumerate(titles):
        ax[0, c].set_title(title)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=120, bbox_inches="tight")
    plt.show()
