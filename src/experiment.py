import json
import os
import random

import numpy as np
import pandas as pd
import torch

from .exudate_data import make_split
from .metrics import best_threshold, curve_metrics, pixel_metrics
from .plotting import (
    plot_confusion,
    plot_linear_coefficients,
    plot_preprocess,
    plot_qualitative,
    plot_roc_pr,
    plot_training_curve,
)
from .traditional_ml import linear_feature_weights, predict_traditional, train_traditional_models
from .unet_model import train_unet, unet_predict


def prepare_environment(seed, out_dir="outputs"):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return out_dir, fig_dir


def run_experiment(
    data_root,
    seed=42,
    work_size=(672, 448),
    val_ratio=0.15,
    unet_encoder="resnet34",
    epochs=35,
    batch=6,
    lr=2e-4,
    out_dir="outputs",
):
    out_dir, fig_dir = prepare_environment(seed, out_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE:", device)

    fit_ids, val_ids, test_ids, fit_s, val_s, test_s = make_split(data_root, seed, val_ratio, work_size)
    print(f"fit {len(fit_ids)} / val {len(val_ids)} / test {len(test_ids)}")
    foreground = np.array([s["mask"].mean() for s in fit_s + val_s + test_s]) * 100
    print(
        f"前景占比: mean {foreground.mean():.4f}%  median {np.median(foreground):.4f}%  "
        f"max {foreground.max():.4f}%  —— 极端类别不平衡"
    )

    test_gt = [s["mask"] for s in test_s]
    test_fov = [s["fov"] for s in test_s]
    val_gt = [s["mask"] for s in val_s]
    val_fov = [s["fov"] for s in val_s]

    plot_preprocess(max(test_s, key=lambda x: x["mask"].mean()), os.path.join(fig_dir, "fig1_preprocess.png"))

    traditional = train_traditional_models(fit_s, seed=seed)
    print(
        f"候选检测召回上限: {traditional.candidate_recall:.3f}  "
        f"训练样本 {traditional.train_shape}  正例率 {traditional.positive_rate:.3f}"
    )

    traditional_runs = []
    for model in traditional.models:
        val_prob = [predict_traditional(model, s) for s in val_s]
        threshold = best_threshold(val_prob, val_gt, val_fov)
        test_prob = [predict_traditional(model, s) for s in test_s]
        metrics = pixel_metrics(test_prob, test_gt, test_fov, threshold)
        curves = curve_metrics(test_prob, test_gt, test_fov)
        traditional_runs.append(
            {
                "name": model.name,
                "model": model,
                "threshold": threshold,
                "test_prob": test_prob,
                "metrics": metrics,
                "curves": curves,
            }
        )
        print(
            f"[方法一-{model.name}] 学习时间 {model.learn_time:.1f}s 阈值{threshold:.2f} "
            f"Dice {metrics['dice']:.3f} IoU {metrics['iou']:.3f} "
            f"Recall {metrics['recall']:.3f} ROC-AUC {curves['roc_auc']:.3f} PR-AUC {curves['pr_auc']:.3f}"
        )
    plot_linear_coefficients(traditional.models, traditional.feature_names, os.path.join(fig_dir, "fig2_feat_coefficients.png"))

    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    model, history, unet_time, best = train_unet(
        fit_s,
        val_s,
        encoder=unet_encoder,
        epochs=epochs,
        batch=batch,
        lr=lr,
        device=device,
        imagenet_mean=imagenet_mean,
        imagenet_std=imagenet_std,
    )
    print(f"U-Net 学习时间 {unet_time:.1f}s  最佳 val_dice {best:.4f}")
    un_val = [unet_predict(model, s, device, imagenet_mean, imagenet_std) for s in val_s]
    un_threshold = best_threshold(un_val, val_gt, val_fov)
    un_test = [unet_predict(model, s, device, imagenet_mean, imagenet_std) for s in test_s]
    un_metrics = pixel_metrics(un_test, test_gt, test_fov, un_threshold)
    un_curves = curve_metrics(un_test, test_gt, test_fov)
    unet_run = {
        "name": f"U-Net ({unet_encoder})",
        "threshold": un_threshold,
        "test_prob": un_test,
        "metrics": un_metrics,
        "curves": un_curves,
        "learn_time": unet_time,
    }
    print(
        f"[方法二-UNet] 阈值{un_threshold:.2f} Dice {un_metrics['dice']:.3f} "
        f"IoU {un_metrics['iou']:.3f} Recall {un_metrics['recall']:.3f} "
        f"ROC-AUC {un_curves['roc_auc']:.3f} PR-AUC {un_curves['pr_auc']:.3f}"
    )

    plot_training_curve(history, os.path.join(fig_dir, "fig3_training_curve.png"))
    plot_roc_pr(traditional_runs, unet_run, os.path.join(fig_dir, "fig4_roc_pr.png"))
    plot_confusion(traditional_runs, unet_run, os.path.join(fig_dir, "fig5_confusion.png"))
    plot_qualitative(test_s, traditional_runs, unet_run, os.path.join(fig_dir, "fig6_qualitative.png"))

    rows = []
    for run in traditional_runs:
        rows.append(_result_row(run["name"] + " (传统ML)", run["metrics"], run["curves"], run["model"].learn_time, run["threshold"]))
    rows.append(_result_row(f"U-Net ({unet_encoder})", un_metrics, un_curves, unet_time, un_threshold))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "results_table.csv"), index=False)

    results = {
        "work_size": work_size,
        "traditional": rows[:-1],
        "unet": rows[-1],
        "confusion": {run["name"]: run["metrics"]["cm"].tolist() for run in traditional_runs},
        "unet_confusion": un_metrics["cm"].tolist(),
        "feature_coefficients": {
            model.name: dict(zip(traditional.feature_names, np.abs(linear_feature_weights(model)).tolist()))
            for model in traditional.models
        },
    }
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("已保存 outputs/results.json 与 results_table.csv\n")
    display_df = df.set_index("Method").T
    try:
        from IPython.display import display

        display(display_df)
    except Exception:
        print(display_df)
    return df, results


def _result_row(name, metrics, curves, learn_time, threshold):
    return {
        "Method": name,
        "Learn time(s)": round(learn_time, 1),
        "Threshold": round(threshold, 2),
        "Accuracy": round(metrics["accuracy"], 4),
        "Precision": round(metrics["precision"], 4),
        "Recall": round(metrics["recall"], 4),
        "Specificity": round(metrics["specificity"], 4),
        "F1": round(metrics["f1"], 4),
        "IoU": round(metrics["iou"], 4),
        "Dice": round(metrics["dice"], 4),
        "ROC-AUC": round(curves["roc_auc"], 4),
        "PR-AUC": round(curves["pr_auc"], 4),
    }
