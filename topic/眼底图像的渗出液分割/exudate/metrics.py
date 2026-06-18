"""共享评价模块（三种方法同一口径，仅统计FOV内像素）。"""
import numpy as np
from sklearn.metrics import (confusion_matrix, roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve)


def _flat(prob, gt, fov):
    P, Y = [], []
    for pr, g, f in zip(prob, gt, fov):
        m = f > 0
        P.append(pr[m].ravel()); Y.append((g[m] > 0).astype(np.uint8).ravel())
    return np.concatenate(P), np.concatenate(Y)


def pixel_metrics(prob, gt, fov, thr=0.5):
    p, y = _flat(prob, gt, fov); pred = (p >= thr).astype(np.uint8)
    cm = confusion_matrix(y, pred, labels=[0, 1]); tn, fp, fn, tp = cm.ravel().astype(float); e = 1e-8
    return {"cm": cm, "accuracy": (tp + tn) / (tp + tn + fp + fn + e), "precision": tp / (tp + fp + e),
            "recall": tp / (tp + fn + e), "specificity": tn / (tn + fp + e),
            "f1": 2 * tp / (2 * tp + fp + fn + e), "iou": tp / (tp + fp + fn + e),
            "dice": 2 * tp / (2 * tp + fp + fn + e)}


def curve_metrics(prob, gt, fov, maxp=600000):
    p, y = _flat(prob, gt, fov); pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    if len(p) > maxp:
        kn = np.random.RandomState(0).choice(neg, min(len(neg), maxp - len(pos)), replace=False)
        idx = np.concatenate([pos, kn]); p, y = p[idx], y[idx]
    fpr, tpr, _ = roc_curve(y, p); prec, rec, _ = precision_recall_curve(y, p)
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "roc": (fpr, tpr), "pr": (rec, prec)}


def best_threshold(prob, gt, fov):
    bt, bf = 0.5, -1
    for t in np.linspace(0.05, 0.95, 19):
        f = pixel_metrics(prob, gt, fov, t)["f1"]
        if f > bf: bf, bt = f, t
    return float(bt)
