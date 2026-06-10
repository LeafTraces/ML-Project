import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def flat_pixels(prob, gt, fov):
    probs, labels = [], []
    for pr, g, f in zip(prob, gt, fov):
        mask = f > 0
        probs.append(pr[mask].ravel())
        labels.append((g[mask] > 0).astype(np.uint8).ravel())
    return np.concatenate(probs), np.concatenate(labels)


def pixel_metrics(prob, gt, fov, thr=0.5):
    p, y = flat_pixels(prob, gt, fov)
    pred = (p >= thr).astype(np.uint8)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().astype(float)
    eps = 1e-8
    return {
        "cm": cm,
        "accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "specificity": tn / (tn + fp + eps),
        "f1": 2 * tp / (2 * tp + fp + fn + eps),
        "iou": tp / (tp + fp + fn + eps),
        "dice": 2 * tp / (2 * tp + fp + fn + eps),
    }


def curve_metrics(prob, gt, fov, maxp=600000):
    p, y = flat_pixels(prob, gt, fov)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(p) > maxp:
        keep_neg = np.random.RandomState(0).choice(
            neg, min(len(neg), maxp - len(pos)), replace=False
        )
        idx = np.concatenate([pos, keep_neg])
        p, y = p[idx], y[idx]
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "roc": (fpr, tpr),
        "pr": (rec, prec),
    }


def best_threshold(prob, gt, fov):
    best_t, best_f = 0.5, -1
    for threshold in np.linspace(0.05, 0.95, 19):
        score = pixel_metrics(prob, gt, fov, threshold)["f1"]
        if score > best_f:
            best_f, best_t = score, threshold
    return float(best_t)

