"""传统机器学习方法：逻辑回归 与 RBF-SVM。

共享 14 维手工特征 + StandardScaler；FOV 内稠密像素分类。
逻辑回归用全量像素；RBF-SVM 因 O(n²) 复杂度需下采样训练集。
"""
import time
import cv2
import numpy as np
from scipy.ndimage import uniform_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from . import config, metrics

FEATURE_NAMES = ["R", "G", "B", "L", "a", "b", "S", "V",
                 "contrast9", "contrast25", "localStd", "tophat9", "tophat21", "gradMag"]
CLF_NAMES = {"logreg": "Logistic Regression", "svm": "SVM (RBF)", "rf": "Random Forest"}


def feature_maps(rgb):
    f = rgb.astype(np.float32) / 255.; R, G, B = f[..., 0], f[..., 1], f[..., 2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, A, Bb = lab[..., 0] / 255, lab[..., 1] / 255, lab[..., 2] / 255
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32); S_, V_ = hsv[..., 1] / 255, hsv[..., 2] / 255
    feats = [R, G, B, L, A, Bb, S_, V_]
    for k in (9, 25): feats.append(L - uniform_filter(L, k))
    m = uniform_filter(L, 9); m2 = uniform_filter(L * L, 9); feats.append(np.sqrt(np.maximum(m2 - m * m, 0)))
    inten = lab[..., 0].astype(np.uint8)
    for ks in (9, 21):
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        feats.append(cv2.morphologyEx(inten, cv2.MORPH_TOPHAT, se).astype(np.float32) / 255.)
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, 3); gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, 3)
    feats.append(np.sqrt(gx * gx + gy * gy))
    return np.stack(feats, -1).astype(np.float32)


def candidate_mask(rgb, fov):
    inten = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0]
    th = cv2.morphologyEx(inten, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    v = th[fov > 0]; c = ((th > v.mean() + 1.0 * v.std()) & (fov > 0)).astype(np.uint8)
    return cv2.morphologyEx(c, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))


def cand_recall(samples):
    tp = fn = 0
    for s in samples:
        c = candidate_mask(s["rgb"], s["fov"]); g = s["mask"] > 0
        tp += int((c & g).sum()); fn += int((g & (c == 0)).sum())
    return tp / (tp + fn + 1e-8)


def build_training_set(samples, neg_per_pos=15, max_samples=None, seed=None):
    seed = config.SEED if seed is None else seed
    rng = np.random.RandomState(seed); X, Y = [], []
    for s in samples:
        fm = feature_maps(s["rgb"]); fov = s["fov"] > 0; lab = s["mask"][fov]; ft = fm[fov]
        pos = np.where(lab == 1)[0]; neg = np.where(lab == 0)[0]
        nn = min(len(neg), max(1, neg_per_pos * max(len(pos), 1)))
        if len(neg) > nn: neg = rng.choice(neg, nn, replace=False)
        sel = np.concatenate([pos, neg]); X.append(ft[sel]); Y.append(lab[sel])
    X = np.concatenate(X); Y = np.concatenate(Y)
    if max_samples and len(Y) > max_samples:
        idx = rng.choice(len(Y), max_samples, replace=False); X, Y = X[idx], Y[idx]
    return X, Y


def make_classifier(kind):
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0))
    if kind == "svm":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced", probability=True,
                                                   C=2.0, gamma="scale", cache_size=1000, random_state=config.SEED))
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=4, n_jobs=-1,
                                  class_weight="balanced", random_state=config.SEED)


def clf_predict(model, s):
    prob = np.zeros(s["mask"].shape, np.float32); fov = s["fov"] > 0
    if fov.any(): prob[fov] = model.predict_proba(feature_maps(s["rgb"])[fov])[:, 1]
    return prob


def run_classical(data, kinds=("logreg", "svm"), verbose=True):
    """训练所选传统方法，返回 {kind: {name,model,time,thr,test_prob,m,c}}。"""
    if verbose:
        print(f"候选检测召回上限: {cand_recall(data['fit']):.3f}\n")
    out = {}
    for kind in kinds:
        Xtr, Ytr = build_training_set(data["fit"], max_samples=(config.SVM_MAX if kind == "svm" else None))
        model = make_classifier(kind)
        t0 = time.time(); model.fit(Xtr, Ytr); tt = time.time() - t0
        vprob = [clf_predict(model, s) for s in data["val"]]
        thr = metrics.best_threshold(vprob, data["val_gt"], data["val_fov"])
        tprob = [clf_predict(model, s) for s in data["test"]]
        m = metrics.pixel_metrics(tprob, data["test_gt"], data["test_fov"], thr)
        c = metrics.curve_metrics(tprob, data["test_gt"], data["test_fov"])
        out[kind] = {"name": CLF_NAMES[kind], "model": model, "time": tt, "thr": thr,
                     "test_prob": tprob, "m": m, "c": c}
        if verbose:
            print(f"[{CLF_NAMES[kind]}] 训练{Xtr.shape} 学习时间{tt:.1f}s 阈值{thr:.2f} "
                  f"Dice {m['dice']:.3f} Recall {m['recall']:.3f} ROC-AUC {c['roc_auc']:.3f} PR-AUC {c['pr_auc']:.3f}")
    return out


def lr_feature_coef(classical):
    """逻辑回归标准化系数（绝对值），作为特征重要性度量。"""
    lr = classical["logreg"]["model"].named_steps["logisticregression"]
    return dict(sorted(zip(FEATURE_NAMES, np.abs(lr.coef_[0]).tolist()), key=lambda x: -x[1]))
