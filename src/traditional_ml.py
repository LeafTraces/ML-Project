import time
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import uniform_filter
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


FEATURE_NAMES = [
    "R",
    "G",
    "B",
    "L",
    "a",
    "b",
    "S",
    "V",
    "contrast9",
    "contrast25",
    "localStd",
    "tophat9",
    "tophat21",
    "gradMag",
]


@dataclass
class TraditionalModel:
    name: str
    estimator: object
    learn_time: float


@dataclass
class TraditionalTrainingResult:
    models: list
    feature_names: list
    train_shape: tuple
    positive_rate: float
    candidate_recall: float


def feature_maps(rgb):
    f = rgb.astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l, a, bb = lab[..., 0] / 255, lab[..., 1] / 255, lab[..., 2] / 255
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    s_, v_ = hsv[..., 1] / 255, hsv[..., 2] / 255
    feats = [r, g, b, l, a, bb, s_, v_]
    for k in (9, 25):
        feats.append(l - uniform_filter(l, k))
    m = uniform_filter(l, 9)
    m2 = uniform_filter(l * l, 9)
    feats.append(np.sqrt(np.maximum(m2 - m * m, 0)))
    inten = lab[..., 0].astype(np.uint8)
    for ks in (9, 21):
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        feats.append(cv2.morphologyEx(inten, cv2.MORPH_TOPHAT, se).astype(np.float32) / 255.0)
    gx = cv2.Sobel(l, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(l, cv2.CV_32F, 0, 1, 3)
    feats.append(np.sqrt(gx * gx + gy * gy))
    return np.stack(feats, -1).astype(np.float32)


def candidate_mask(rgb, fov):
    inten = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0]
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    th = cv2.morphologyEx(inten, cv2.MORPH_TOPHAT, se)
    values = th[fov > 0]
    c = ((th > values.mean() + values.std()) & (fov > 0)).astype(np.uint8)
    return cv2.morphologyEx(
        c, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )


def candidate_recall(samples):
    tp = 0
    fn = 0
    for sample in samples:
        cand = candidate_mask(sample["rgb"], sample["fov"])
        gt = sample["mask"] > 0
        tp += int((cand & gt).sum())
        fn += int((gt & (cand == 0)).sum())
    return tp / (tp + fn + 1e-8)


def build_training_set(samples, neg_per_pos=15, seed=42):
    rng = np.random.RandomState(seed)
    features, labels = [], []
    for sample in samples:
        fm = feature_maps(sample["rgb"])
        fov = sample["fov"] > 0
        lab = sample["mask"][fov]
        ft = fm[fov]
        pos = np.where(lab == 1)[0]
        neg = np.where(lab == 0)[0]
        nn = min(len(neg), max(1, neg_per_pos * max(len(pos), 1)))
        if len(neg) > nn:
            neg = rng.choice(neg, nn, replace=False)
        sel = np.concatenate([pos, neg])
        features.append(ft[sel])
        labels.append(lab[sel])
    return np.concatenate(features), np.concatenate(labels)


def _calibrated_linear_svm(seed):
    base = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "svm",
                LinearSVC(
                    C=1.0,
                    class_weight="balanced",
                    dual=False,
                    max_iter=5000,
                    random_state=seed,
                ),
            ),
        ]
    )
    try:
        return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=3)


def _make_models(seed):
    svm = _calibrated_linear_svm(seed)
    lr = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    return [("SVM", svm), ("Logistic Regression", lr)]


def train_traditional_models(samples, seed=42, neg_per_pos=15):
    x_train, y_train = build_training_set(samples, neg_per_pos=neg_per_pos, seed=seed)
    trained = []
    for name, estimator in _make_models(seed):
        t0 = time.time()
        estimator.fit(x_train, y_train)
        trained.append(TraditionalModel(name=name, estimator=estimator, learn_time=time.time() - t0))
    return TraditionalTrainingResult(
        models=trained,
        feature_names=list(FEATURE_NAMES),
        train_shape=x_train.shape,
        positive_rate=float(y_train.mean()),
        candidate_recall=float(candidate_recall(samples)),
    )


def predict_traditional(model, sample):
    prob = np.zeros(sample["mask"].shape, np.float32)
    fov = sample["fov"] > 0
    if not fov.any():
        return prob
    x = feature_maps(sample["rgb"])[fov]
    if hasattr(model.estimator, "predict_proba"):
        score = model.estimator.predict_proba(x)[:, 1]
    else:
        raw = model.estimator.decision_function(x)
        score = 1.0 / (1.0 + np.exp(-np.clip(raw, -50, 50)))
    prob[fov] = score.astype(np.float32)
    return prob


def linear_feature_weights(model):
    estimator = model.estimator
    if model.name == "Logistic Regression":
        return estimator.named_steps["lr"].coef_.ravel()
    if model.name == "SVM":
        calibrated = estimator.calibrated_classifiers_[0]
        base = getattr(calibrated, "estimator", None) or getattr(calibrated, "base_estimator", None)
        return base.named_steps["svm"].coef_.ravel()
    raise ValueError(f"Unsupported model for coefficients: {model.name}")

