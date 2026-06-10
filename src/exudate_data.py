import glob
import os

import cv2
import numpy as np


def _stem(path):
    return os.path.splitext(os.path.basename(path))[0].replace("_EX", "").upper()


def build_index(root):
    img = {
        _stem(p): p
        for p in glob.glob(os.path.join(root, "images", "*"))
        if os.path.isfile(p)
    }
    msk = {
        _stem(p): p
        for p in glob.glob(os.path.join(root, "groudtruth", "*"))
        if os.path.isfile(p)
    }
    tr = sorted(_stem(p) for p in glob.glob(os.path.join(root, "groudtruth", "训练", "*")))
    te = sorted(_stem(p) for p in glob.glob(os.path.join(root, "groudtruth", "测试", "*")))
    return tr, te, {k: (img[k], msk[k]) for k in img if k in msk}


def compute_fov_mask(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    thr = max(8, int(0.5 * g[g > 0].mean())) if (g > 0).any() else 8
    m = (g > thr).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        m = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    ff = m.copy()
    h, w = m.shape
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    return (m | (1 - ff)).astype(np.uint8)


def remove_illumination(rgb, fov):
    out = np.zeros_like(rgb)
    ks = max(3, (min(rgb.shape[:2]) // 12) | 1)
    for c in range(3):
        ch = rgb[..., c].astype(np.float32)
        bg = cv2.medianBlur(rgb[..., c], ks).astype(np.float32)
        center = float(ch[fov > 0].mean() if (fov > 0).any() else ch.mean())
        out[..., c] = np.clip(ch - bg + center, 0, 255).astype(np.uint8)
    out[fov == 0] = 0
    return out


def apply_clahe(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(2.0, (8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def load_sample(image_path, mask_path, size, illum=True, clahe=True):
    rgb = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    fov = compute_fov_mask(rgb)
    raw = rgb.copy()
    if illum:
        rgb = remove_illumination(rgb, fov)
    if clahe:
        rgb = apply_clahe(rgb)
    rgb[fov == 0] = 0
    return {"rgb": rgb, "raw": raw, "mask": (mask > 127).astype(np.uint8), "fov": fov}


def load_split(root, ids, size, **kwargs):
    _, _, idx = build_index(root)
    out = []
    for image_id in ids:
        if image_id not in idx:
            raise FileNotFoundError(f"Missing paired image/mask for {image_id}")
        sample = load_sample(*idx[image_id], size=size, **kwargs)
        sample["id"] = image_id
        out.append(sample)
    return out


def make_split(root, seed, val_ratio, size):
    train_ids, test_ids, _ = build_index(root)
    rng = np.random.RandomState(seed)
    shuffled = list(train_ids)
    rng.shuffle(shuffled)
    nval = max(4, int(len(shuffled) * val_ratio))
    val_ids, fit_ids = shuffled[:nval], shuffled[nval:]
    fit_s = load_split(root, fit_ids, size=size)
    val_s = load_split(root, val_ids, size=size)
    test_s = load_split(root, test_ids, size=size)
    return fit_ids, val_ids, test_ids, fit_s, val_s, test_s

