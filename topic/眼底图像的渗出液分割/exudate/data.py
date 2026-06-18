"""数据读取与预处理（与已验证的流程一致）。

唯一数据源：images/(原图) + groudtruth/(掩膜)，完整且1:1对应。
预处理：统一降采样 → FOV视野掩膜 → 亮度归一化(去光照) → CLAHE。
"""
import os
import glob
import cv2
import numpy as np

from . import config


def _stem(p):
    return os.path.splitext(os.path.basename(p))[0].replace("_EX", "").upper()


def build_index(root):
    img = {_stem(p): p for p in glob.glob(os.path.join(root, "images", "*")) if os.path.isfile(p)}
    msk = {_stem(p): p for p in glob.glob(os.path.join(root, "groudtruth", "*")) if os.path.isfile(p)}
    tr = sorted(_stem(p) for p in glob.glob(os.path.join(root, "groudtruth", "训练", "*")))
    te = sorted(_stem(p) for p in glob.glob(os.path.join(root, "groudtruth", "测试", "*")))
    assert set(img) == set(msk), "原图与mask的ID集合不一致"
    return tr, te, {k: (img[k], msk[k]) for k in img}


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
    ff = m.copy(); h, w = m.shape
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    return (m | (1 - ff)).astype(np.uint8)


def remove_illumination(rgb, fov):
    out = np.zeros_like(rgb)
    ks = max(3, (min(rgb.shape[:2]) // 12) | 1)
    for c in range(3):
        ch = rgb[..., c].astype(np.float32)
        bg = cv2.medianBlur(rgb[..., c], ks).astype(np.float32)
        base = float(ch[fov > 0].mean() if (fov > 0).any() else ch.mean())
        out[..., c] = np.clip(ch - bg + base, 0, 255).astype(np.uint8)
    out[fov == 0] = 0
    return out


def apply_clahe(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(2.0, (8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def load_sample(ip, mp, size=None, illum=True, clahe=True):
    size = size or config.img_size()
    rgb = cv2.cvtColor(cv2.imread(ip, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
    rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    m = cv2.resize(m, size, interpolation=cv2.INTER_NEAREST)
    fov = compute_fov_mask(rgb); raw = rgb.copy()
    if illum: rgb = remove_illumination(rgb, fov)
    if clahe: rgb = apply_clahe(rgb)
    rgb[fov == 0] = 0
    return {"rgb": rgb, "raw": raw, "mask": (m > 127).astype(np.uint8), "fov": fov}


def load_split(root, ids, **kw):
    _, _, idx = build_index(root)
    out = []
    for i in ids:
        s = load_sample(*idx[i], **kw); s["id"] = i; out.append(s)
    return out


def load_data(data_root, verbose=True):
    """加载并划分 fit/val/test，返回一个 data 字典。"""
    train_ids, test_ids, _ = build_index(data_root)
    rng = np.random.RandomState(config.SEED)
    t = list(train_ids); rng.shuffle(t)
    nval = max(4, int(len(t) * config.VAL_RATIO))
    val_ids, fit_ids = t[:nval], t[nval:]
    data = {
        "root": data_root,
        "fit": load_split(data_root, fit_ids),
        "val": load_split(data_root, val_ids),
        "test": load_split(data_root, test_ids),
        "fit_ids": fit_ids, "val_ids": val_ids, "test_ids": test_ids,
    }
    data["val_gt"] = [s["mask"] for s in data["val"]]
    data["val_fov"] = [s["fov"] for s in data["val"]]
    data["test_gt"] = [s["mask"] for s in data["test"]]
    data["test_fov"] = [s["fov"] for s in data["test"]]
    if verbose:
        fg = np.array([s["mask"].mean() for s in data["fit"] + data["val"] + data["test"]]) * 100
        print(f"fit {len(fit_ids)} / val {len(val_ids)} / test {len(test_ids)}")
        print(f"前景占比: mean {fg.mean():.4f}%  median {np.median(fg):.4f}%  max {fg.max():.4f}%  —— 极端类别不平衡")
    return data
