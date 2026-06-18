"""深度学习方法：U-Net（ImageNet 预训练编码器，迁移学习）。

默认用 segmentation-models-pytorch 的预训练编码器；若不可用则退回内置轻量U-Net。
损失：BCE + Tversky 组合，对抗极端类别不平衡。
"""
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from . import config, metrics


# --------------------------- 数据集与损失 ---------------------------
class SegDS(Dataset):
    def __init__(self, samples, aug=False):
        self.s = samples; self.aug = aug

    def __len__(self): return len(self.s)

    def __getitem__(self, i):
        s = self.s[i]
        img = s["rgb"].astype(np.float32) / 255.
        m = s["mask"].astype(np.float32); f = (s["fov"] > 0).astype(np.float32)
        if self.aug:
            if np.random.rand() < .5: img, m, f = img[:, ::-1], m[:, ::-1], f[:, ::-1]
            if np.random.rand() < .5: img, m, f = img[::-1], m[::-1], f[::-1]
            if np.random.rand() < .5: img, m, f = np.rot90(img, 2), np.rot90(m, 2), np.rot90(f, 2)
            if np.random.rand() < .5: img = np.clip(img * np.random.uniform(.85, 1.15), 0, 1)
        img = (img - config.IMAGENET_MEAN) / config.IMAGENET_STD
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        return (torch.from_numpy(img).float(),
                torch.from_numpy(np.ascontiguousarray(m)[None]).float(),
                torch.from_numpy(np.ascontiguousarray(f)[None]).float())


class TverskyLoss(nn.Module):
    def __init__(self, a=.3, b=.7, s=1.): super().__init__(); self.a, self.b, self.s = a, b, s
    def forward(self, logit, t):
        p = torch.sigmoid(logit).reshape(logit.size(0), -1); t = t.reshape(t.size(0), -1)
        tp = (p * t).sum(1); fp = (p * (1 - t)).sum(1); fn = ((1 - p) * t).sum(1)
        return (1 - (tp + self.s) / (tp + self.a * fp + self.b * fn + self.s)).mean()


class ComboLoss(nn.Module):
    def __init__(self, pw=10., a=.3, b=.7, w=.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw)); self.tv = TverskyLoss(a, b); self.w = w
    def forward(self, l, t): return self.w * self.bce(l, t) + (1 - self.w) * self.tv(l, t)


# --------------------------- 可复现 ---------------------------
def seed_everything(seed=42):
    """固定随机种子：确保消融对比中两次训练除“编码器初始权重”外完全一致。"""
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------- 模型 ---------------------------
def build_unet(encoder=None, encoder_weights="imagenet"):
    encoder = encoder or config.UNET_ENCODER
    try:
        import segmentation_models_pytorch as smp
        return smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights,
                        in_channels=3, classes=1)
    except Exception as e:
        print(f"[警告] 未能创建 smp U-Net ({e})，改用内置轻量 U-Net。")
        from ._fallback_unet import UNet
        return UNet(base_ch=32, depth=4)


# --------------------------- 推理与验证 ---------------------------
def _device(): return "cuda" if torch.cuda.is_available() else "cpu"


def unet_predict(model, s, device=None):
    device = device or next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x = ((s["rgb"].astype(np.float32) / 255. - config.IMAGENET_MEAN) / config.IMAGENET_STD).transpose(2, 0, 1)
        p = torch.sigmoid(model(torch.from_numpy(x[None]).float().to(device))).cpu().numpy()[0, 0]
    return (p * (s["fov"] > 0)).astype(np.float32)


def _val_dice(model, samples, device):
    probs = [(unet_predict(model, s, device), s["mask"] > 0, s["fov"] > 0) for s in samples]
    best = 0.
    for t in np.linspace(0.1, 0.9, 17):
        inter = union = 0.
        for p, m, f in probs:
            pb = (p >= t) & f; inter += (pb & m).sum(); union += pb.sum() + m.sum()
        best = max(best, 2 * inter / (union + 1e-8))
    return float(best)


# --------------------------- 训练 ---------------------------
def run_unet(data, encoder=None, epochs=None, batch=None, lr=None,
             encoder_weights="imagenet", seed=None, verbose=True):
    encoder = encoder or config.UNET_ENCODER
    epochs = epochs or config.EPOCHS; batch = batch or config.BATCH; lr = lr or config.LR
    if seed is not None:
        seed_everything(seed)
    device = _device()
    model = build_unet(encoder, encoder_weights).to(device)
    crit = ComboLoss(pw=10.).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    _wi = (lambda wid: np.random.seed((seed + wid) % (2 ** 31 - 1))) if seed is not None else None
    loader = DataLoader(SegDS(data["fit"], aug=True), batch_size=batch, shuffle=True,
                        num_workers=2, worker_init_fn=_wi)
    use_amp = (device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    hist = []; best = -1; best_state = None; t0 = time.time()
    for ep in range(epochs):
        model.train(); rl = 0.
        for img, m, _ in loader:
            img, m = img.to(device), m.to(device); opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = crit(model(img), m)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); rl += loss.item() * img.size(0)
        sched.step(); vd = _val_dice(model, data["val"], device); hist.append((rl / len(data["fit"]), vd))
        if vd > best: best = vd; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"epoch {ep:>2d}  loss {rl/len(data['fit']):.4f}  val_dice {vd:.4f}  best {best:.4f}")
    train_time = time.time() - t0
    if best_state: model.load_state_dict(best_state)
    vprob = [unet_predict(model, s, device) for s in data["val"]]
    thr = metrics.best_threshold(vprob, data["val_gt"], data["val_fov"])
    tprob = [unet_predict(model, s, device) for s in data["test"]]
    m = metrics.pixel_metrics(tprob, data["test_gt"], data["test_fov"], thr)
    c = metrics.curve_metrics(tprob, data["test_gt"], data["test_fov"])
    if verbose:
        print(f"[U-Net] 学习时间{train_time:.1f}s 阈值{thr:.2f} "
              f"Dice {m['dice']:.3f} Recall {m['recall']:.3f} ROC-AUC {c['roc_auc']:.3f} PR-AUC {c['pr_auc']:.3f}")
    return {"name": f"U-Net ({encoder})", "model": model, "time": train_time, "thr": thr,
            "test_prob": tprob, "m": m, "c": c, "hist": hist}
