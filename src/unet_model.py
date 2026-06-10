import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class SegDS(Dataset):
    def __init__(self, samples, imagenet_mean, imagenet_std, aug=False):
        self.samples = samples
        self.imagenet_mean = imagenet_mean
        self.imagenet_std = imagenet_std
        self.aug = aug

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        img = sample["rgb"].astype(np.float32) / 255.0
        mask = sample["mask"].astype(np.float32)
        fov = (sample["fov"] > 0).astype(np.float32)
        if self.aug:
            if np.random.rand() < 0.5:
                img, mask, fov = img[:, ::-1], mask[:, ::-1], fov[:, ::-1]
            if np.random.rand() < 0.5:
                img, mask, fov = img[::-1], mask[::-1], fov[::-1]
            if np.random.rand() < 0.5:
                img, mask, fov = np.rot90(img, 2), np.rot90(mask, 2), np.rot90(fov, 2)
            if np.random.rand() < 0.5:
                img = np.clip(img * np.random.uniform(0.85, 1.15), 0, 1)
        img = (img - self.imagenet_mean) / self.imagenet_std
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        return (
            torch.from_numpy(img).float(),
            torch.from_numpy(np.ascontiguousarray(mask)[None]).float(),
            torch.from_numpy(np.ascontiguousarray(fov)[None]).float(),
        )


class TverskyLoss(nn.Module):
    def __init__(self, a=0.3, b=0.7, s=1.0):
        super().__init__()
        self.a, self.b, self.s = a, b, s

    def forward(self, logit, target):
        prob = torch.sigmoid(logit).reshape(logit.size(0), -1)
        target = target.reshape(target.size(0), -1)
        tp = (prob * target).sum(1)
        fp = (prob * (1 - target)).sum(1)
        fn = ((1 - prob) * target).sum(1)
        return (1 - (tp + self.s) / (tp + self.a * fp + self.b * fn + self.s)).mean()


class ComboLoss(nn.Module):
    def __init__(self, pos_weight=10.0, a=0.3, b=0.7, w=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        self.tv = TverskyLoss(a, b)
        self.w = w

    def forward(self, logit, target):
        return self.w * self.bce(logit, target) + (1 - self.w) * self.tv(logit, target)


def unet_val_dice(model, samples, device, imagenet_mean, imagenet_std):
    model.eval()
    probs = []
    with torch.no_grad():
        for sample in samples:
            x = ((sample["rgb"].astype(np.float32) / 255.0 - imagenet_mean) / imagenet_std).transpose(2, 0, 1)
            pred = torch.sigmoid(model(torch.from_numpy(x[None]).float().to(device))).cpu().numpy()[0, 0]
            probs.append((pred, sample["mask"] > 0, sample["fov"] > 0))
    best = 0.0
    for threshold in np.linspace(0.1, 0.9, 17):
        inter = 0
        union = 0
        for pred, mask, fov in probs:
            pb = (pred >= threshold) & fov
            inter += (pb & mask).sum()
            union += pb.sum() + mask.sum()
        best = max(best, 2 * inter / (union + 1e-8))
    return best


def unet_predict(model, sample, device, imagenet_mean, imagenet_std):
    model.eval()
    with torch.no_grad():
        x = ((sample["rgb"].astype(np.float32) / 255.0 - imagenet_mean) / imagenet_std).transpose(2, 0, 1)
        pred = torch.sigmoid(model(torch.from_numpy(x[None]).float().to(device))).cpu().numpy()[0, 0]
    return (pred * (sample["fov"] > 0)).astype(np.float32)


def train_unet(
    fit_s,
    val_s,
    encoder,
    epochs,
    batch,
    lr,
    device,
    imagenet_mean,
    imagenet_std,
):
    import segmentation_models_pytorch as smp

    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)
    criterion = ComboLoss(pos_weight=10.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(SegDS(fit_s, imagenet_mean, imagenet_std, aug=True), batch_size=batch, shuffle=True, num_workers=2)
    scaler = torch.cuda.amp.GradScaler()
    hist = []
    best = -1
    best_state = None
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        running_loss = 0
        for img, mask, _ in loader:
            img, mask = img.to(device), mask.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = criterion(model(img), mask)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running_loss += loss.item() * img.size(0)
        sched.step()
        val_dice = unet_val_dice(model, val_s, device, imagenet_mean, imagenet_std)
        hist.append((running_loss / len(fit_s), val_dice))
        if val_dice > best:
            best = val_dice
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"epoch {ep:>2d}  loss {running_loss / len(fit_s):.4f}  val_dice {val_dice:.4f}  best {best:.4f}")
    learn_time = time.time() - t0
    model.load_state_dict(best_state)
    return model, np.array(hist), learn_time, best

