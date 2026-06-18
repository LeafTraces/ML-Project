"""
U-Net（可配置宽度/深度的轻量实现，无外部预训练依赖，沙箱与Colab同一份代码）。
Colab上可把 base_ch 调大(如48/64)、img尺寸调大以获得更强性能。
带 BatchNorm + 双卷积 + 转置卷积上采样 + 跳跃连接。
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x): return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch=3, n_classes=1, base_ch=32, depth=4):
        super().__init__()
        self.depth = depth
        chs = [base_ch * (2 ** i) for i in range(depth + 1)]   # e.g. 32,64,128,256,512
        # 编码器
        self.downs = nn.ModuleList()
        prev = in_ch
        for c in chs[:-1]:
            self.downs.append(DoubleConv(prev, c)); prev = c
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(chs[-2], chs[-1])
        # 解码器
        self.ups = nn.ModuleList(); self.dec = nn.ModuleList()
        for i in range(depth):
            cin = chs[-1 - i]; cout = chs[-2 - i]
            self.ups.append(nn.ConvTranspose2d(cin, cout, 2, stride=2))
            self.dec.append(DoubleConv(cout * 2, cout))
        self.head = nn.Conv2d(chs[0], n_classes, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x)
        for i in range(self.depth):
            x = self.ups[i](x)
            skip = skips[-1 - i]
            # 尺寸对齐(应对奇数尺寸)
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:],
                                              mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = self.dec[i](x)
        return self.head(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6
