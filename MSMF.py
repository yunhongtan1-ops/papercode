import torch
import torch.nn as nn
from einops import rearrange


# =========================
# SGM：Spatial-Guided Module
# =========================
class SpatialAttention(nn.Module):
    """
    Spatial attention:
    GMP + GAP -> concat -> 5x5 -> 1x1 -> 7x7
    输出 ws: [B,1,H,W]
    （全程 2->1 通道，计算量极小）
    """
    def __init__(self):
        super().__init__()
        self.conv5 = nn.Conv2d(2, 1, 5, padding=2, bias=True)
        self.conv1 = nn.Conv2d(1, 1, 1, bias=True)
        self.conv7 = nn.Conv2d(1, 1, 7, padding=3, bias=True)

    def forward(self, x):
        gap = torch.mean(x, dim=1, keepdim=True)
        gmp, _ = torch.max(x, dim=1, keepdim=True)
        s = torch.cat([gap, gmp], dim=1)
        s = self.conv7(self.conv1(self.conv5(s)))
        return s


class ChannelAttention(nn.Module):
    """
    Channel attention:
    GAP -> 1x1 -> ReLU -> 1x1
    输出 wc: [B,C,1,1]
    """
    def __init__(self, dim, reduction=8):
        super().__init__()
        hidden = max(1, dim // reduction)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, 1, bias=True),
        )

    def forward(self, x):
        return self.mlp(self.gap(x))


class SGM(nn.Module):
    """
    SGM：
    ws + wc -> w_cos
    与原特征 concat 后，通过 depthwise 7x7 生成像素级门控 g
    """
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)

        # depthwise 7x7，保持 O(C·H·W)
        self.dw_conv = nn.Conv2d(
            2 * dim, dim,
            kernel_size=7,
            padding=3,
            groups=dim,
            bias=True
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B,C,H,W]
        ws = self.sa(x)                     # [B,1,H,W]
        wc = self.ca(x)                     # [B,C,1,1]

        ws = ws.expand(-1, x.size(1), -1, -1)
        wc = wc.expand(-1, -1, x.size(2), x.size(3))
        w_cos = ws + wc                     # [B,C,H,W]

        z = torch.cat([x, w_cos], dim=1)    # [B,2C,H,W]
        g = self.dw_conv(z)                 # depthwise
        return self.sigmoid(g)              # [B,C,H,W]


# =========================
# FMS：Feature Modulation Submodule
# =========================
class FMS(nn.Module):
    """
    FMS：
    AvgPool -> 3x3Conv -> Sigmoid
    + 轻量 η 残差标定
    输出 w_final: [B,C,1,1]
    """
    def __init__(self, dim, eta=1.0):
        super().__init__()
        self.eta = eta
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv3 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w_base = self.sigmoid(self.conv3(self.avg(x)))   # [B,C,1,1]
        x_hat = w_base.detach()                           # 稳定项
        w_final = self.sigmoid(w_base + self.eta * (w_base - x_hat))
        return w_final


# =========================
# Fusion：最终融合模块
# =========================
class MSMF(nn.Module):
    """
    外部接口（Ultralytics 友好）：
        forward(x)  where x = (f_low, f_high)

    流程：
        ctx = f_low + f_high
        g = SGM(ctx)
        w = FMS(ctx)
        g = g * w
        out = f_low * g + f_high * (1 - g)
        1x1Conv -> F_fuse
    """
    def __init__(self, dim, reduction=8, eta=1.0):
        super().__init__()
        self.sgm = SGM(dim, reduction)
        self.fms = FMS(dim, eta)
        self.proj = nn.Conv2d(dim, dim, 1, bias=True)

    def forward(self, x):
        # Ultralytics 会传 list / tuple
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("MSMF expects input as (f_low, f_high)")
        f_low, f_high = x

        ctx = f_low + f_high
        g = self.sgm(ctx)                  # [B,C,H,W]
        w = self.fms(ctx)                  # [B,C,1,1]
        g = g * w                          # 幅值调制

        out = f_low * g + f_high * (1.0 - g)
        return self.proj(out)


