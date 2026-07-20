import math
import torch
import torch.nn as nn


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d),
                              groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else \
                   act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 == 1 else t + 1
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = self.sig(self.conv(
            self.pool(x).squeeze(-1).transpose(-1, -2)
        ).transpose(-1, -2).unsqueeze(-1))
        return x * y


class ECACSPBlock(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        ch = c2 // 2
        self.cv1 = Conv(c1, ch, 1)
        self.cv2 = Conv(c1, ch, 1)
        self.conv_branch = nn.Sequential(Conv(ch, ch, 3), Conv(ch, ch, 3))
        self.cv3 = Conv(ch * 2, c2, 1)
        self.eca = ECA(c2)

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.conv_branch(self.cv2(x))
        return self.eca(self.cv3(torch.cat([y1, y2], dim=1)))


class MultiBranchStripConvBlock(nn.Module):
    def __init__(self, c1, c2=None, strip_k=21):
        super().__init__()
        c2 = c2 or c1
        n_branches = 5
        c_ = max(8, (c2 + n_branches - 1) // n_branches)

        self.branch_d1 = Conv(c1, c_, 3, 1, d=1)
        self.branch_d2 = nn.Sequential(
            Conv(c1, c_, 3, 1, d=1),
            Conv(c_, c_, 3, 1, d=1))
        self.branch_d3 = Conv(c1, c_, 3, 1, d=3)
        self.branch_h = Conv(c1, c_, (1, strip_k), 1)
        self.branch_v = Conv(c1, c_, (strip_k, 1), 1)
        self.fusion = Conv(c_ * n_branches, c2, 1, 1)

    def forward(self, x):
        b1 = self.branch_d1(x)
        b2 = self.branch_d2(x)
        b3 = self.branch_d3(x)
        bh = self.branch_h(x)
        bv = self.branch_v(x)
        fused = self.fusion(torch.cat([b1, b2, b3, bh, bv], dim=1))
        assert fused.shape == x.shape, \
            f"MultiBranchStripConvBlock: shape mismatch {fused.shape} vs {x.shape}"
        return fused + x


class LyaNet(nn.Module):
    _print_shapes = False

    def __init__(self, num_classes=3, dropout=0.2, strip_k=21):
        super().__init__()
        self.stem   = Conv(3, 48, 3, 2)
        self.stage1 = ECACSPBlock(48, 96)
        self.down1  = Conv(96, 192, 3, 2)
        self.stage2 = MultiBranchStripConvBlock(192, 192, strip_k=strip_k)
        self.down2  = Conv(192, 320, 3, 2)
        self.stage3 = ECACSPBlock(320, 384)
        self.down3  = Conv(384, 512, 3, 2)
        self.sppf   = SPPF(512, 512)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.head   = nn.Linear(512, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        v = self._print_shapes
        if v: print(f"  Input:          {tuple(x.shape)}")
        x = self.stem(x)
        if v: print(f"  Stem:           {tuple(x.shape)}")
        x = self.stage1(x)
        if v: print(f"  Stage1(ECACSP): {tuple(x.shape)}")
        x = self.down1(x)
        if v: print(f"  Down1:          {tuple(x.shape)}")
        x = self.stage2(x)
        if v: print(f"  Stage2(MBC):    {tuple(x.shape)}")
        x = self.down2(x)
        if v: print(f"  Down2:          {tuple(x.shape)}")
        x = self.stage3(x)
        if v: print(f"  Stage3(ECACSP): {tuple(x.shape)}")
        x = self.down3(x)
        if v: print(f"  Down3:          {tuple(x.shape)}")
        x = self.sppf(x)
        if v: print(f"  SPPF:           {tuple(x.shape)}")
        x = self.pool(x).flatten(1)
        if v: print(f"  GAP->Flatten:   {tuple(x.shape)}")
        x = self.dropout(x)
        if v: print(f"  Dropout({self.head.in_features}): {tuple(x.shape)}")
        x = self.head(x)
        if v: print(f"  FC->Logits:     {tuple(x.shape)}")
        return x


if __name__ == "__main__":
    for nc in (2, 3):
        model = LyaNet(num_classes=nc)
        params_m = sum(p.numel() for p in model.parameters()
                        if p.requires_grad) / 1e6
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            y = model(x)

        eca_info = []
        for name, mod in model.named_modules():
            if isinstance(mod, ECA):
                k_val = mod.conv.kernel_size[0]
                for pname, pmod in model.named_modules():
                    if isinstance(pmod, ECACSPBlock) and pmod.eca is mod:
                        ch = pmod.cv3.conv.out_channels
                        eca_info.append(f"C={ch}->k={k_val}")
                        break

        print(f"{nc}-class | {params_m:.3f}M params")
        print(f"ECA: {', '.join(eca_info)}")
        print(f"Shape: {tuple(x.shape)} -> {tuple(y.shape)}")

    LyaNet._print_shapes = True
    model3 = LyaNet(num_classes=3)
    with torch.no_grad():
        _ = model3(torch.randn(2, 3, 224, 224))
    LyaNet._print_shapes = False
