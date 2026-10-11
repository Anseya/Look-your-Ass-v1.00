LyaNet
面向工业视觉分类任务的轻量级卷积神经网络，融合 CSP 跨阶段局部连接、ECA 通道注意力、多分支条带卷积与 SPPF 空间金字塔池化，以最小单元 Conv2d → BN → SiLU 构建全网络。
A lightweight CNN for industrial visual classification, integrating CSP cross-stage partial connections, ECA channel attention, multi-branch strip convolution, and SPPF spatial pyramid pooling. The entire network is built on the minimal unit Conv2d → BN → SiLU.

目录 / Table of Contents
网络结构 / Architecture
核心模块 / Core Modules
张量形状流转 / Tensor Shape Flow
安装与依赖 / Installation
快速使用 / Quick Start
参数量与性能 / Parameters & Performance
架构图生成 / Architecture Diagram
设计要点 / Design Highlights

网络结构 / Architecture
Input (3×224×224)
  │
  ├─ Stem        Conv(3→48, k=3, s=2)          → 48×112×112
  ├─ Stage 1     ECACSPBlock(48→96)            → 96×112×112
  ├─ Down 1      Conv(96→192, k=3, s=2)        → 192×56×56
  ├─ Stage 2     MultiBranchStripConvBlock     → 192×56×56
  ├─ Down 2      Conv(192→320, k=3, s=2)       → 320×28×28
  ├─ Stage 3     ECACSPBlock(320→384)          → 384×28×28
  ├─ Down 3      Conv(384→512, k=3, s=2)       → 512×14×14
  ├─ SPPF        SPPF(512→512, k=5)            → 512×14×14
  ├─ GAP         AdaptiveAvgPool2d(1)          → 512×1×1
  ├─ Dropout     p=0.2
  └─ FC          Linear(512→num_classes)


核心模块 / Core Modules
Conv（最小单元 / Minimal Unit）
Conv2d(c1, c2, k, s, p, groups, dilation, bias=False)
  → BatchNorm2d(c2)
  → SiLU()

act=False 时激活替换为 Identity（SPPF 的 cv1 使用）。
When act=False, activation is replaced with Identity (used by SPPF's cv1).
autopad 自动处理 dilation 下的等效 padding。
autopad automatically computes equivalent padding under dilation.
ECA（Efficient Channel Attention）
全局平均池化 → 1D 卷积（kernel size 由通道数自适应计算）→ Sigmoid → 逐通道加权。
Global average pooling → 1D convolution (kernel size adaptively derived from channel count) → Sigmoid → channel-wise scaling.
核大小公式 / Kernel size formula：k = t if t%2==1 else t+1，其中 / where t = |(log₂(C) + b) / γ|，默认 / defaults γ=2, b=1。
本网络中 / In this network：C=96 → k=3；C=384 → k=5。
ECACSPBlock
Input ──┬── cv1 (1×1) ──────────────────────┐
        │                                    ├─ Concat → cv3 (1×1) → ECA → Output
        └── cv2 (1×1) → Conv(3×3) → Conv(3×3)┘

两路分支通道数均为 c2//2，拼接后经 1×1 卷积整合，再由 ECA 重标定通道权重。
Both branches output c2//2 channels; after concatenation, a 1×1 conv integrates features, followed by ECA channel recalibration.
无残差连接，允许通道数变化（c1 → c2）。
No residual connection; allows channel count change (c1 → c2).
MultiBranchStripConvBlock
五分支并行特征提取，融合后残差相加。 Five-branch parallel feature extraction with residual addition after fusion.
分支 Branch
卷积配置 Conv Config
感受野特征 Receptive Field
1
Conv(c1→c_, k=3, d=1)
各向同性标准卷积 / Isotropic standard conv
2
Conv(k=3) → Conv(k=3) 串联 / serial
堆叠扩大感受野 / Stacked enlarged RF
3
Conv(c1→c_, k=3, d=3)
空洞卷积，大感受野 / Dilated conv, large RF
4
Conv(c1→c_, k=(1, 21))
水平条带卷积 / Horizontal strip conv
5
Conv(c1→c_, k=(21, 1))
垂直条带卷积 / Vertical strip conv

每分支输出 c_ = ceil(c2 / 5) 通道，五路拼接后经 Conv(5c_ → c2, 1×1) 融合。
Each branch outputs c_ = ceil(c2 / 5) channels; five branches are concatenated and fused via Conv(5c_ → c2, 1×1).
融合输出与输入残差相加，要求输入输出形状一致（c1 = c2）。
Fused output is added to the input residually, requiring identical input/output shapes (c1 = c2).
条带卷积专门捕获长距离水平/垂直方向依赖，适用于焊缝、纹理等线性结构。
Strip convolutions specifically capture long-range horizontal/vertical dependencies, suited for linear structures like weld seams and textures.
SPPF（Spatial Pyramid Pooling - Fast）
Input → cv1 (1×1, no SiLU) ──┬───────────────────────┐
                              ├─ MaxPool(k=5,s=1,p=2) ─┤
                              │                          ├─ MaxPool → ┤
                              │                          │            ├─ MaxPool → ┤
                              └──────────────────────────┴────────────┴────────────┘
                                                                           │
                              Concat(4×256=1024) → cv2 (1×1) → Output

三个 MaxPool 串联，等效感受野分别为 5×5、9×9、13×13。
Three MaxPool layers in series, with equivalent receptive fields of 5×5, 9×9, and 13×13.
四路（cv1 输出 + 三级池化）拼接后经 1×1 卷积降回目标通道。
Four paths (cv1 output + three pooling stages) are concatenated, then a 1×1 conv reduces back to the target channel count.

张量形状流转 / Tensor Shape Flow
以 batch=2, input=3×224×224, num_classes=3 为例。 Example with batch=2, input=3×224×224, num_classes=3.
阶段 Stage
输出形状 Output Shape
Input
(2, 3, 224, 224)
Stem
(2, 48, 112, 112)
Stage1 (ECACSP)
(2, 96, 112, 112)
Down1
(2, 192, 56, 56)
Stage2 (MultiBranch)
(2, 192, 56, 56)
Down2
(2, 320, 28, 28)
Stage3 (ECACSP)
(2, 384, 28, 28)
Down3
(2, 512, 14, 14)
SPPF
(2, 512, 14, 14)
GAP → Flatten
(2, 512)
Dropout
(2, 512)
FC → Logits
(2, 3)


安装与依赖 / Installation
pip install torch matplotlib numpy

Python ≥ 3.8
PyTorch ≥ 1.10

快速使用 / Quick Start
基本推理 / Basic Inference
import torch
from lyanet import LyaNet

model = LyaNet(num_classes=3, dropout=0.2, strip_k=21)
model.eval()

x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    logits = model(x)
print(logits.shape)  # torch.Size([1, 3])

打印各层形状 / Print Layer Shapes
LyaNet._print_shapes = True
model = LyaNet(num_classes=3)
_ = model(torch.randn(2, 3, 224, 224))

查看参数量与 ECA 核大小 / Check Parameters & ECA Kernel Sizes
python lyanet.py

输出示例 / Sample output：
2-class | X.XXXM params
ECA: C=96->k=3, C=384->k=5
Shape: (2, 3, 224, 224) -> (2, 2)
3-class | X.XXXM params
ECA: C=96->k=3, C=384->k=5
Shape: (2, 3, 224, 224) -> (2, 3)


参数量与性能 / Parameters & Performance
配置 Config
参数量 Params
输入 Input
输出 Output
num_classes=2
运行 python lyanet.py 查看 / Run to check
3×224×224
(B, 2)
num_classes=3
运行 python lyanet.py 查看 / Run to check
3×224×224
(B, 3)

权重初始化 / Weight initialization：
Conv2d：Kaiming Uniform（fan_out, relu）
BatchNorm2d：weight=1.0, bias=0.0
Linear：Normal(0, 0.01), bias=0.0

架构图生成 / Architecture Diagram
项目包含 draw_architecture.py，使用 matplotlib 生成完整网络架构图（分解到 Conv 单元级别）。 The project includes draw_architecture.py, which generates a full architecture diagram using matplotlib (decomposed down to the Conv unit level).
python draw_architecture.py

输出 / Output：network_architecture.png
图中包含 / The diagram includes：
全部模块逐层展开（Conv / Concat / MaxPool / ECA / GAP / FC / Dropout）
All modules expanded layer by layer (Conv / Concat / MaxPool / ECA / GAP / FC / Dropout)
多分支条带卷积的五路并行结构
Five-branch parallel structure of MultiBranchStripConvBlock
ECACSPBlock 的双分支 + ECA 结构
Dual-branch + ECA structure of ECACSPBlock
SPPF 的四级池化拼接结构
Four-path pooling-concat structure of SPPF
顶部通道数与空间尺寸演化标注
Top annotations for channel count and spatial dimension evolution
残差连接（虚线）与图例
Residual connections (dashed lines) and legend

设计要点 / Design Highlights
最小单元统一 / Unified Minimal Unit：全网络所有卷积均为 Conv2d → BN → SiLU，便于推理融合（forward_fuse）与量化部署。
All convolutions follow Conv2d → BN → SiLU, enabling inference fusion (forward_fuse) and quantization deployment.
CSP 分流降本 / CSP Split for Efficiency：ECACSPBlock 将通道一分为二，仅一路经过深度卷积，在保持表达能力的同时降低计算量。
ECACSPBlock splits channels in half; only one branch passes through deep convolutions, reducing compute while preserving representational capacity.
条带卷积捕获线性特征 / Strip Convolution for Linear Features：水平/垂直 1×21 与 21×1 卷积对焊缝、划痕等长条状缺陷具有强适配性。
Horizontal/vertical 1×21 and 21×1 convolutions are highly suited for elongated defects such as weld seams and scratches.
ECA 轻量注意力 / Lightweight ECA Attention：仅用 1D 卷积实现通道注意力，参数量可忽略，避免 SE 模块的降维信息损失。
Channel attention implemented with only 1D convolution, negligible parameters, avoiding the dimensionality-reduction information loss of SE modules.
SPPF 多尺度聚合 / SPPF Multi-Scale Aggregation：串联池化以极低开销实现 5×5 / 9×9 / 13×13 多尺度感受野融合。
Series pooling achieves 5×5 / 9×9 / 13×13 multi-scale receptive field fusion at minimal cost.
残差约束 / Residual Constraint：MultiBranchStripConvBlock 内部断言输入输出形状一致，确保残差加法安全。
MultiBranchStripConvBlock internally asserts identical input/output shapes to ensure safe residual addition.
