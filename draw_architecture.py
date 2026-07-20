"""
Generate LyaNet architecture diagram.
All modules decomposed to Conv unit level (Conv2d->BN->SiLU as minimal unit).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── 全局字体 ──
plt.rcParams['font.family'] = 'monospace'

# ── 颜色方案 ──────────────────────────────────────────────
COLORS = {
    'conv':      dict(fc='#d5e8d4', ec='#5a9e5a', lw=1.8),
    'concat':    dict(fc='#fff2cc', ec='#c9a83a', lw=1.8),
    'pool':      dict(fc='#e1d5e7', ec='#7b5ea7', lw=1.8),
    'eca':       dict(fc='#f8cecc', ec='#b85450', lw=1.8),
    'gap':       dict(fc='#dae8fc', ec='#4a7eb5', lw=1.8),
    'fc':        dict(fc='#d5e8d4', ec='#5a9e5a', lw=1.8),
    'add':       dict(fc='#ffe6cc', ec='#d79b00', lw=1.8),
    'io':        dict(fc='#f0f0f0', ec='#777777', lw=1.8),
    'dropout':   dict(fc='#f0e6d0', ec='#b89a5a', lw=1.5),
}

# ── 图幅 ─────────────────────────────────────────────────
W, H = 95, 42
MY = 21                 # 主流程 y 中心
fig, ax = plt.subplots(1, 1, figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis('off')
ax.set_aspect('equal')

# ── 卷积单元宽高 ──────────────────────────────────────────
BW, BH = 1.8, 0.85

# ── 工具函数 ──────────────────────────────────────────────
def rbox(x, y, w, h, label, style='conv', fs=7.5, bold=False, z=5):
    """圆角矩形框 + 居中文字"""
    s = COLORS[style]
    b = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle="round,pad=0.08", zorder=z,
                       facecolor=s['fc'], edgecolor=s['ec'], linewidth=s['lw'])
    ax.add_patch(b)
    wt = 'bold' if bold else 'normal'
    ax.text(x, y, label, ha='center', va='center', fontsize=fs,
            fontfamily='monospace', weight=wt, zorder=z+1)

def arow(x1, y1, x2, y2, color='#555555', lw=1.2, z=1):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), zorder=z,
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle='arc3,rad=0'))

def arow_line(x1, y1, x2, y2, color='#888888', lw=0.9, z=1):
    """无箭头连线（用于多路汇聚的水平段）"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), zorder=z,
                arrowprops=dict(arrowstyle='-', color=color, lw=lw,
                                connectionstyle='arc3,rad=0'))

def lbl(x, y, text, fs=7, color='#666666', ha='center'):
    ax.text(x, y, text, ha=ha, va='center', fontsize=fs,
            color=color, fontfamily='monospace', style='italic')

def dash_box(x, y, w, h, label='', fs=9, lw=1.2, ec='#aaaaaa', z=1):
    b = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle="round,pad=0.1", zorder=z,
                       facecolor='none', edgecolor=ec, linewidth=lw,
                       linestyle='--')
    ax.add_patch(b)
    if label:
        ax.text(x, y + h/2 + 0.25, label, ha='center', va='bottom',
                fontsize=fs, fontfamily='monospace', color='#555555',
                style='italic', zorder=z+1)

# ═══════════════════════════════════════════════════════════
# Stage 0: Input
# ═══════════════════════════════════════════════════════════
X0 = 2.0
rbox(X0, MY, 1.8, 1.0, 'Input\n3x224x224', 'io', fs=8, bold=True)
lbl(X0, MY + 0.9, '224x224')
# 记录主流程节点
acc_x = X0 + 0.9     # 当前箭头起点 x

# ═══════════════════════════════════════════════════════════
# Stage 1: Stem  (3 -> 48, k3, s2)
# ═══════════════════════════════════════════════════════════
S1_X = 5.5
rbox(S1_X, MY, 2.8, BH, 'Conv(3->48, k3, s2)', 'conv', fs=7)
arow(acc_x, MY, S1_X - 1.4, MY)
lbl(S1_X, MY + 0.75, '48x112x112')
acc_x = S1_X + 1.4

# ═══════════════════════════════════════════════════════════
# Stage 2: ECACSPBlock 1  (48 -> 96)
# ═══════════════════════════════════════════════════════════
split_x = acc_x + 2.5
block_cx = split_x + 2.0
cat_cx = block_cx + 5.0         # Concat/cv3/ECA 列
flow1_y = MY + 7.0
flow2_y = MY - 6.0
cat_cy = MY + 2.0
cv3_y = MY - 2.2
eca_y = MY - 6.0
# 输出终点 x
ec1_end_x = cat_cx + 5.5

rbox(split_x + 0.3, MY, 0.6, 0.6, '', 'io', fs=6)
arow(acc_x, MY, split_x + 0.3, MY)

# flow1 (上): cv1
rbox(block_cx, flow1_y, BW, BH, 'Conv(48->48,k1)', 'conv', fs=7)
# flow2 (下): cv2 -> cb0 -> cb1  (chain gap=3.5)
rbox(block_cx, flow2_y, BW, BH, 'Conv(48->48,k1)', 'conv', fs=7)
rbox(block_cx + 3.5, flow2_y, BW, BH, 'Conv(48->48,k3)', 'conv', fs=7)
rbox(block_cx + 7.0, flow2_y, BW, BH, 'Conv(48->48,k3)', 'conv', fs=7)

# Concat
rbox(cat_cx, cat_cy, 3.0, BH, 'Concat\n(48+48=96)', 'concat', fs=7.5)
# cv3
rbox(cat_cx, cv3_y, BW, BH, 'Conv(96->96,k1)', 'conv', fs=7)
# ECA
rbox(cat_cx, eca_y, 3.5, BH+0.1, 'ECA: GAP->C1d(3)->s->x', 'eca', fs=7)

# --- 箭头 ---
arow(split_x + 0.3, MY, block_cx, flow1_y + 0.43, lw=1.2)
arow(split_x + 0.3, MY, block_cx, flow2_y + 0.43, lw=1.2)
# flow2 内部
arow(block_cx + 0.9, flow2_y, block_cx + 2.6, flow2_y)
arow(block_cx + 4.4, flow2_y, block_cx + 6.1, flow2_y)
# flow1 -> Concat, cb1 -> Concat
arow(block_cx + 0.9, flow1_y - 0.43, cat_cx - 1.5, cat_cy)
# cb1 输出: 先垂直到 Concat 高度, 再水平入 Concat
ax.plot([block_cx + 7.9, block_cx + 7.9, cat_cx - 1.3],
        [flow2_y + 0.43, cat_cy - 0.1, cat_cy - 0.1],
        color='#555555', lw=1.2, zorder=3)
arow_line(block_cx + 7.9, cat_cy - 0.1, cat_cx - 1.3, cat_cy - 0.1,
          color='#555555', lw=1.2)
# Concat -> cv3
arow(cat_cx, cat_cy - 0.43, cat_cx, cv3_y + 0.43)
# cv3 -> ECA
arow(cat_cx, cv3_y - 0.43, cat_cx, eca_y + 0.48)
# ECA -> 输出
arow(cat_cx + 1.75, eca_y, ec1_end_x, MY, lw=1.5)

dash_box(cat_cx, MY, 13.0, 26, 'ECACSPBlock (48->96)', fs=9)
lbl(ec1_end_x, MY + 0.85, '96x112x112')
acc_x = ec1_end_x

# ═══════════════════════════════════════════════════════════
# Stage 3: Down1  (96 -> 192, k3, s2)
# ═══════════════════════════════════════════════════════════
S3_X = acc_x + 3.0
rbox(S3_X, MY, 2.8, BH, 'Conv(96->192, k3, s2)', 'conv', fs=7)
arow(acc_x, MY, S3_X - 1.4, MY)
lbl(S3_X, MY + 0.85, '192x56x56')
acc_x = S3_X + 1.4

# ═══════════════════════════════════════════════════════════
# Stage 4: MultiBranchStripConvBlock  (192 -> 192)
# ═══════════════════════════════════════════════════════════
mb_split_x = acc_x + 2.5
mb_branch_cx = mb_split_x + 3.0
mb_end_x = mb_branch_cx + 16.0

rbox(mb_split_x + 0.3, MY, 0.6, 0.6, '', 'io', fs=6)
arow(acc_x, MY, mb_split_x + 0.3, MY)

# 5 分支，垂直均匀分布
branch_labels = [
    'Conv(192->39,k3,d=1)\nisotropic',
    'Conv(192->39,k3)\nserial_0',
    'Conv(192->39,k3,d=3)\ndilated',
    'Conv(192->39,(1,21))\nhoriz-strip',
    'Conv(192->39,(21,1))\nvert-strip',
]
# 从 MY+8 到 MY-8，均匀分布 5 个分支
branch_y_top = MY + 8.0
branch_y_bot = MY - 8.0
branch_ys = [branch_y_top - i * 4.0 for i in range(5)]  # gap=4

# serial 分支特殊处理：有两个串联网
serial_idx = 1          # 第2个是串联分支
# 串联分支的子单元坐标
s_y = branch_ys[serial_idx]

for i, (by, blab) in enumerate(zip(branch_ys, branch_labels)):
    if i == serial_idx:
        rbox(mb_branch_cx, by, BW+0.3, BH, blab, 'conv', fs=6.5)
        rbox(mb_branch_cx + 4.5, by, BW, BH, 'Conv(39->39,k3)\nserial_1', 'conv', fs=6.5)
    else:
        rbox(mb_branch_cx, by, BW+0.3, BH, blab, 'conv', fs=6.5)
    arow(mb_split_x + 0.3, MY, mb_branch_cx, by + 0.43, lw=1.0)

# 串联内部箭头
arow(mb_branch_cx + 0.9 + 0.15, s_y, mb_branch_cx + 3.6, s_y)

# Concat (跨垂直范围)
cat_mb_cx = mb_branch_cx + 7.5
cat_mb_cy = MY
# 垂直 tall concat 框
concat_mb_h = branch_y_top - branch_y_bot + 1.2
rbox(cat_mb_cx, cat_mb_cy, 3.0, concat_mb_h,
     'Concat\n5x39=195', 'concat', fs=7.5)

# 5 路 -> Concat
for i, by in enumerate(branch_ys):
    src_x = mb_branch_cx + 1.0 if i != serial_idx else mb_branch_cx + 5.4
    arow_line(src_x, by, cat_mb_cx - 1.5, by, color='#777777', lw=0.8)

# Fusion Conv (在 Concat 右侧中间)
fuse_x = cat_mb_cx + 3.5
fuse_y = cat_mb_cy + 1.5
rbox(fuse_x, fuse_y, BW+0.3, BH, 'Conv(195->192,k1)\nfuse', 'conv', fs=7)
arow(cat_mb_cx + 1.5, fuse_y, fuse_x - BW/2 - 0.15, fuse_y)

# Residual + (在 fusion 右侧)
add_x = fuse_x + BW/2 + 1.5
rbox(add_x, fuse_y, 0.8, 0.8, '+', 'add', fs=10, bold=True)
arow(fuse_x + BW/2 + 0.15, fuse_y, add_x - 0.4, fuse_y)

# 残差连线(虚线) 从 split 绕到下侧的 add
ax.plot([mb_split_x + 0.3, add_x], [MY - 1.5, fuse_y - 1.0],
        color='#999999', lw=1.2, ls='--', zorder=1)
arow_line(mb_split_x + 0.3, MY - 1.5, add_x, fuse_y - 1.0, color='#999999', lw=1.2)

# 残差输出 -> 主流程
arow(add_x + 0.4, fuse_y, mb_end_x, MY, lw=1.5)

dash_box(cat_mb_cx, cat_mb_cy, 16.5, branch_y_top - branch_y_bot + 6,
         'MultiBranchStripConvBlock (192->192)', fs=9)
lbl(mb_end_x, MY + 0.85, '192x56x56')
acc_x = mb_end_x

# ═══════════════════════════════════════════════════════════
# Stage 5: Down2  (192 -> 320, k3, s2)
# ═══════════════════════════════════════════════════════════
S5_X = acc_x + 3.0
rbox(S5_X, MY, 2.8, BH, 'Conv(192->320, k3, s2)', 'conv', fs=7)
arow(acc_x, MY, S5_X - 1.4, MY)
lbl(S5_X, MY + 0.85, '320x28x28')
acc_x = S5_X + 1.4

# ═══════════════════════════════════════════════════════════
# Stage 6: ECACSPBlock 2  (320 -> 384)
# ═══════════════════════════════════════════════════════════
split2_x = acc_x + 2.5
block2_cx = split2_x + 2.0
cat2_cx = block2_cx + 5.0
cat2_cy = MY + 2.0
cv3_y2 = MY - 2.2
eca_y2 = MY - 6.0
ec2_end_x = cat2_cx + 5.5

rbox(split2_x + 0.3, MY, 0.6, 0.6, '', 'io', fs=6)
arow(acc_x, MY, split2_x + 0.3, MY)

rbox(block2_cx, MY + 7.0, BW, BH, 'Conv(320->192,k1)', 'conv', fs=7)
rbox(block2_cx, MY - 6.0, BW, BH, 'Conv(320->192,k1)', 'conv', fs=7)
rbox(block2_cx + 3.5, MY - 6.0, BW, BH, 'Conv(192->192,k3)', 'conv', fs=7)
rbox(block2_cx + 7.0, MY - 6.0, BW, BH, 'Conv(192->192,k3)', 'conv', fs=7)

rbox(cat2_cx, cat2_cy, 3.0, BH, 'Concat\n(192+192=384)', 'concat', fs=7.5)
rbox(cat2_cx, cv3_y2, BW, BH, 'Conv(384->384,k1)', 'conv', fs=7)
rbox(cat2_cx, eca_y2, 3.5, BH+0.1, 'ECA: GAP->C1d(5)->s->x', 'eca', fs=7)

arow(split2_x + 0.3, MY, block2_cx, MY + 6.57, lw=1.2)
arow(split2_x + 0.3, MY, block2_cx, MY - 5.57, lw=1.2)
arow(block2_cx + 0.9, MY - 6.0, block2_cx + 2.6, MY - 6.0)
arow(block2_cx + 4.4, MY - 6.0, block2_cx + 6.1, MY - 6.0)
arow(block2_cx + 0.9, MY + 6.57, cat2_cx - 1.5, cat2_cy)
# cb1 输出: 先垂直再水平
ax.plot([block2_cx + 7.9, block2_cx + 7.9, cat2_cx - 1.3],
        [MY - 5.57, cat2_cy - 0.1, cat2_cy - 0.1],
        color='#555555', lw=1.2, zorder=3)
arow_line(block2_cx + 7.9, cat2_cy - 0.1, cat2_cx - 1.3, cat2_cy - 0.1,
          color='#555555', lw=1.2)
arow(cat2_cx, cat2_cy - 0.43, cat2_cx, cv3_y2 + 0.43)
arow(cat2_cx, cv3_y2 - 0.43, cat2_cx, eca_y2 + 0.48)
arow(cat2_cx + 1.75, eca_y2, ec2_end_x, MY, lw=1.5)

dash_box(cat2_cx, MY, 13.0, 26, 'ECACSPBlock (320->384)', fs=9)
lbl(ec2_end_x, MY + 0.85, '384x28x28')
acc_x = ec2_end_x

# ═══════════════════════════════════════════════════════════
# Stage 7: Down3  (384 -> 512, k3, s2)
# ═══════════════════════════════════════════════════════════
S7_X = acc_x + 3.0
rbox(S7_X, MY, 2.8, BH, 'Conv(384->512, k3, s2)', 'conv', fs=7)
arow(acc_x, MY, S7_X - 1.4, MY)
lbl(S7_X, MY + 0.85, '512x14x14')
acc_x = S7_X + 1.4

# ═══════════════════════════════════════════════════════════
# Stage 8: SPPF  (512 -> 512)
# ═══════════════════════════════════════════════════════════
sp_start_x = acc_x + 2.5
sp_cx = sp_start_x + 3.0
sp_end_x = sp_cx + 8.5

# cv1 (无 SiLU) 在上方
cv1_y = MY + 6.0
rbox(sp_cx, cv1_y, BW+0.4, BH, 'Conv(512->256,k1)\n(no SiLU)', 'conv', fs=7)
arow(acc_x, MY, sp_cx, cv1_y - 0.43)

# MaxPool  x3 串联 (在 cv1 正下方)
pool_y_start = MY + 1.5
pool_ys = [pool_y_start - i * 2.2 for i in range(3)]
rf_labels = ['RF=5x5', 'RF=9x9', 'RF=13x13']
for i in range(3):
    rbox(sp_cx, pool_ys[i], BW+0.4, BH,
         f'MaxPool(k=5,s=1,p=2)\n{rf_labels[i]}', 'pool', fs=6.5)

# cv1 -> pool0 -> pool1 -> pool2 串联
arow(sp_cx, cv1_y - 0.43, sp_cx, pool_ys[0] + 0.43)
for i in range(2):
    arow(sp_cx, pool_ys[i] - 0.43, sp_cx, pool_ys[i+1] + 0.43)

# Concat (4 路)
cat_sp_cx = sp_cx + 4.5
cat_sp_cy = MY - 2.0
cat_sp_h = 4.0
rbox(cat_sp_cx, cat_sp_cy, 3.0, cat_sp_h,
     'Concat\n4x256=1024', 'concat', fs=7)

# cv1 直连 -> Concat (水平线)
arow_line(sp_cx + 1.1, cv1_y, cat_sp_cx - 1.5, cv1_y, color='#777777', lw=0.9)
# 各 pool -> Concat (水平线)
for i in range(3):
    arow_line(sp_cx + 1.1, pool_ys[i], cat_sp_cx - 1.5, pool_ys[i],
              color='#777777', lw=0.9)

# cv2
cv2_x = cat_sp_cx + 3.5
cv2_y = cat_sp_cy
rbox(cv2_x, cv2_y, BW+0.4, BH, 'Conv(1024->512,k1)', 'conv', fs=7)
arow(cat_sp_cx + 1.5, cv2_y + 0.3, cv2_x - BW/2 - 0.2, cv2_y + 0.3)

# 输出 -> 主流程
arow(cv2_x + BW/2 + 0.2, cv2_y + 0.3, sp_end_x, MY, lw=1.5)

dash_box(cat_sp_cx, MY - 2.0, 15.5, 21, 'SPPF (512->512)', fs=9)
lbl(sp_end_x, MY + 0.85, '512x14x14')
acc_x = sp_end_x

# ═══════════════════════════════════════════════════════════
# Stage 9: GAP
# ═══════════════════════════════════════════════════════════
S9_X = acc_x + 3.0
rbox(S9_X, MY, BW, BH, 'GAP\n512x1x1', 'gap', fs=7.5)
arow(acc_x, MY, S9_X - BW/2, MY)
lbl(S9_X, MY + 0.85, '512x1x1')
acc_x = S9_X + BW/2

# ═══════════════════════════════════════════════════════════
# Stage 10: Dropout
# ═══════════════════════════════════════════════════════════
S10_X = acc_x + 2.0
rbox(S10_X, MY, BW - 0.3, BH - 0.1, 'Dropout\np=0.2', 'dropout', fs=7)
arow(acc_x, MY, S10_X - (BW - 0.3)/2, MY)
acc_x = S10_X + (BW - 0.3)/2

# ═══════════════════════════════════════════════════════════
# Stage 11: FC
# ═══════════════════════════════════════════════════════════
S11_X = acc_x + 1.5
rbox(S11_X, MY, BW - 0.3, BH, 'FC\n512->3', 'fc', fs=7.5, bold=True)
arow(acc_x, MY, S11_X - (BW - 0.3)/2, MY)
acc_x = S11_X + (BW - 0.3)/2

# ═══════════════════════════════════════════════════════════
# Stage 12: Output
# ═══════════════════════════════════════════════════════════
S12_X = acc_x + 2.0
rbox(S12_X, MY, 2.0, 1.1, 'Output\nN-class logits', 'io', fs=8, bold=True)
arow(acc_x, MY, S12_X - 1.0, MY)

# ═══════════════════════════════════════════════════════════
# 顶部维度 + 通道演化标注
# ═══════════════════════════════════════════════════════════
anno_y_hw = MY + 14.0
anno_y_ch = MY + 15.2

# 各阶段输出 x 坐标（手动按布局计算）
# Stage 0: X0, Stage 1: S1_X,  EC1 输出: ec1_end_x,  Down1: S3_X
# MB 输出: mb_end_x,  Down2: S5_X,  EC2 输出: ec2_end_x
# Down3: S7_X,  SPPF 输出: sp_end_x,  GAP: S9_X
annotations = [
    (X0,           '224x224',  '-'),
    (S1_X,         '112x112',  'C=48'),
    (ec1_end_x,    '112x112',  'C=96'),
    (S3_X,         '56x56',    'C=192'),
    (mb_end_x,     '56x56',    'C=192'),
    (S5_X,         '28x28',    'C=320'),
    (ec2_end_x,    '28x28',    'C=384'),
    (S7_X,         '14x14',    'C=512'),
    (sp_end_x,     '14x14',    'C=512'),
    (S9_X,         '1x1',      'C=512'),
]

for dx, hw, ch in annotations:
    lbl(dx, anno_y_hw, f'HxW: {hw}', fs=7, color='#3377aa')
    lbl(dx, anno_y_ch, ch, fs=6.5, color='#cc5555')

# ═══════════════════════════════════════════════════════════
# 图例
# ═══════════════════════════════════════════════════════════
legend_items = [
    ('conv',    'Conv (Conv2d+BN+SiLU)'),
    ('concat',  'Concat'),
    ('pool',    'MaxPool'),
    ('eca',     'ECA Attention'),
    ('gap',     'GAP'),
    ('add',     'Residual Add'),
    ('io',      'Input / Output'),
    ('dropout', 'Dropout'),
]
lx_start = 1.0
ly = H - 2.5
for i, (style, desc) in enumerate(legend_items):
    li_x = lx_start + i * 8.0
    rbox(li_x, ly, 1.0, 0.5, '', style, fs=5, z=8)
    ax.text(li_x + 0.8, ly, desc, fontsize=7.5, va='center',
            fontfamily='monospace', color='#333333', zorder=8)

# ═══════════════════════════════════════════════════════════
# 保存
# ═══════════════════════════════════════════════════════════
out_path = r'e:\Programming-related Engineering Project\1.tianquanzhineng\Patent project\ji guang hanjie\1111\network_architecture.png'
plt.tight_layout(pad=0.5)
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Architecture diagram saved: {out_path}')
