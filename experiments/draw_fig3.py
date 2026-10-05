# -*- coding: utf-8 -*-
"""重绘 Figure 3（自动激活准则链流程图）：Times New Roman、大字号、紧凑布局。
替换 paper/figures_spie/fig3_chain.png。"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=300)
ax.set_xlim(0, 60)
ax.set_ylim(0, 56)
ax.axis('off')

C_EDGE = '#222222'
C_MAIN = '#f4f7fb'
C_REJECT = '#fbeeee'
C_ANNO = '#fdf6e3'
FS_T = 12      # 框标题
FS_B = 10.5    # 框正文
FS_D = 10      # 菱形
FS_L = 9.5     # 边标签


def box(x, y, w, h, title, rest=None, fc='white', dashed=False, fs_t=FS_T, fs_b=FS_B):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.25,rounding_size=0.7',
                                fc=fc, ec=C_EDGE, lw=1.2,
                                linestyle=(0, (4, 2)) if dashed else 'solid'))
    if rest:
        ax.text(x + w / 2, y + h - 0.4, title, ha='center', va='top', fontsize=fs_t,
                fontweight='bold', linespacing=1.25)
        n = title.count('\n') + 1
        ax.text(x + w / 2, y + h - 0.4 - 2.4 * n, rest, ha='center', va='top',
                fontsize=fs_b, linespacing=1.28)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha='center', va='center', fontsize=fs_t,
                fontweight='bold' if len(title) < 30 else 'normal', linespacing=1.25)


def diamond(cx, cy, w, h, text, fs=FS_D):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, closed=True, fc='white', ec=C_EDGE, lw=1.2))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs, linespacing=1.22)


def arrow(x1, y1, x2, y2, dashed=False, label=None, lpos=(0.5, 0), fs=FS_L, color=C_EDGE):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=13,
                                 lw=1.2 if not dashed else 1.1,
                                 linestyle=(0, (4, 2)) if dashed else 'solid',
                                 color=color, shrinkA=1.5, shrinkB=1.5))
    if label:
        lx = x1 + (x2 - x1) * lpos[0] + lpos[1]
        ly = y1 + (y2 - y1) * lpos[0] + (0.9 if y2 == y1 else 0)
        ax.text(lx, ly, label, ha='center', va='center', fontsize=fs, style='italic',
                color='#333333', bbox=dict(fc='white', ec='none', pad=0.5))


X0, W = 3.0, 30.0      # 主流程列
CX = X0 + W / 2        # 主流程中心 x = 18

# 顶部：实例
box(X0 + 5, 51.5, 20, 3.6, 'problem instance $X$', fc=C_MAIN, fs_t=11)
arrow(CX, 51.5, CX, 48.6)

# S1
box(X0, 41.8, W, 6.8, 'S1 — Union-probe profiling',
    'passive corner archive, 3 seeds;\nclassify gap: generative vs selective',
    fc=C_MAIN, fs_t=11.5)
arrow(CX, 41.8, CX, 38.6)

# 候选菱形
diamond(CX, 34.0, 27, 9.0, 'S1 gain: $\\Delta$IGD$^+$ \u2264 \u22125% (3/3)\nor \u2264 \u22123% (\u22652/3) ?')
arrow(CX, 29.5, CX, 26.8, label='candidate', lpos=(0.5, 1.6))

# S2（三条件合一，紧凑）
box(X0, 17.8, W, 9.0, 'S2 — Three-condition validation',
    '10-seed same-process paired off/on:\n(1) IGD & IGD$^+$ improve, Wilcoxon $p$ < 0.05;\n(2) \u2265 8/10 seeds improve;  (3) worst seed \u2264 1.05\u00d7',
    fc=C_MAIN, fs_t=11.5, fs_b=10)
arrow(CX, 17.8, CX, 15.2, label='all pass', lpos=(0.5, 1.6))

# S3
box(X0 + 3, 9.4, W - 6, 5.8, 'S3 — Regression gate', 'zero regression', fc=C_MAIN, fs_t=11.5)
arrow(CX, 9.4, CX, 6.8, label='pass', lpos=(0.5, 1.6))

# 激活
box(X0 + 3, 2.2, W - 6, 4.6, 'activate module (write override)', fc='#e8f3e8', fs_t=10.5)

# 右侧：reject
RX, RW = 41.5, 15.5
box(RX, 30.2, RW, 4.6, 'reject (keep off)', fc=C_REJECT, fs_t=11)
arrow(X0 + W + 1.0, 34.0, RX, 32.5, label='below threshold', lpos=(0.42, 0), fs=9)
arrow(X0 + W, 22.3, RX - 1.5, 32.5, label='any condition fails', lpos=(0.35, 0), fs=9)
arrow(X0 + W - 3, 12.3, RX - 1.0, 32.9, label='fail', lpos=(0.30, 0), fs=9)

# 审计记录（底部，reject 虚线汇入）
box(RX - 3, 2.2, RW + 6, 4.6, 'decision log + audit trail', fc='#f0f0f0', fs_t=10.5)
arrow(RX + RW / 2, 30.2, RX + RW / 2, 6.8, dashed=True)

# activate → decision log 连接
seg_y = 4.5
ax.add_patch(FancyArrowPatch((X0 + 3 + (W - 6), seg_y), (RX - 3, seg_y), arrowstyle='-|>',
                             mutation_scale=13, lw=1.2, color=C_EDGE, shrinkA=0, shrinkB=0))

# 右上注释：gap nature
box(41.5, 43.4, 17.5, 8.0, 'gap nature:',
    'generative \u2192 operator response\n(escape mode)\nselective \u2192 archive-repairable',
    fc=C_ANNO, dashed=True, fs_t=10, fs_b=9)
arrow(41.5, 45.6, X0 + W, 44.5, dashed=True)

plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
out = os.path.join(ROOT, 'paper', 'figures_spie', 'fig3_chain.png')
plt.savefig(out, dpi=300)
print('saved', out)